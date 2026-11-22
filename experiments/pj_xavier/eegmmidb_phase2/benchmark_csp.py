"""Task-specific regularized CSP + LDA. fit uses train/val only; no test scores."""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import time

import numpy as np
from scipy.linalg import eigh
from scipy.special import expit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from threadpoolctl import threadpool_limits

import benchmark_common as common


def trial_covariance(eeg):
    x=np.asarray(eeg,np.float64)
    common.require(x.ndim==2 and np.isfinite(x).all(),'nonfinite/malformed EEG')
    x=x-x.mean(axis=1,keepdims=True)
    covariance=x@x.T
    trace=float(np.trace(covariance))
    common.require(trace>0 and math.isfinite(trace),'flat EEG cannot fit CSP')
    return covariance/trace


def fit_filters(covariances, y, shrinkage=.1):
    y=np.asarray(y,int)
    common.require(set(y.tolist())=={0,1},'CSP needs both train classes')
    matrices=[]
    for label in (0,1):
        cov=np.asarray(covariances)[y==label].mean(axis=0)
        channels=len(cov)
        matrices.append((1-shrinkage)*cov+shrinkage*np.trace(cov)/channels*np.eye(channels))
    eigenvalues,eigenvectors=eigh(matrices[1],matrices[0]+matrices[1],check_finite=True)
    order=[]
    for i in range(len(eigenvalues)//2):order.extend([len(eigenvalues)-1-i,i])
    if len(eigenvalues)%2:order.append(len(eigenvalues)//2)
    return eigenvectors[:,order].T


def features(eeg, filters):
    x=np.asarray(eeg,np.float64); x=x-x.mean(axis=1,keepdims=True)
    projected=np.asarray(filters)@x
    return np.log(np.maximum(np.mean(projected**2,axis=1),1e-30))


def subject_selection_score(rows, probabilities):
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[(row['dataset'],row['subject'])].append(i)
    scores=[]
    labels=common.TASKS[rows[0]['task_family']]
    y=np.array([labels.index(r['label']) for r in rows])
    for ids in groups.values():
        value=common.metrics(y[ids],probabilities[ids])['balanced_accuracy']
        if value is not None:scores.append(value)
    common.require(scores,'no validation subject has both classes')
    return float(np.mean(scores))


def fit(root, seal_path, expected_seal, output, max_seconds=1200):
    started=time.monotonic(); output=Path(output)
    common.require(common.sha256(seal_path)==expected_seal,'execution seal mismatch')
    seal=common.read_json(seal_path)
    plan_path=Path(__file__).with_name('protocol.benchmark.v1.json')
    common.require(common.sha256(plan_path)==seal['analysis_plan_sha256'],'analysis plan changed')
    common.require(common.sha256(__file__)==seal['source_sha256'][Path(__file__).name],'CSP implementation changed')
    common.require(common.sha256(common.__file__)==seal['source_sha256'][Path(common.__file__).name],'shared checks changed')
    rows,artifacts,audit=common.verify_prepared(root,seal)
    common.require(not output.exists(),'refuse to overwrite or refit a prior run')
    output.mkdir(parents=True)
    common.save_json(output/'execution_seal.json',seal)
    plan=common.read_json(plan_path)
    store=common.EEGStore(root); selections={}
    train_ids=sorted(r['id'] for r in rows if r['split']=='train'); val_ids=sorted(r['id'] for r in rows if r['split']=='val')
    common.save_json(output/'training_membership.json',{'manifest_sha256':seal['prepared_manifest_sha256'],'training_ids':train_ids,'validation_ids':val_ids,
        'test_ids':sorted(r['id'] for r in rows if r['split']=='test'),'test_used_for_tuning':False,'model_selection_partition':'val'})
    with threadpool_limits(limits=2):
        for task,labels in common.TASKS.items():
            train=[r for r in rows if r['split']=='train' and r['task_family']==task]
            val=[r for r in rows if r['split']=='val' and r['task_family']==task]
            common.require(train and val,'missing task train/val')
            y=np.array([labels.index(r['label']) for r in train])
            cov=[]
            for r in train:
                common.require(time.monotonic()-started<max_seconds,'CSP train budget expired')
                cov.append(trial_covariance(store.get(r)))
            filters=fit_filters(cov,y,plan['csp']['class_covariance_shrinkage'])
            x=np.stack([features(store.get(r),filters[:16]) for r in train])
            v=np.stack([features(store.get(r),filters[:16]) for r in val])
            candidates=[]; fitted=[]
            for k in plan['csp']['components']:
                common.require(time.monotonic()-started<max_seconds,'CSP selection budget expired')
                lda=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=np.array([.5,.5])).fit(x[:,:k],y)
                probabilities=lda.predict_proba(v[:,:k]); value=subject_selection_score(val,probabilities)
                candidates.append({'components':k,'validation_mean_subject_balanced_accuracy':value})
                fitted.append(lda)
            selected=max(range(len(candidates)),key=lambda i:(candidates[i]['validation_mean_subject_balanced_accuracy'],-candidates[i]['components']))
            k=candidates[selected]['components']; lda=fitted[selected]
            path=output/(task+'.npz')
            with path.open('xb') as stream:np.savez(stream,filters=filters[:k],coef=lda.coef_,intercept=lda.intercept_,classes=lda.classes_)
            reconstructed=expit(v[:,:k]@lda.coef_[0]+lda.intercept_[0])
            common.require(np.allclose(reconstructed,lda.predict_proba(v[:,:k])[:,1],atol=1e-12,rtol=1e-12),'saved linear inference differs from fitted LDA')
            selections[task]={'candidates':candidates,'selected_components':k,'selected_validation_mean_subject_balanced_accuracy':candidates[selected]['validation_mean_subject_balanced_accuracy'],
                              'artifact_path':path.name,'artifact_sha256':common.sha256(path),'training_n':len(train),'validation_n':len(val)}
    summary={'schema':'xavier-phase2-CSP-LDA-trained-v1','status':'completed_train_val_only','selections':selections,'execution_seal_sha256':expected_seal,
        'analysis_plan_sha256':seal['analysis_plan_sha256'],'manifest_sha256':seal['prepared_manifest_sha256'],'data_audit':audit,'test_prediction_count':0,
        'training_ids_sha256':common.object_hash(train_ids),'validation_ids_sha256':common.object_hash(val_ids),'elapsed_seconds':time.monotonic()-started,
        'process_ru_maxrss_native':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'process_ru_maxrss_unit':'bytes_on_mac_kib_on_linux',
        'dependencies':{x:importlib.metadata.version(x) for x in ['numpy','scipy','scikit-learn']},'source_sha256':{Path(__file__).name:common.sha256(__file__),Path(common.__file__).name:common.sha256(common.__file__)}}
    common.save_json(output/'run_summary.json',summary)
    common.save_json(output/'selection_seal.json',{'summary_sha256':common.sha256(output/'run_summary.json'),'artifacts':{t:v['artifact_sha256'] for t,v in selections.items()},'test_used_for_selection':False})
    print(json.dumps({'status':summary['status'],'selections':selections,'elapsed_seconds':summary['elapsed_seconds']}))


class Runtime:
    def __init__(self,run_dir):
        self.run_dir=Path(run_dir); self.summary=common.read_json(self.run_dir/'run_summary.json');seal=common.read_json(self.run_dir/'selection_seal.json')
        common.require(self.summary['status']=='completed_train_val_only' and seal['test_used_for_selection'] is False,'unfrozen baseline')
        common.require(common.sha256(self.run_dir/'run_summary.json')==seal['summary_sha256'],'selection changed')
        self.models={}
        for task,value in self.summary['selections'].items():
            path=self.run_dir/value['artifact_path'];common.require(common.sha256(path)==seal['artifacts'][task],'baseline weights changed')
            with np.load(path,allow_pickle=False) as z:self.models[task]={key:z[key].copy() for key in z.files}
        self.checkpoint_sha256=common.object_hash(seal['artifacts'])

    def predict_batch(self,eeg,task_families,condition):
        result=[]
        with threadpool_limits(limits=2):
            for x,task in zip(eeg,task_families,strict=True):
                start=time.perf_counter();model=self.models[task]
                p=.5 if condition=='context_only' else float(expit(features(x,model['filters'])@model['coef'][0]+model['intercept'][0]))
                probabilities={common.TASKS[task][0]:1-p,common.TASKS[task][1]:p}
                result.append({'task_family':task,'predicted_label':max(probabilities,key=probabilities.get),'probabilities':probabilities,
                    'latency_ms':(time.perf_counter()-start)*1000,'latency_scope':'CSP projection, log variance and LDA; excludes causal stream preparation and harness'})
        return result


def load_runtime(run_dir,device='cpu',memory_budget_gib=None,batch_size=None):
    common.require(device=='cpu','CSP baseline is CPU only')
    return Runtime(run_dir)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['prepared','seal','expected-seal-sha256','output']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--max-seconds',type=int,default=1200)
    a=parser.parse_args();fit(a.prepared,a.seal,a.expected_seal_sha256,a.output,a.max_seconds)
