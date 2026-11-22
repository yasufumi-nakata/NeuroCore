#!/usr/bin/env python3
"""Freeze-and-evaluate 29606906's five validation subjects on two fixed SSL models.

Never decodes the single test BDF, never uses BDF validation for training,
checkpoint selection, or online EEG-to-action claims. All raw files stay on smac.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

SNAPSHOT=Path.home()/'pj-xavier/smac35/training_code_bdf29606906_v1/experiments/pj_xavier'
sys.path.insert(0,str(SNAPSHOT/'next_multichannel'))
sys.path.insert(0,str(SNAPSHOT/'model'))
import prepare_smac_bdf29606906 as bdf
import cbramod_ssl_smac as trainer


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for v in iter(lambda:f.read(1<<20),b''):h.update(v)
    return h.hexdigest()


def require(cond,reason):
    if not cond:raise ValueError(reason)


def freeze(cfg):
    host=platform.node().split('.')[0].lower()
    require(host=='smac35','Only verified smac35 offline evaluation is permitted')
    root=Path.home()/'pj-xavier/smac35'
    out=Path(cfg['output']).absolute()
    require(out.parent==root/'evaluations' and not out.exists(),'Immutable smac output only')
    require(cfg['schema']=='xavier-bdf29606906-independent-validation-v1','Protocol unknown')
    require(cfg['seed']==20261123 and cfg['no_checkpoint_selection'] is True and cfg['test_open_allowed'] is False,'No test or selection')
    require(cfg['archive']==str(root/'source_archives/figshare29606906_original_v2/dataset.rar'),'Archive location changed')
    require(Path(cfg['corpus']).parent==root/'prepared' and Path(cfg['corpus']).name=='smac806023_CNT_InMID_P300_plusBDF29606906_ssl_v1','Changed corpus')
    require(sha(cfg['archive'])==bdf.ARCHIVE_SHA,'Raw archive SHA mismatch')
    require(sha(cfg['corpus']+'/split_manifest.json')==cfg['split_sha'],'Frozen split seal differs')
    require(sha(cfg['corpus']+'/completion.json')==cfg['completion_sha'],'Corpus seal differs')
    require(sha(bdf.__file__)==cfg['bdf_preparation_sha'],'Prep source changed')
    require(sha(trainer.__file__)==cfg['trainer_sha'],'Trainer source changed')
    require(sha(cfg['official_weights'])==cfg['weights_sha'],'Official CBraMod weights changed')
    require(sha(cfg['before'])==cfg['before_sha'] and sha(cfg['after'])==cfg['after_sha'],'Trained checkpoint SHA mismatch')
    require(cfg['upstream']==str(root/'upstream/CBraMod_pinned'),'Changed official source')
    require(shutil.disk_usage(root).free >= 110*1024**3,'S Mac evaluation recovery storage below 110GiB')
    return out


def validation_rows(cfg,out):
    members=bdf.freeze_members(cfg['archive'])
    require(len(members)==20 and sum(v['split']=='val' for v in members.values())==5
            and sum(v['split']=='test' for v in members.values())==1,'Frozen BDF split changed')
    old=[]
    with (Path(cfg['corpus'])/'manifest.jsonl').open() as f:
        for line in f:old.append(json.loads(line))
    train_groups={x['subject_group'] for x in old if x['split']=='train'}
    train_sources={x['source_sha256'] for x in old if x['split']=='train'}
    train_payload={x['eeg_payload_sha256'] for x in old if x['split']=='train'}
    heldout=[(subject,m) for subject,m in sorted(members.items()) if m['split']=='val']
    require(all(m['group'] not in train_groups for _,m in heldout),'BDF heldout group in training')
    out.mkdir(mode=0o700,parents=False,exist_ok=False)
    (out/'bdf_sources').mkdir(mode=0o700)
    (out/'arrays').mkdir(mode=0o700)
    summary={'schema':'xavier-bdf29606906-independent-validation-manifest-v1','validation_groups':5,
             'test_groups_not_decoded':1,'training_groups_not_decoded':14,'mask_seed':cfg['seed'],
             'archive_sha256':sha(cfg['archive']),'source':bdf.SOURCE,'subject_group_details':[]}
    rows=[]
    for subject,m in heldout:
        source=out/'bdf_sources'/(subject+'.bdf')
        with source.open('xb') as f:
            p=subprocess.run(['/usr/bin/bsdtar','-xOf',cfg['archive'],m['member']],stdout=f,
                             stderr=subprocess.PIPE,timeout=110)
        require(p.returncode==0 and source.stat().st_size>1000,'BDF validation member extraction failed')
        original_sha=sha(source)
        require(original_sha not in train_sources,'Heldout original BDF was in train')
        x,channels,proof=bdf.read_bdf_physical_v(source)
        payload=hashlib.sha256(np.asarray(x.T,dtype='<f8',order='C').tobytes()).hexdigest()
        require(payload not in train_payload,'Heldout native EEG payload was in train')
        arr,windows,rejects=bdf.causal_windows(x)
        require(len(windows)>0,'Heldout validation subject has zero valid windows')
        path=out/'arrays'/(original_sha+'.npy')
        np.save(path,arr,allow_pickle=False)
        for i,w in enumerate(windows):
            rows.append({'id':original_sha+':'+str(i),'split':'val','subject_group':m['group'],
                         'source_sha256':original_sha,'channels':channels,'patches':bdf.PATCHES,
                         'montage_key':'bdf29606906-original-validation-16ch-500hz',
                         'reference':bdf.REFERENCE,'native_sample_rate_hz':bdf.NATIVE_FS,
                         'eeg_index':i,'_path':str(path),**w})
        summary['subject_group_details'].append({'group':m['group'],'source_sha256':original_sha,
            'native_payload_sha256':payload,'array_sha256':sha(path),'windows':len(windows),
            'rejected':rejects,'native_seconds':proof['native_seconds']})
    require(len({x['subject_group'] for x in rows})==5,'BDF validation group count not five')
    summary['total_valid_windows']=len(rows)
    (out/'validation_provenance.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text());cfg['config_sha256']=sha(args.config)
    out=freeze(cfg)
    lock=Path.home()/"pj-xavier/smac35/train_gpu.lock"
    require(lock.is_file(),"GPU lock file missing")
    gpu=lock.open("a")
    fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic()
    rows=validation_rows(cfg,out)
    device=trainer.legacy.frozen.select_device('mps')
    require(device.type=='mps','Real MPS required')
    trainer.seed_all(cfg['seed'],device);torch.set_num_threads(4)
    model=trainer.legacy.load_full_encoder(cfg['upstream'],cfg['official_weights']).to(device=device,dtype=torch.float32)
    scores={}
    for name in ['before','after']:
        checkpoint=torch.load(cfg[name],map_location='cpu',weights_only=True)
        require(checkpoint.get('kind')=='train' and 'model' in checkpoint,'Actual fit checkpoint required')
        model.load_state_dict(checkpoint['model'],strict=True)
        scores[name]={**trainer.validate(model,rows,trainer.Store(),device,32,cfg['seed'],40),
                      'checkpoint_sha256':cfg[name+'_sha']}
    result={'schema':'xavier-bdf29606906-independent-validation-result-v1','status':'completed',
        'scope':'Five heldout validation subjects from original 20 BDF; no test or checkpoint selection',
        'groups':5,'validation_windows':len(rows),'test_opened':0,'fit_steps':0,
        'action_or_online_success_claim':False,'checkpoint_selection':False,
        'source_archive_sha256':bdf.ARCHIVE_SHA,'protocol_sha256':cfg['config_sha256'],
        'validation_provenance_sha256':sha(out/'validation_provenance.json'),
        'scores':scores,'after_minus_before':scores['after']['mean_subject_masked_mse']-scores['before']['mean_subject_masked_mse'],
        'elapsed_seconds':time.monotonic()-start,
        'limitations':['offline masked EEG reconstruction, not intent decoding','five BDF validation groups only','original recording online causality unknown','cannot use this validation for subsequent model selection']}
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'validation_windows':len(rows),'groups':5,
       'before_mse':scores['before']['mean_subject_masked_mse'],
       'after_mse':scores['after']['mean_subject_masked_mse'],'change':result['after_minus_before'],
       'elapsed_seconds':result['elapsed_seconds'],'output':str(out)},ensure_ascii=False))


if __name__=='__main__':main()
