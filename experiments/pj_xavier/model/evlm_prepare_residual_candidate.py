#!/usr/bin/env python3
"""Train-only residual-target representation candidate. Does not launch fit or admit data."""
import argparse, datetime, hashlib, json, os, pathlib, tempfile
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def transform(row):
    if row.get('split')!='train' or row.get('task_kind')!='quantized_eeg_reconstruction':
        raise ValueError('train-only source contract')
    rec=row['reconstruction']; o=rec['observed']; ix=rec['masked_indices']; ch,n=rec['geometry']['shape']
    if len(o)!=ch*n or len(ix)!=4 or ix!=list(range(ix[0],ix[0]+4)) or ix[0]//n!=ix[-1]//n or ix[0]%n==0 or ix[-1]%n==n-1 or [i for i,v in enumerate(o) if v is None]!=ix:
        raise ValueError('invalid mask geometry')
    y=json.loads(row['messages'][1]['content'])
    if len(y)!=4 or any(type(v) is not int or not -100<=v<=100 for v in y+[v for v in o if v is not None]):
        raise ValueError('invalid measured quantization')
    l,r=o[ix[0]-1],o[ix[-1]+1]
    baseline=[round((l*(5-i)+r*i)/5) for i in range(1,5)]
    residual=[a-b for a,b in zip(y,baseline)]
    if any(not -200<=v<=200 for v in residual) or [a+b for a,b in zip(baseline,residual)]!=y:
        raise ValueError('residual roundtrip failure')
    prompt=row['messages'][0]['content']
    suffix='Return only a JSON integer array in masked-index order.'
    if not prompt.endswith(suffix): raise ValueError('unknown prompt contract')
    prompt=prompt[:-len(suffix)]+'Deterministic within-channel linear baseline (rounded to integer microvolts): '+json.dumps(baseline,separators=(',',':'))+'. Return only a JSON integer array of signed residuals (measured value minus baseline), in masked-index order.'
    return {**row,'id':row['id']+':residual-candidate-v1','parent_example_id':row['id'],'task_kind':'quantized_eeg_residual_candidate_v1','absolute_target':y,'deterministic_baseline':baseline,'messages':[{'role':'user','content':prompt},{'role':'assistant','content':json.dumps(residual,separators=(',',':'))}]}
def selftest():
    x={'split':'train','task_kind':'quantized_eeg_reconstruction','id':'synthetic','reconstruction':{'observed':[1,None,None,None,None,6,7],'masked_indices':[1,2,3,4],'geometry':{'shape':[1,7]}},'messages':[{'role':'user','content':'Return only a JSON integer array in masked-index order.'},{'role':'assistant','content':'[2,4,4,5]'}]}
    z=transform(x)
    assert z['deterministic_baseline']==[2,3,4,5] and json.loads(z['messages'][1]['content'])==[0,1,0,0]
    for split in ('val','test','review'):
        try:transform({**x,'split':split})
        except ValueError:pass
        else:raise AssertionError('denied split accepted')
    assert x['messages'][1]['content']=='[2,4,4,5]'
    return True
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',type=pathlib.Path);ap.add_argument('--expected-sha');ap.add_argument('--output',type=pathlib.Path);a=ap.parse_args()
    selftest()
    if not a.train: print(json.dumps({'synthetic_tests':'passed'}));return
    if not a.output or not a.expected_sha or sha(a.train)!=a.expected_sha:raise ValueError('source hash mismatch')
    if a.train.name!='train.jsonl':raise ValueError('not train file')
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True,mode=0o700)
    tmp=a.output/'train_residual_candidate.jsonl.partial';dst=a.output/'train_residual_candidate.jsonl'
    ids=set();parents=set();groups=set();n=0;ss=0;absvals=[]
    with a.train.open() as inp,tmp.open('x') as out:
        for line in inp:
            row=json.loads(line);z=transform(row)
            if z['id'] in ids:raise ValueError('duplicate example')
            ids.add(z['id']);parents.add(row['provenance']['parent_window_id']);groups.add(row['canonical_subject_id'])
            res=json.loads(z['messages'][1]['content']);ss+=sum(v*v for v in res);absvals.extend(abs(v) for v in res);n+=1
            out.write(json.dumps(z,ensure_ascii=False,separators=(',',':'))+'\n')
        out.flush();os.fsync(out.fileno())
    if sha(a.train)!=a.expected_sha or not n:raise ValueError('input changed or empty')
    os.replace(tmp,dst)
    result={'created_at':datetime.datetime.now().astimezone().isoformat(),'status':'prepared_not_admitted','source_train_sha256':a.expected_sha,'candidate_sha256':sha(dst),'examples':n,'unique_parent_windows':len(parents),'groups':len(groups),'targets':len(absvals),'absolute_recovery_exact':True,'zero_residual_mse':ss/len(absvals),'residual_abs_le1_fraction':sum(v<=1 for v in absvals)/len(absvals),'residual_abs_max':max(absvals),'synthetic_tests':'passed','dev_test_review_read':False,'gradient_steps':0,'queue_modified':False,'requires_new_runner_and_absolute_mse_gate':True}
    (a.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
