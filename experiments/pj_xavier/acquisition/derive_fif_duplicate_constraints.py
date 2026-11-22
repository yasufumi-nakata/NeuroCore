"""Derive future must-link folds from sealed exact calibrated EEG duplicates.

This is a source-identity restriction, not evidence of the actual person count.
Original files, record labels and acquired quantities remain unchanged.
"""
from __future__ import annotations
import argparse
import collections
from fractions import Fraction
import json
from pathlib import Path
import time
from delegated_acquisition_v2 import immutable,require,sha


def components(groups,edges):
    parent={x:x for x in groups}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for left,right in edges:
        require(left in parent and right in parent,'Unknown source group')
        a,b=find(left),find(right)
        if a!=b:parent[max(a,b)]=min(a,b)
    out=collections.defaultdict(list)
    for g in sorted(groups):out[find(g)].append(g)
    return sorted(sorted(v) for v in out.values())


def derive(records):
    require(records and len({r['recording_id'] for r in records})==len(records),'Repeated record identity')
    seen={};pairs=[];seconds=Fraction();unique_bytes=0
    for r in sorted(records,key=lambda x:x['recording_id']):
        require(r['catalog_id']=='001148' and r['sample_rate_hz']==256,'Unexpected source or rate')
        key=(r['decoded_EEG_V_sha256'],tuple(r['channels']),r['sample_count'],r['sample_rate_hz'])
        if key in seen:
            old=seen[key]
            pairs.append({'left_record':old['recording_id'],'right_record':r['recording_id'],'left_group':old['subject_group'],'right_group':r['subject_group'],
                'decoded_EEG_V_sha256':key[0],'samples_per_channel':r['sample_count'],'sample_rate_hz':256,'channel_order':r['channels'],
                'whole_file_SHA256_identical':old['source_sha256']==r['source_sha256'],'cross_subject_ID':old['subject_group']!=r['subject_group'],
                'original_events_identical':old['original_stim_events']==r['original_stim_events'],'calibration_metadata_identical':old['channel_calibration']==r['channel_calibration'],
                'datetime_metadata_identical':old['measurement_datetime']==r['measurement_datetime']})
        else:seen[key]=r;seconds+=Fraction(r['sample_count'],256);unique_bytes+=r['EEG_scalar_bytes']
    groups=sorted({r['subject_group'] for r in records});edges=[(r['left_group'],r['right_group']) for r in pairs if r['cross_subject_ID']];cc=components(groups,edges)
    full=components([r['recording_id'] for r in records],[(r['left_record'],r['right_record']) for r in pairs])
    total=sum((Fraction(r['sample_count'],256) for r in records),Fraction())
    aggregate={'status':'passed','source_records':len(records),'nominal_source_ID_groups':len(groups),'unique_decoded_EEG_records':len(seen),
        'duplicate_record_occurrences':len(pairs),'cross_subject_ID_duplicate_pairs':sum(r['cross_subject_ID'] for r in pairs),'whole_file_identical_duplicate_pairs':sum(r['whole_file_SHA256_identical'] for r in pairs),
        'nontrivial_group_must_link_components':sum(len(v)>1 for v in cc),'nominal_ID_groups_in_nontrivial_components':sum(len(v) for v in cc if len(v)>1),
        'all_future_group_split_components':len(cc),'components_are_not_claimed_actual_people':True,
        'all_duplicate_calibration_metadata_identical':all(r['calibration_metadata_identical'] for r in pairs),
        'all_duplicate_event_arrays_identical':all(r['original_events_identical'] for r in pairs),
        'all_duplicate_datetime_metadata_identical':all(r['datetime_metadata_identical'] for r in pairs),
        'all_record_seconds_exact_fraction':[total.numerator,total.denominator],'unique_processed_EEG_seconds_exact_fraction':[seconds.numerator,seconds.denominator],
        'unique_processed_EEG_hours':float(seconds/3600),'unique_EEG_scalar_bytes':unique_bytes,'new_training_hours_added':0}
    constraints={'schema':'xavier_exact_EEG_duplicate_future_fold_constraints_v1','pairs':pairs,'must_link_subject_groups':[v for v in cc if len(v)>1],
        'all_group_components':cc,'exact_EEG_record_components':[v for v in full if len(v)>1],
        'mandatory_rule':'Before any future split, all recordings and sessions belonging to every ID in a must-link component must share a fold, or the entire ambiguous component must be held out. Do not split identical waveforms between training and evaluation.',
        'does_not_assert_same_human_or_cause_of_duplication':True,'original_waveform_or_label_mutations':0}
    return aggregate,constraints


def main():
    p=argparse.ArgumentParser();p.add_argument('--records',required=True);p.add_argument('--expected-records-sha256',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    require(sha(args.records)==args.expected_records_sha256,'Sealed record ledger changed');raw=Path(args.records).read_bytes();rr=[json.loads(x) for x in raw.splitlines()];rr=[r for r in rr if r['catalog_id']=='001148'];a,c=derive(rr)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);immutable(out/'future_split_constraints.private.json',c)
    a.update(source_sha256=sha(__file__),record_ledger_sha256=args.expected_records_sha256,private_constraints_sha256=sha(out/'future_split_constraints.private.json'),created_epoch=time.time())
    immutable(out/'aggregate.json',a);require(sha(args.records)==args.expected_records_sha256,'Original ledger changed');print(json.dumps(a,sort_keys=True))


if __name__=='__main__':main()
