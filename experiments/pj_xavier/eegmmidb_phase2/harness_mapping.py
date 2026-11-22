"""Synthetic bounded target environment. Context never receives the gold class."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import itertools
import json
import re

import numpy as np

import benchmark_common as common

IDS=('X','Y')
POSITIONS=('left','right')


def environments(record_id,task_family,seed=20261122):
    """Exhaustive four-way counterbalance; hash changes only presentation order."""
    labels=common.TASKS[task_family]
    variants=[]
    for binding in itertools.permutations(IDS):
        for positions in itertools.permutations(POSITIONS):
            objects=[{'id':target,'position':positions[i],'control':labels[binding.index(target)]} for i,target in enumerate(IDS)]
            public={'task_family':task_family,'objects':objects}
            variants.append({'environment':public,'environment_sha256':common.object_hash(public),
                             'episode_id':hashlib.sha256((str(seed)+'|'+record_id+'|'+common.object_hash(public)).encode()).hexdigest()})
    variants.sort(key=lambda x:hashlib.sha256((str(seed)+'|'+record_id+'|'+x['environment_sha256']).encode()).hexdigest())
    return variants


def target_for(environment,control):
    values=[x['id'] for x in environment['objects'] if x['control']==control]
    common.require(len(values)==1,'control must map to exactly one target')
    return values[0]


def router_prompt(environment,decoded_control):
    common.require(set(environment)=={'task_family','objects'},'unexpected context field')
    common.require(decoded_control in common.TASKS[environment['task_family']],'invalid decoded control')
    objects=environment['objects']
    common.require(len(objects)==2 and {x['id'] for x in objects}==set(IDS),'invalid target set')
    common.require(all(set(x)=={'id','position','control'} for x in objects),'context contains unapproved information')
    return ('Select the target assigned to the decoded motor-imagery control. '
            'Return exactly one character, X or Y, without explanations.\n'
            +'\n'.join(f"Target {x['id']}: screen position {x['position']}; assigned control {x['control']}." for x in objects)
            +f'\nDecoded control: {decoded_control}\nAnswer:')


def parse_action(text):
    if not isinstance(text,str):return None
    value=text.strip()
    return value if re.fullmatch(r'[XY]',value) else None


class Environment:
    def __init__(self,context):
        self.context=context
        self.state={'selected_target':None,'transitions':0}

    def step(self,output,abstain=False):
        if abstain:return {'status':'abstained','changed':False}
        action=parse_action(output)
        if action is None or action not in {x['id'] for x in self.context['objects']}:
            return {'status':'invalid_action','changed':False}
        self.state['selected_target']=action;self.state['transitions']+=1
        return {'status':'selected','changed':True,'target_id':action}

    def oracle(self,gold_control):
        return self.state['selected_target']==target_for(self.context,gold_control)


def required_prompts(seed=20261122):
    prompts={}
    for task,labels in common.TASKS.items():
        for item in environments('constant_context_template',task,seed):
            for label in labels:
                text=router_prompt(item['environment'],label)
                prompts[hashlib.sha256(text.encode()).hexdigest()]={'prompt':text,'environment':item['environment'],'decoded_control':label}
    common.require(len(prompts)==16,'context counterbalance must produce16unique routing prompts')
    return prompts


def run_recorded(rows,route_cache,threshold=.8,seed=20261122):
    """Every call changes a fresh state. Cached text is an explicitly identical prompt."""
    traces=[];seen=set()
    for row in rows:
        key=(row['condition'],row['task_family'],row['id'])
        common.require(key not in seen,'duplicate decoder result');seen.add(key)
        task=row['task_family'];labels=common.TASKS[task]
        p=common.checked_probabilities(row['probabilities'],labels)
        decoded=labels[int(p.argmax())];abstain=bool(p.max()<threshold)
        for item in environments(row['id'],task,seed):
            context=item['environment'];prompt=router_prompt(context,decoded);digest=hashlib.sha256(prompt.encode()).hexdigest()
            common.require(digest in route_cache and route_cache[digest]['prompt']==prompt,'routing cache missing or mismatched prompt')
            response=route_cache[digest]['text'];env=Environment(context);transition=env.step(response,abstain)
            ref=Environment(context);ref.step(target_for(context,decoded),abstain)
            traces.append({'id':row['id'],'episode_id':item['episode_id'],'dataset':row['dataset'],'subject':row['subject'],
                           'task_family':task,'condition':row['condition'],'environment_sha256':item['environment_sha256'],
                           'true_label':row['true_label'],
                           'router_prompt_sha256':digest,'decoder_correct':decoded==row['true_label'],
                           'decoder_class':decoded,'max_probability':float(p.max()),'abstained':abstain,
                           'router_text':response,'transition':transition,'router_correct_given_decoded_class':parse_action(response)==target_for(context,decoded),
                           'task_success':env.oracle(row['true_label']),'reference_router_success':ref.oracle(row['true_label']),
                           'final_state':env.state,'real_OS_action':False,'synthetic_environment':True})
    summaries={}
    groups=defaultdict(list)
    for row in traces:groups[row['condition']+'::'+row['task_family']].append(row)
    for key,values in sorted(groups.items()):
        n=len(values);accepted=[r for r in values if not r['abstained']]
        labels=common.TASKS[values[0]['task_family']]
        per_class={label:[r for r in values if r['true_label']==label] for label in labels}
        balanced=float(np.mean([sum(r['task_success'] for r in v)/len(v) for v in per_class.values()])) if all(per_class.values()) else None
        summaries[key]={'episodes':n,'unique_EEG_records':n//4,'task_success_rate':sum(r['task_success'] for r in values)/n,
                       'class_balanced_task_success_rate':balanced,
                       'reference_router_success_rate':sum(r['reference_router_success'] for r in values)/n,
                       'abstention_rate':sum(r['abstained'] for r in values)/n,'invalid_action_count':sum(r['transition']['status']=='invalid_action' for r in values),
                       'incorrect_selection_count':sum(r['transition']['status']=='selected' and not r['task_success'] for r in values),
                       'router_error_count_among_nonabstained':sum(not r['router_correct_given_decoded_class'] for r in accepted),
                       'decoder_error_count_among_nonabstained':sum(not r['decoder_correct'] for r in accepted),
                       'real_OS_actions':0}
    return traces,{'conditions':summaries,'routing_inference_cache_entries':len(route_cache),'mapping_variants_per_record':4,
                   'interpretation':'Measured synthetic state transitions on public cue labels. Not spontaneous intent or real OS task completion.',
                   'autonomous_real_world_goal_validated':False}


def gold_rows(rows):
    unique={r['id']:r for r in rows if r['condition']=='fused'}
    result=[]
    for r in unique.values():
        value=dict(r);value['condition']='gold_class_upper_bound'
        value['probabilities']={k:float(k==r['true_label']) for k in common.TASKS[r['task_family']]}
        result.append(value)
    return result
