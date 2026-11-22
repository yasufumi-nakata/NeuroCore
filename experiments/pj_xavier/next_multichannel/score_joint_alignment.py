"""Strict scorer for frozen cue-to-synthetic-action EEG/environment proxy runs.

It scores submitted predictions, never manufactures predictions or treats cue proxy
as free-intent/OS gold. Source/EKG verification must happen upstream; the scorer
checks the sealed trial/group/layout/control contract before any metrics.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import hashlib
import math
import random

from joint_alignment import LAYOUTS, mapping, synthetic_action_gold, validate_input_context

SCENARIOS = (
    'model', 'context_only', 'EEG_removed', 'within_record_EEG_shuffle',
    'real_EEG_minus_10_seconds_common_ids', 'correct_class_routing_upper_bound',
)
REQUIRED = frozenset({
    'scenario', 'split', 'signal_trial_id_sha256', 'must_link_component_sha256',
    'window_sha256', 'input_environment', 'original_source_cue_gold',
    'synthetic_mapping_action_gold', 'model_prob_x', 'prediction_receipt_sha256',
    'model_checkpoint_sha256', 'intervention_receipt_sha256',
    'label_provenance', 'free_intent_gold', 'os_action_executed',
    'heldout_group_required',
})
HEX = set('0123456789abcdef')
REGISTRY_FIELDS = frozenset({
    'signal_trial_id_sha256', 'must_link_component_sha256', 'window_sha256',
    'split', 'original_source_cue_gold', 'label_provenance',
    'training_exposure_proven_zero',
})

def digest_registry(registry: list[dict]) -> str:
    if not isinstance(registry,list):
        raise ValueError('Registry must be an immutable archived list')
    payload = json.dumps(registry, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _digest(v):
    return isinstance(v, str) and len(v) == 64 and set(v) <= HEX


def _ba(rows):
    recalls = []
    for y in ('X', 'Y'):
        subset = [r for r in rows if r['synthetic_mapping_action_gold'] == y]
        if not subset:
            raise ValueError('Both proxy classes are required within each group')
        recalls.append(sum((r['model_prob_x'] >= .5) == (y == 'X') for r in subset) / len(subset))
    return sum(recalls) / 2


def _metrics(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r['must_link_component_sha256']].append(r)
    scores = [_ba(v) for v in groups.values()]
    p = [r['model_prob_x'] for r in rows]
    y = [float(r['synthetic_mapping_action_gold'] == 'X') for r in rows]
    eps = 1e-12
    nll = -sum(a * math.log(max(eps, pi)) + (1-a) * math.log(max(eps, 1-pi)) for a,pi in zip(y,p)) / len(y)
    brier = sum((a-pi)**2 for a,pi in zip(y,p)) / len(y)
    selective = [r for r in rows if max(r['model_prob_x'], 1-r['model_prob_x']) >= .8]
    return {'groups': len(groups), 'trials': len(rows)//4, 'environment_views': len(rows),
            'mean_group_balanced_accuracy': sum(scores)/len(scores),
            'nll': nll, 'brier': brier, 'coverage_confidence_0_8': len(selective)/len(rows),
            'selective_accuracy': (sum((r['model_prob_x'] >= .5) ==
                              (r['synthetic_mapping_action_gold'] == 'X') for r in selective)/len(selective)) if selective else None}


def score(rows: list[dict], *, frozen_registry: list[dict],
          expected_registry_sha256: str, seed: int = 20261122, bootstrap: int = 1000) -> dict:
    """Require matched, source-labelled, heldout 4-layout predictions and controls.

    This does not certify the intervention itself: each condition's distinct
    frozen execution receipt must be independently checked by its runner.
    """
    if not _digest(expected_registry_sha256) or digest_registry(frozen_registry) != expected_registry_sha256:
        raise ValueError('Frozen independent registry SHA mismatch')
    registry = {}
    for entry in frozen_registry:
        if set(entry) != REGISTRY_FIELDS or entry['training_exposure_proven_zero'] is not True:
            raise ValueError('Incomplete / exposed registry item')
        if any(not _digest(entry[k]) for k in ('signal_trial_id_sha256','must_link_component_sha256','window_sha256')):
            raise ValueError('Invalid frozen registry identity')
        k=entry['signal_trial_id_sha256']
        if k in registry or entry['split'] not in ('val','test') or entry['original_source_cue_gold'] not in ('A','B'):
            raise ValueError('Invalid independent registry split or duplicate')
        registry[k] = entry
    if bootstrap < 1 or bootstrap > 10000 or not rows or not registry:
        raise ValueError('Non-empty fixed benchmark and finite bootstrap budget required')
    by_scenario = defaultdict(list)
    seen = set()
    identity = {}
    receipt_by_scenario = defaultdict(set)
    prediction_by_scenario = defaultdict(set)
    checkpoint_by_scenario = defaultdict(set)
    for row in rows:
        if set(row) != REQUIRED:
            raise ValueError('Unexpected or missing prediction/label fields')
        scenario = row['scenario']
        if scenario not in SCENARIOS or row['split'] not in ('val','test'):
            raise ValueError('Only registered controls and untouched heldout splits')
        if any(not _digest(row[k]) for k in (
            'signal_trial_id_sha256','must_link_component_sha256','window_sha256',
            'prediction_receipt_sha256','model_checkpoint_sha256','intervention_receipt_sha256')):
            raise ValueError('Missing attestation SHA256')
        if row['heldout_group_required'] is not True or \
           row['label_provenance'] != 'cue_original_plus_synthetic_environment_proxy' or \
           row['free_intent_gold'] is not None or row['os_action_executed'] is not False:
            raise ValueError('Cannot claim spontaneous intent or real OS actions')
        ctx=row['input_environment']
        validate_input_context(ctx)
        if ctx != mapping('sealed-layout',ctx['layout_id']):
            raise ValueError('Layout map differs from preregistered geometry')
        label=synthetic_action_gold(row['original_source_cue_gold'],ctx)
        if label != row['synthetic_mapping_action_gold']:
            raise ValueError('Proxy mapping gold mismatch')
        prob=row['model_prob_x']
        if not isinstance(prob,(float,int)) or isinstance(prob,bool) or not math.isfinite(prob) or not 0<=prob<=1:
            raise ValueError('Invalid probability')
        if scenario == 'correct_class_routing_upper_bound' and prob != float(label=='X'):
            raise ValueError('Incorrect oracle-ceiling control')
        trial=row['signal_trial_id_sha256'];layout=ctx['layout_id'];key=(scenario,trial,layout)
        evidence=registry.get(trial)
        if evidence is None or any(row[k] != evidence[k] for k in
              ('must_link_component_sha256','window_sha256','split',
               'original_source_cue_gold','label_provenance')):
            raise ValueError('Prediction label/split not attested by frozen registry')
        if key in seen:raise ValueError('Repeated scenario/trial/layout')
        seen.add(key)
        orig=(row['split'],row['must_link_component_sha256'],row['window_sha256'],row['original_source_cue_gold'])
        if trial in identity and identity[trial]!=orig:
            raise ValueError('EEG identity or must-link split changed across context/conditions')
        identity[trial]=orig
        receipt_by_scenario[scenario].add(row['intervention_receipt_sha256'])
        prediction_by_scenario[scenario].add(row['prediction_receipt_sha256'])
        checkpoint_by_scenario[scenario].add(row['model_checkpoint_sha256'])
        by_scenario[scenario].append(row)
    if set(identity) != set(registry):
        raise ValueError('Partial missing or unseen frozen registry trials')
    if set(by_scenario) != set(SCENARIOS):
        raise ValueError('All preregistered negative controls required')
    signatures = [{(r['signal_trial_id_sha256'],r['input_environment']['layout_id']) for r in arr}
                  for arr in by_scenario.values()]
    if any(x != signatures[0] for x in signatures[1:]):
        raise ValueError('Negative controls lack paired common trial/layout IDs')
    if len(set(i[0] for i in identity.values())) != 1:
        raise ValueError('Val/test must not be mixed within an evaluation bundle')
    counts=defaultdict(Counter)
    for r in by_scenario['model']:
        counts[r['signal_trial_id_sha256']][r['input_environment']['layout_id']]+=1
    if any(c != Counter(dict.fromkeys(LAYOUTS,1)) for c in counts.values()):
        raise ValueError('Each original EEG trial requires exactly four layout views')
    if any(len(prediction_by_scenario[k])!=1 or len(checkpoint_by_scenario[k])!=1 for k in SCENARIOS):
        raise ValueError('Mixed checkpoint/prediction receipts within control')
    if any(len(receipt_by_scenario[k])!=1 for k in SCENARIOS):
        raise ValueError('One frozen intervention receipt required per scenario')
    if len({next(iter(v)) for v in receipt_by_scenario.values()}) != len(SCENARIOS):
        raise ValueError('Different control interventions require different receipts')
    metrics={k:_metrics(by_scenario[k]) for k in SCENARIOS}
    by_group=defaultdict(lambda:defaultdict(list))
    for scenario,arr in by_scenario.items():
        for row in arr:by_group[scenario][row['must_link_component_sha256']].append(row)
    groups=sorted(by_group['model']); rng=random.Random(seed)
    paired=[]
    for _ in range(bootstrap):
        chosen=[rng.choice(groups) for _ in groups]
        a=sum(_ba(by_group['model'][g]) for g in chosen)/len(chosen)
        b=sum(_ba(by_group['context_only'][g]) for g in chosen)/len(chosen)
        paired.append(a-b)
    paired.sort()
    result={
        'schema':'xavier-joint-synthetic-proxy-scoring-v1',
        'frozen_registry_sha256':expected_registry_sha256,
        'split':next(iter(identity.values()))[0],
        'n_unique_EEG_trials':len(identity),
        'n_independent_must_link_components':len(groups),
        'environment_views_per_trial':4,
        'control_receipt_sha256':{k:next(iter(v)) for k,v in receipt_by_scenario.items()},
        'metrics':metrics,
        'paired_model_minus_context_only_BA':metrics['model']['mean_group_balanced_accuracy']-metrics['context_only']['mean_group_balanced_accuracy'],
        'paired_group_bootstrap_95pct':{'lower':paired[int(.025*(bootstrap-1))],
                                        'upper':paired[int(.975*(bootstrap-1))],
                                        'seed':seed,'draws':bootstrap},
        'gold_type':'original_cue_mapped_to_synthetic_action_only',
        'free_intent_evaluated':False,'real_OS_success_evaluated':False,
        'intervention_receipts_cryptographically_verified_by_scorer':False,
    }
    return result
