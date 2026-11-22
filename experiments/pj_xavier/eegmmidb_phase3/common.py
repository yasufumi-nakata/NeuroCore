"""Immutable phase3 dependencies and provenance primitives."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
PHASE2_ROOT=ROOT.parent/"eegmmidb_phase2"
PHASE2_SOURCES={
 "__init__.py":"d981967c79ade19e2715af248d50380698870dbbc390976c720ca3cb51b008f2",
 "data_contract.py":"22510965669dd2f4180d37d0efe3d11ae5090667c5d2fdde31132756349849a5",
 "model.py":"5285c00e64bdcbc79b04bbfb236b85c05a035affed06a24b16e768fbcf5606ac",
 "runtime.py":"fea63a48172d65f136a1dc169ee9ffd027d90b9162efa527699c982bcae01ee5",
 "train_models.py":"19f3bd513003d717d4de48a50f310e0350e6c990ec2bd15cebc837f23dfbbffb",
}
SCIENTIFIC_PROTOCOL_SHA="984b516475b06f4e7641074596790c43f76e34d33b6f047239a4ddbd2d0d4016"

def file_sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(2**20),b""):h.update(b)
 return h.hexdigest()

def require(value,message):
 if not value:raise ValueError(message)

for name,digest in PHASE2_SOURCES.items():
 require(file_sha(PHASE2_ROOT/name)==digest,"Immutable phase2 dependency changed: "+name)

from eegmmidb_phase2.data_contract import (  # noqa: E402
 EEGStore,TASKS,contained,is_sha,object_hash,sha256,source_interval_union,write_json,
)
from eegmmidb_phase2.train_models import clone_cpu,rng_state,restore_rng,seed_everything  # noqa: E402
from eegmmidb_phase2.runtime import (  # noqa: E402
 select_device,apply_budget,hardware,memory_sample,synchronize,within_budget,
)

def scientific_plan():
 path=ROOT/"protocol.scientific.v1.json"
 require(file_sha(path)==SCIENTIFIC_PROTOCOL_SHA,"Scientific design changed after registration")
 return json.loads(path.read_text())

def tensor_sha(state):
 import torch
 h=hashlib.sha256()
 for name,tensor in sorted(state.items()):
  require(isinstance(tensor,torch.Tensor),"Tensor-only encoder state required")
  value=tensor.detach().cpu().contiguous()
  h.update(json.dumps([name,str(value.dtype),list(value.shape)]).encode())
  h.update(value.numpy().tobytes())
 return h.hexdigest()

def source_seals():
 names=("__init__.py","common.py","data_ssl.py","ssl_model.py","train_ssl.py")
 result={n:file_sha(ROOT/n) for n in names}
 result["protocol.scientific.v1.json"]=SCIENTIFIC_PROTOCOL_SHA
 result.update({"phase2/"+k:v for k,v in PHASE2_SOURCES.items()})
 return result

