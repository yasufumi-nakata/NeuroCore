"""Fresh scratch encoder plus a reconstruction-only decoder."""
from __future__ import annotations
import hashlib
import numpy as np
import torch
from torch import nn
from eegmmidb_phase3.common import require,scientific_plan
from eegmmidb_phase2.model import EEGEncoder

class MaskedEEGModel(nn.Module):
 def __init__(self):
  super().__init__()
  scientific_plan()
  self.encoder=EEGEncoder()
  self.decoder=nn.Sequential(
   nn.Conv1d(32,64,7,padding=3),nn.GELU(),
   nn.Upsample(size=600,mode="linear",align_corners=False),
   nn.Conv1d(64,64,15,padding=7,groups=64),nn.GELU(),nn.Conv1d(64,64,1),
  )
 def forward(self,masked_eeg_volts):
  return self.decoder(self.encoder(masked_eeg_volts).transpose(1,2))

def patch_mask(identity,split,epoch=0,seed=20261122):
 require(split in ("train","val"),"Mask split must be train or SSL val")
 require(isinstance(identity,str) and identity,"Stable sample ID required")
 text=f"{seed}|val|{identity}" if split=="val" else f"{seed}|train|{epoch}|{identity}"
 number=int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],"big")
 rng=np.random.default_rng(number)
 value=np.zeros(192,dtype=bool)
 value[rng.choice(192,96,replace=False)]=True
 return torch.from_numpy(value.reshape(64,3)).repeat_interleave(200,dim=1)

def make_masked_input(eeg_volts,masks):
 require(eeg_volts.ndim==3 and tuple(eeg_volts.shape[1:])==(64,600),"Wrong EEG shape")
 require(masks.shape==eeg_volts.shape and masks.dtype==torch.bool,"Wrong mask shape")
 require(bool(torch.all(masks.sum((1,2))==19200)),"Exactly half of all channel-time samples must be masked")
 return eeg_volts.masked_fill(masks,0.)

def losses_per_window(prediction,target_volts,masks):
 require(prediction.shape==target_volts.shape==masks.shape and masks.dtype==torch.bool,"Misaligned masked-only targets")
 require(bool(torch.all(masks.sum((1,2))==19200)),"Mask count changed")
 # Select before arithmetic: even an extreme unmasked output never enters loss.
 masked_prediction=prediction[masks].float().reshape(len(prediction),-1)
 masked_target=target_volts[masks].float().reshape(len(prediction),-1)*10000.
 return (masked_prediction-masked_target).square().mean(1)

