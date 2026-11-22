"""Causal, bounded-memory next-experiment EEG preprocessor.

All inputs MUST be native chronological samples already proven to be physical
volts. This module cannot repair offline filtfilt/ICA/source preprocessing.
No phase2/phase3 recipe or checkpoint is modified.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import hashlib
import json
import numpy as np
from scipy import signal

ALLOWED_C = frozenset((14, 16, 27, 32, 64))
TARGET_HZ = 200


@dataclass(frozen=True)
class Montage:
    names: tuple[str, ...]
    reference: str
    source_hz: int
    source_provenance: str

    def __post_init__(self):
        if len(self.names) not in ALLOWED_C:
            raise ValueError("14/16/27/32/64 EEG channels only")
        if len(set(self.names)) != len(self.names) or any(not n for n in self.names):
            raise ValueError("nonempty distinct ordered channel names required")
        if not self.reference or not self.source_provenance:
            raise ValueError("reference and raw causal provenance are required")
        if not isinstance(self.source_hz, int) or self.source_hz < 90:
            raise ValueError("supported native sample rates are integral >=90 Hz")

    @property
    def key(self) -> str:
        fields = (self.names, self.reference, self.source_hz,
                  self.source_provenance, "physical_V", "causal_sos3_40_lp80_v1")
        return hashlib.sha256(json.dumps(fields,ensure_ascii=False).encode()).hexdigest()


class CausalRecordStream:
    """State persists over chunks, and starts afresh at each true record boundary.

    A sample emitted at target index j depends only on native source samples
    <= floor(j * source_hz / 200). The interpolation is a declared ONE
    native-sample delay, not a forward-looking 'resample_poly' operation.
    """

    def __init__(self, montage: Montage, *, unit_is_verified_physical_volts: bool,
                 source_is_raw_causal: bool, warmup_seconds: float = 2.0):
        if not unit_is_verified_physical_volts:
            raise ValueError("raw units not verified as physical volts")
        if not source_is_raw_causal:
            raise ValueError("offline noncausal preprocessing cannot be repaired")
        if warmup_seconds < 1:
            raise ValueError("at least 1 s causal filter warmup required")
        self.montage = montage
        self.n_c = len(montage.names)
        self.fs = montage.source_hz
        self.warmup_native = int(np.ceil(warmup_seconds * self.fs))
        self.sos = signal.butter(4,(3,40),btype="bandpass",fs=self.fs,output="sos")
        # Target Nyquist is 100Hz. Use a causal additional guard when
        # downsampling; no symmetric / future padding is used.
        self.anti_alias = (signal.butter(6,80,btype="lowpass",fs=self.fs,output="sos")
                           if self.fs > TARGET_HZ else None)
        self.zi = np.zeros((self.sos.shape[0],self.n_c,2),dtype=np.float64)
        self.ai_zi = (np.zeros((self.anti_alias.shape[0],self.n_c,2),dtype=np.float64)
                      if self.anti_alias is not None else None)
        self.seen = 0
        self.next_j = 0
        self.prev = None

    def push(self, native_V: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (V [C,N_out], target_index, native_last_available_index).

        Zero samples, missing data, NaN, clipping or gaps must be handled at
        source-specific QC gate. A missing contiguous interval cannot be
        silently closed: instantiate a NEW stream after the documented gap.
        """
        x = np.asarray(native_V,dtype=np.float64)
        if x.ndim != 2 or x.shape[0] != self.n_c:
            raise ValueError("native samples must be channel-major [C,N]")
        if x.shape[1] == 0:
            return np.zeros((self.n_c,0),dtype=np.float32), np.array([],np.int64),np.array([],np.int64)
        if not np.isfinite(x).all():
            raise ValueError("missing/nonfinite input; terminate segment and record gap")
        if np.max(np.abs(x))>0.1:
            raise ValueError("unverified physical-V amplitudes; source QC required")
        filtered,self.zi=signal.sosfilt(self.sos,x,axis=1,zi=self.zi)
        if self.anti_alias is not None:
            filtered,self.ai_zi=signal.sosfilt(self.anti_alias,filtered,axis=1,zi=self.ai_zi)
        start = self.seen
        self.seen += x.shape[1]
        prev = self.prev
        segment = (np.concatenate((prev[:,None],filtered),axis=1)
                   if prev is not None else filtered)
        offset = start-1 if prev is not None else start
        self.prev=filtered[:,-1].copy()
        # A conservative upper bound for j with source n <= seen-1.
        end_j=(self.seen*TARGET_HZ + self.fs-1)//self.fs + 1
        j=np.arange(self.next_j,end_j,dtype=np.int64)
        n=(j*self.fs)//TARGET_HZ
        j=j[(n<self.seen)&(n>=self.warmup_native)&(n>0)]
        n=(j*self.fs)//TARGET_HZ
        if len(j):
            local=n-offset
            if local.min()<1 or local.max()>=segment.shape[1]:
                raise AssertionError("inconsistent chunk output / source cutoff")
            frac=((j*self.fs)%TARGET_HZ).astype(np.float64)/TARGET_HZ
            # Delayed interpolation x[n-1]*(1-alpha)+x[n]*alpha:
            # effective input timestamp (n-1+alpha)/fs = j/200 -1/fs.
            y=(segment[:,local-1]*(1-frac)+segment[:,local]*frac)
            self.next_j=int(j[-1]+1)
            return y.astype(np.float32),j,n.astype(np.int64)
        # Advance over indices skipped because startup warmup, if available.
        while (self.next_j*self.fs)//TARGET_HZ < self.seen:
            self.next_j+=1
        return np.zeros((self.n_c,0),dtype=np.float32),np.array([],np.int64),np.array([],np.int64)


def patch_shards(output: np.ndarray, target_indices: np.ndarray,
                 source_cutoff: np.ndarray, montage: Montage, *,
                 consecutive_patches: int=3) -> list[dict]:
    """Construct nonoverlapping 200Hz patches without inventing time continuity.

    This in-memory helper expects emitted output from one chronological stream.
    Production callers must carry a <200-sample buffer between push calls;
    splitting a stream into chunks and treating their first patch as independent
    is forbidden. Each shard records its latest source index.
    """
    y=np.asarray(output)
    js=np.asarray(target_indices)
    cutoff=np.asarray(source_cutoff)
    if consecutive_patches < 1 or y.ndim!=2 or y.shape[0]!=len(montage.names):
        raise ValueError("patch configuration mismatch")
    if not (y.shape[1]==len(js)==len(cutoff)):
        raise ValueError("time provenance length mismatch")
    chunk_len=TARGET_HZ*consecutive_patches
    out=[]
    for start in range(0,len(js)-chunk_len+1,chunk_len):
        sl=slice(start,start+chunk_len)
        if np.any(np.diff(js[sl])!=1):
            raise ValueError("gap in target indices")
        out.append({"data_V":np.asarray(y[:,sl]).reshape(len(montage.names),
                      consecutive_patches,TARGET_HZ).copy(),
                    "montage_key":montage.key,"target_first":int(js[start]),
                    "target_last":int(js[start+chunk_len-1]),
                    "latest_native_source_index":int(cutoff[start+chunk_len-1]),
                    "feature_available_native_s":int(cutoff[start+chunk_len-1])/montage.source_hz,
                    "native_input_read_ahead":False})
    return out



class StreamingShardAssembler:
    """Keep at most (P*200-1) output samples between source chunks.

    A discontinuity fails closed. Call a new instance after a documented gap;
    never combine fragments across independent records or reset boundaries.
    """
    def __init__(self, montage: Montage, consecutive_patches: int=3):
        if consecutive_patches<1: raise ValueError("invalid patch count")
        self.montage=montage
        self.patch_len=TARGET_HZ*consecutive_patches
        self.pending_y=np.zeros((len(montage.names),0),dtype=np.float32)
        self.pending_j=np.array([],dtype=np.int64)
        self.pending_cutoff=np.array([],dtype=np.int64)
        self.last_target_index: int | None = None

    def append(self, samples: np.ndarray, targets: np.ndarray,
               cutoffs: np.ndarray) -> list[dict]:
        y=np.asarray(samples)
        t=np.asarray(targets)
        c=np.asarray(cutoffs)
        if y.ndim!=2 or y.shape[0]!=len(self.montage.names) or not (y.shape[1]==len(t)==len(c)):
            raise ValueError("shard input shape mismatch")
        if len(t) and (np.any(np.diff(t)!=1) or
                       (self.last_target_index is not None and t[0]!=self.last_target_index+1)):
            raise ValueError("discontinuity in source target stream")
        if len(t):self.last_target_index=int(t[-1])
        self.pending_y=np.concatenate((self.pending_y,y),axis=1)
        self.pending_j=np.concatenate((self.pending_j,t))
        self.pending_cutoff=np.concatenate((self.pending_cutoff,c))
        n=(len(self.pending_j)//self.patch_len)*self.patch_len
        if not n:return []
        shards=patch_shards(self.pending_y[:,:n],self.pending_j[:n],
                            self.pending_cutoff[:n],self.montage,
                            consecutive_patches=self.patch_len//TARGET_HZ)
        self.pending_y=self.pending_y[:,n:]
        self.pending_j=self.pending_j[n:]
        self.pending_cutoff=self.pending_cutoff[n:]
        if self.pending_y.shape[1]>=self.patch_len:
            raise AssertionError("unbounded stream assembly")
        return shards


def cbramod_encoder_input(shard_V: np.ndarray, *, physical_unit_verified: bool) -> np.ndarray:
    """Physical V -> CBraMod nominal input V*1e4, exactly once.

    Fails rather than silently clipping or double-scaling when unit/QC unknown.
    This only defines a representation; it does not verify full model compatibility.
    """
    if not physical_unit_verified:raise ValueError("unverified source physical volts")
    x=np.asarray(shard_V,dtype=np.float64)
    if x.ndim!=3 or x.shape[0] not in ALLOWED_C or x.shape[-1]!=TARGET_HZ:
        raise ValueError("[C,P,200] physical-volts shard expected")
    if not np.isfinite(x).all():raise ValueError("nonfinite signal")
    out=x*1e4
    if np.max(np.abs(out))>1.00001:
        raise ValueError("source-specific QC/encoder range not met; do not rescale or clip")
    return out.astype(np.float32)

def assert_same_montage_batch(shards: Sequence[dict]) -> np.ndarray:
    if not shards: raise ValueError("empty batch")
    keys={s["montage_key"] for s in shards}
    if len(keys)!=1: raise ValueError("different montage/reference/source-rate cannot share a batch")
    shapes={s["data_V"].shape for s in shards}
    if len(shapes)!=1: raise ValueError("variable channel shape requires independent buckets")
    return np.stack([s["data_V"] for s in shards],axis=0)


try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None

if nn is not None:
    class ChannelQueryResampler(nn.Module):
        """Variable-C/P to eight LLM prefix tokens, without channel zero padding.

        Applies to features from a NEW source-appropriate encoder. It does not
        change the pretrained CBraMod/old eight-channel adapter weights.
        """
        def __init__(self, feature_dim: int, num_queries: int=8, n_heads: int=4):
            super().__init__()
            if feature_dim % n_heads or feature_dim<=0: raise ValueError("bad feature width")
            self.queries=nn.Parameter(torch.randn(num_queries,feature_dim)*0.02)
            self.attn=nn.MultiheadAttention(feature_dim,n_heads,batch_first=True)
            self.out_norm=nn.LayerNorm(feature_dim)

        def forward(self, features, *, montage_keys):
            if not isinstance(montage_keys,(list,tuple)) or len(montage_keys)!=features.size(0) or not montage_keys[0] or len(set(montage_keys))!=1:
                raise ValueError("homogeneous ordered montage keys required per batch")
            if features.ndim!=4 or features.size(1) not in ALLOWED_C:
                raise ValueError("expected [B,C,P,D] with declared montage")
            b,c,p,d=features.shape
            if d!=self.queries.shape[1]:raise ValueError("feature width mismatch")
            if not torch.isfinite(features).all():raise ValueError("nonfinite features")
            keys=features.reshape(b,c*p,d)
            q=self.queries.unsqueeze(0).expand(b,-1,-1)
            result,_=self.attn(q,keys,keys,need_weights=False)
            return self.out_norm(result)
else:
    class ChannelQueryResampler:
        def __init__(self,*args,**kwargs):
            raise ImportError("PyTorch is required for trainable resampler")
