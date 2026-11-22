"""Phase3 keeps phase2 EEG/task semantics; only needed A/B head rows are evaluated."""
from __future__ import annotations
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from eegmmidb_phase3.common import require
from eegmmidb_phase2 import model as original

ARCHITECTURE=original.ARCHITECTURE
BACKBONE_ID=original.BACKBONE_ID
BACKBONE_REVISION=original.BACKBONE_REVISION
CLASS_TOKEN_IDS=original.CLASS_TOKEN_IDS
ROUTE_SYSTEM=original.ROUTE_SYSTEM
ScratchDecoder=original.ScratchDecoder
EEGEncoder=original.EEGEncoder
load_backbone=original.load_backbone
load_tokenizer=original.load_tokenizer
static_prompt=original.static_prompt
learned_state=original.learned_state
restore_learned=original.restore_learned
trainable_parameter_count=original.trainable_parameter_count

class FusedDecoder(original.FusedDecoder):
    """Same two restricted logits, without the unused full-vocabulary projections.

    The pinned Llama causal model computes last_hidden_state -> lm_head. Since
    the registered objective normalizes only A/B, applying those two frozen
    lm_head rows to the same final hidden state has the same mathematical loss.
    Full-vocabulary routing generation still calls the original frozen base.
    """
    def __init__(self,base,*,optimized_symbol_head=True,**kwargs):
        super().__init__(base,**kwargs)
        self.optimized_symbol_head=optimized_symbol_head

    def logits_from_ids(self,eeg,prompt_ids,condition="fused"):
        if not self.optimized_symbol_head:
            return super().logits_from_ids(eeg,prompt_ids,condition)
        require(getattr(self.base.config,"model_type",None)=="llama","Optimized symbol path is restricted to pinned Llama architecture")
        require(len(prompt_ids)==len(eeg) and all(len(ids)>0 for ids in prompt_ids),"Invalid static prompt batch")
        if condition=="context_only":eeg=torch.zeros_like(eeg)
        prefix=self.projector(self.encoder.prefix_features(eeg,8))
        embeddings=self.base.get_input_embeddings();sequences=[];lengths=[]
        for index,ids in enumerate(prompt_ids):
            ids=torch.as_tensor(ids,device=prefix.device,dtype=torch.long)
            text=embeddings(ids)
            sequence=torch.cat((prefix[index].to(text.dtype),text),dim=0)
            sequences.append(sequence);lengths.append(len(sequence))
        padded=pad_sequence(sequences,batch_first=True)
        mask=torch.arange(padded.shape[1],device=padded.device)[None]<torch.tensor(lengths,device=padded.device)[:,None]
        output=self.base.model(inputs_embeds=padded,attention_mask=mask.long(),use_cache=False)
        last=output.last_hidden_state[torch.arange(len(lengths),device=padded.device),torch.tensor(lengths,device=padded.device)-1]
        head=self.base.get_output_embeddings()
        require(isinstance(head,torch.nn.Linear),"Pinned output head must remain linear")
        indices=torch.tensor(self.class_token_ids,device=last.device)
        bias=head.bias[indices] if head.bias is not None else None
        return F.linear(last,head.weight[indices],bias).float()

