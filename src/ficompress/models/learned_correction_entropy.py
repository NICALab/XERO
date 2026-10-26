"""Learned categorical correction model on decoder-known mode-5 contexts.

Unlike the old fixed costs, every symbol is priced by its conditional learned
probability. CDFs use exactly the deployed 12-bit frequency quantization in the
forward loss. A soft bin-membership surrogate supplies gradients to prediction;
hard context indices and exact-source escapes are not differentiated.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..differentiable_spatial import STEP


def context_indexes(q,previous=None,mode=5):
    if q.ndim!=4:raise ValueError('B,T,H,W, with independent batch histories')
    if previous is None:previous=torch.zeros_like(q[:,:1])
    if previous.shape!=q[:,:1].shape:raise ValueError('previous correction must be B,1,H,W')
    left=F.pad(q[...,:-1],(1,0));up=F.pad(q[...,:-1,:],(0,0,1,0))
    ul=F.pad(q[...,:-1,:-1],(1,0,1,0));ur=F.pad(q[...,:-1,1:],(0,1,1,0))
    temporal=torch.cat([previous,q[:,:-1]],1)
    index=torch.zeros_like(q,dtype=torch.long)
    values=[left,up,temporal,ul,ur]
    if mode==7:
        values.extend([F.pad(temporal[...,1:],(0,1)),F.pad(temporal[...,1:,:],(0,0,0,1))])
    elif mode!=5:raise ValueError('mode5 or mode7 required')
    for v in values:index=index*5+(v.long().clamp(-2,2)+2)
    return index


class LearnedCorrectionEntropy(nn.Module):
    def __init__(self,temperature=.15):
        super().__init__()
        prior=torch.tensor([.002,.006,.03,.922,.03,.006,.002,.002])
        self.logits=nn.Parameter(prior.log()[None].repeat(3125,1))
        self.temperature=temperature

    @torch.no_grad()
    def initialize_counts(self,counts):
        counts=torch.as_tensor(counts,device=self.logits.device,dtype=torch.float32)
        if counts.shape!=(3125,8) or (counts<0).any():raise ValueError('invalid context counts')
        prior=(counts.sum(0)+1)/(counts.sum()+8)
        probability=(counts+16*prior)/(counts.sum(1,keepdim=True)+16)
        self.logits.copy_(probability.log())

    def frequencies(self):
        p=self.logits.float().softmax(-1)
        freq=(p*4088).floor().to(torch.int32)+1
        extra=4096-freq.sum(-1,keepdim=True)
        return freq.scatter_add(1,p.argmax(-1,keepdim=True),extra.to(torch.int32))

    @torch.no_grad()
    def cdf(self):
        f=self.frequencies()
        return torch.cat([torch.zeros_like(f[:,:1]),f.cumsum(1)],1).cpu().numpy().astype('<u2')

    def forward(self,source,post,q,previous=None):
        if source.shape!=post.shape or q.shape!=source.shape:raise ValueError('matching B,T,H,W required')
        ctx=context_indexes(q.detach(),None if previous is None else previous.detach())
        probabilities=self.logits.float().softmax(-1)
        deployed=self.frequencies().float()/4096
        probabilities=probabilities+(deployed-probabilities).detach()
        lengths=-probabilities.clamp_min(1/4096).log2()
        costs=lengths[ctx]
        return correction_cost(source,post,q,costs,self.temperature)


def correction_cost(source,post,q,costs,temperature=.15):
        labels=torch.where((q>=-3)&(q<=3),q.long()+3,7)
        categorical=costs.gather(-1,labels[...,None]).squeeze(-1)
        # Large q are int16 bypass values, exact source escapes additionally
        # store float32. Their zstd savings and framing are audited, not faked.
        side=(labels==7).float()*16+(q==-32768).float()*32
        hard=categorical+side
        # For positive non-escaped values log-domain q is the actual quantizer
        # coordinate. Forward q/rec remain the authoritative hard outputs.
        valid=(source>0)&(post>0)&(q!=-32768)
        u=(source.float().clamp_min(1e-12).log()-post.float().clamp_min(1e-12).log())/STEP
        u=torch.where(valid,u,torch.zeros_like(u))
        k=torch.arange(-3,4,device=post.device,dtype=torch.float32)
        left=torch.sigmoid((u[...,None]-k+.5)/temperature)
        right=torch.sigmoid((u[...,None]-k-.5)/temperature)
        masses=(left-right).clamp_min(0)
        escape=(1-masses.sum(-1)).clamp_min(0)
        smooth=(masses*costs[...,:7].detach()).sum(-1)+escape*(costs[...,7].detach()+16)
        smooth=torch.where(valid,smooth,torch.zeros_like(smooth))
        value=hard+(smooth-smooth.detach())
        return value.mean(),dict(categorical_bpv=categorical.detach().mean(),bypass_raw_bpv=side.detach().mean())
