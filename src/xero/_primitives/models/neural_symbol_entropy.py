"""A small neural conditional symbol model, compiled to a causal rANS LUT.

The network shares parameters between contexts instead of fitting an unrelated
row for every context. Exporting all finite contexts removes neural inference
from the sequential decoder. The exported CDF, not a free side input, is stored.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..context_rans import context_count


class NeuralSymbolEntropy(nn.Module):
    def __init__(self, mode=7, width=64):
        super().__init__()
        if mode not in (5, 7):
            raise ValueError('initial ablation supports mode5 and mode7')
        self.mode, self.width = mode, width
        self.net = nn.Sequential(nn.Linear(5*mode,width),nn.SiLU(),
                                 nn.Linear(width,width),nn.SiLU(),nn.Linear(width,8))
        nn.init.zeros_(self.net[-1].weight)
        prior=torch.tensor([.002,.006,.03,.922,.03,.006,.002,.002])
        with torch.no_grad(): self.net[-1].bias.copy_(prior.log())

    def forward(self, indexes):
        # Same most-significant-first context packing as context_rans_cpu.cpp.
        digits=torch.stack([(indexes//(5**k))%5 for k in range(self.mode-1,-1,-1)],-1)
        return self.net(F.one_hot(digits,5).float().flatten(-2)).softmax(-1)

    @staticmethod
    def frequencies(probabilities):
        f=(probabilities*4088).floor().to(torch.int32)+1
        return f.scatter_add(-1,probabilities.argmax(-1,keepdim=True),
                             (4096-f.sum(-1,keepdim=True)).to(torch.int32))

    def deployed_probabilities(self, indexes):
        p=self(indexes)
        return p+(self.frequencies(p).float()/4096-p).detach()

    def rate(self,source,post,q,previous=None):
        from .learned_correction_entropy import context_indexes,correction_cost
        ctx=context_indexes(q.detach(),None if previous is None else previous.detach(),self.mode)
        # Compute each finite context once, then gather. No per-voxel MLP graph.
        idx=torch.arange(context_count(self.mode),device=q.device)
        costs=-self.deployed_probabilities(idx).log2()
        return correction_cost(source,post,q,costs[ctx])

    @torch.no_grad()
    def cdf(self, batch=4096):
        device=next(self.parameters()).device
        f=torch.cat([self.frequencies(self(torch.arange(i,min(i+batch,context_count(self.mode)),device=device))).cpu()
                     for i in range(0,context_count(self.mode),batch)])
        return torch.cat([torch.zeros_like(f[:,:1]),f.cumsum(-1)],-1).numpy().astype('<u2')


def empirical_cdf(counts, prior=None, strength=0.):
    """Optional shared neural pseudo-counts; returned per-GOP table is transmitted."""
    c=np.asarray(counts,dtype=np.float64)
    if prior is not None:c=c+float(strength)*np.asarray(prior,dtype=np.float64)
    p=c/np.maximum(c.sum(-1,keepdims=True),1)
    f=np.floor(p*4088).astype(np.int64)+1
    f[np.arange(len(f)),c.argmax(-1)]+=4096-f.sum(-1)
    return np.concatenate([np.zeros((len(f),1),dtype=np.int64),f.cumsum(-1)],-1).astype('<u2')
