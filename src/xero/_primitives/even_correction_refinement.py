"""Optional decoder-causal two-phase predictor, default off for old codecs.

Only already decoded even checkerboard corrections may influence odd sites.
No current odd target, correction symbol, or reconstruction is consumed.
"""
import torch
from torch.nn import functional as F
from .checkerboard_entropy import odd_sites


def refine_from_even(post,even_reconstruction,seed_mask,gain,weights=None):
    odd=odd_sites(post)
    # where before arithmetic makes unknown odd entries truly irrelevant,
    # even if they contain NaN/huge values in a source-hidden decoder test.
    known=torch.where(odd,post,even_reconstruction)
    delta=known-post
    h,w=post.shape[-2:];p=F.pad(delta,(1,1,1,1))
    if weights is not None:
        # Fixed transmitted coefficients: up, left, right, down. Zero padding
        # matches calibration; unlike legacy averaging, no edge renormalization.
        total=(weights[0]*p[...,0:h,1:w+1]+weights[1]*p[...,1:h+1,0:w]
               +weights[2]*p[...,1:h+1,2:w+2]+weights[3]*p[...,2:h+2,1:w+1])
        return torch.where(odd&~seed_mask,post+gain*total,post)
    total=p[...,0:h,1:w+1]+p[...,2:h+2,1:w+1]+p[...,1:h+1,0:w]+p[...,1:h+1,2:w+2]
    valid=F.pad(torch.ones_like(post),(1,1,1,1))
    count=valid[...,0:h,1:w+1]+valid[...,2:h+2,1:w+1]+valid[...,1:h+1,0:w]+valid[...,1:h+1,2:w+2]
    return torch.where(odd&~seed_mask,post+gain*total/count.clamp_min(1),post)
