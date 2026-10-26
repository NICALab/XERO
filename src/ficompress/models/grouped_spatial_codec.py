"""Four corrected pixel groups, no AE / Y / Z / transmitted feature map.

Within a group, prediction and probabilities are parallel. Later groups see
only earlier corrected groups and prior corrected frames. Probability tensors
are reproduced at the decoder, rather than serialized for free.
"""
import math
import torch
from torch import nn
from .neural_symbol_entropy import NeuralSymbolEntropy
from .learned_correction_entropy import correction_cost

STEP=2*math.log1p(.01*.9999)
ESCAPE=-32768
GROUPS=((0,0),(0,1),(1,0),(1,1))


def restore_symbols(prediction,q,escaped,step=STEP):
    p=prediction.float();pd=p.double()
    safe=torch.where(q==ESCAPE,0,q).double()
    anchor=torch.where(pd==0,1.,pd.abs())
    r=(torch.where(pd<0,-anchor,anchor)*torch.exp(safe*step)).float()
    r=torch.where(q==0,p,r)
    r=r.clone();r[q==ESCAPE]=escaped
    return r


@torch.no_grad()
def bounded_symbols(source,prediction,step=STEP,tolerance=.01):
    s=source.float();p=prediction.float();sd=s.double();pd=p.double()
    anchor=torch.where(pd==0,1.,pd.abs())
    raw=torch.round(torch.log(sd.abs()/anchor)/step)
    valid=torch.isfinite(raw)&(raw>ESCAPE)&(raw<=32767)
    q=torch.where(valid,raw,0).to(torch.int16)
    q=torch.where((sd-pd).abs()<=tolerance*sd.abs(),0,q)
    restored=(torch.where(pd<0,-anchor,anchor)*torch.exp(q.double()*step)).float()
    r=torch.where(q==0,p,restored)
    bad=~torch.isfinite(r)|((sd-r.double()).abs()>tolerance*sd.abs())
    q=torch.where(bad,ESCAPE,q);r=torch.where(bad,s,r)
    return q,r


def fixed_symbol_surrogate(post,source,q,hard,step=STEP):
    safe=torch.where(q==ESCAPE,0,q).double();pd=post.double()
    gain=torch.exp(safe*step)
    r=torch.where((pd==0)&(q!=0),gain,pd*gain).float()
    r=torch.where(q==ESCAPE,source.detach(),r)
    return r+(hard-r).detach()


class GroupedSpatialCodec(nn.Module):
    def __init__(self,history=8,width=16,anchor_spatial_weights=None):
        super().__init__();self.history=history;self.width=width
        self.anchor=None
        if anchor_spatial_weights is not None:
            from ..differentiable_spatial import DifferentiableSpatial
            self.anchor=DifferentiableSpatial(anchor_spatial_weights).requires_grad_(False)
        self.net=nn.Sequential(nn.Conv2d(history+7,width,3,padding=1),nn.SiLU(),
            nn.Conv2d(width,width,3,padding=1),nn.SiLU(),
            nn.Conv2d(width,width,3,padding=1),nn.SiLU(),nn.Conv2d(width,9,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        prior=torch.tensor([.002,.006,.03,.922,.03,.006,.002,.002])
        with torch.no_grad():self.net[-1].bias[1:].copy_(prior.log())

    @property
    def config(self):
        c=dict(history=self.history,width=self.width)
        if self.anchor is not None:c['anchor_spatial_weights']=self.anchor.weights.detach().cpu().tolist()
        return c

    @torch.no_grad()
    def encode_group(self,source,prediction,group):
        """Native reference for optional causal refinement inside anchor A."""
        if group!=0 or self.anchor is None:return bounded_symbols(source,prediction)
        import numpy as np
        from ..context_rans import spatial_quantize
        s=source.detach().float().cpu().numpy();p=prediction.detach().float().cpu().numpy()
        weights=self.anchor.weights.detach().cpu().numpy()
        values=[spatial_quantize(a[0],b[0],gain=1.,weights=weights) for a,b in zip(s,p)]
        q=torch.from_numpy(np.stack([v[0] for v in values])[:,None]).to(prediction.device)
        r=torch.from_numpy(np.stack([v[1] for v in values])[:,None]).to(prediction.device)
        return q,r

    @torch.no_grad()
    def restore_group(self,prediction,q,escaped,group):
        if group!=0 or self.anchor is None:return restore_symbols(prediction,q,escaped)
        import numpy as np
        from ..context_rans import spatial_restore
        p=prediction.detach().float().cpu().numpy();qq=q.detach().cpu().numpy();ex=escaped.cpu().numpy()
        weights=self.anchor.weights.detach().cpu().numpy();values=[];at=0
        for pp,qs in zip(p,qq):
            n=int((qs==-32768).sum())
            values.append(spatial_restore(pp[0],qs[0],ex[at:at+n],gain=1.,weights=weights));at+=n
        if at!=len(ex):raise ValueError('anchor escape count mismatch')
        return torch.from_numpy(np.stack(values)[:,None]).to(prediction.device)

    def predict_group(self,history,prediction,known,mask,group,mean,std):
        if group not in range(4):raise ValueError('four groups required')
        if history.shape[1]<self.history:raise ValueError('not enough corrected history')
        # The caller's unknown entries may contain anything: force them out.
        innovation=torch.where(mask,known.float()-prediction.float(),0.)/std
        flags=torch.zeros((len(prediction),4,*prediction.shape[-2:]),device=prediction.device)
        flags[:,group]=1
        x=torch.cat([(history[:,-self.history:].float()-prediction.float())/std,
                     (prediction.float()-mean)/std,innovation,mask.float(),flags],1)
        out=self.net(x).float()
        post=prediction.float()+out[:,:1]*std
        logits=out[:,1:].permute(0,2,3,1).unsqueeze(1).contiguous()
        return post,logits

    @staticmethod
    def probabilities(logits):
        p=logits.float().softmax(-1)
        deployed=NeuralSymbolEntropy.frequencies(p).float()/4096
        return p+(deployed-p).detach()

    @staticmethod
    @torch.no_grad()
    def cdf(logits):
        f=NeuralSymbolEntropy.frequencies(logits.float().softmax(-1))
        return torch.cat([torch.zeros_like(f[...,:1]),f.cumsum(-1)],-1).reshape(-1,9).cpu().numpy().astype('<u2')

    def forward(self,source,history,prediction,mean,std,compute_rate=True):
        known=torch.zeros_like(source);mask=torch.zeros_like(source,dtype=torch.bool)
        post_all=torch.zeros_like(source);q_all=torch.zeros_like(source,dtype=torch.int16)
        loss=prediction.new_zeros(());details=[]
        for group,(y,x) in enumerate(GROUPS):
            post,logits=self.predict_group(history,prediction,known,mask,group,mean,std)
            sl=(slice(None),slice(None),slice(y,None,2),slice(x,None,2))
            s=source[sl];pp=post[sl];ll=logits[sl]
            if group==0 and self.anchor is not None:
                pp,hard,q=self.anchor(s,pp)
                rec=hard  # Custom adjoint already differentiates anchor recurrence.
            else:
                q,hard=bounded_symbols(s,pp)
                rec=fixed_symbol_surrogate(pp,s,q,hard)
            if compute_rate:
                cost,parts=correction_cost(s,pp,q,-self.probabilities(ll).log2())
                loss=loss+cost*(s.numel()/source.numel())
            else:cost=loss
            next_known=known.clone();next_known[sl]=rec;known=next_known
            mask=mask.clone();mask[sl]=True
            post_all[sl]=pp;q_all[sl]=q
            details.append(dict(rate=cost.detach(),nonzero=(q!=0).float().mean().detach()))
        return known,post_all,q_all,loss,details
