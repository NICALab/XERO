"""Compact causal neural predictor for a bounded full-resolution innovation code."""

import torch
from torch import nn
from torch.nn import functional as F


class CausalPixelPredictor(nn.Module):
    def __init__(self, history=4, width=32, interpolate=False,input_mode='absolute'):
        super().__init__()
        self.history=history
        self.interpolate=interpolate
        if input_mode not in ('absolute','relative_latest'):raise ValueError('invalid temporal input mode')
        self.input_mode=input_mode
        def block(inc,out):
            return nn.Sequential(nn.Conv2d(inc,out,3,padding=1),nn.LeakyReLU(.1),nn.Conv2d(out,out,3,padding=1),nn.LeakyReLU(.1))
        self.fine=block(history,width)
        self.down=block(width,2*width)
        self.middle=block(2*width,2*width)
        self.up=block(3*width,width)
        self.out=nn.Conv2d(width,1,3,padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self,previous):
        # Form small temporal differences in float32 BEFORE the autocast conv;
        # this avoids spending low-precision mantissa on a shared static image.
        features=previous
        if self.input_mode=='relative_latest':
            features=torch.cat([previous[:,:-1].float()-previous[:,-1:].float(),previous[:,-1:].float()],1)
        f=self.fine(features)
        c=self.middle(self.down(F.avg_pool2d(f,2)))
        u=self.up(torch.cat([f,F.interpolate(c,size=f.shape[-2:],mode='bilinear',align_corners=False)],1))
        # Keep the exact context copy and final sum in float32 under AMP.
        base=(previous[:,:1].float()+previous[:,1:2].float())*.5 if self.interpolate else previous[:,-1:].float()
        return base+self.out(u).float()


def expanded_predictor_state(old_state, old_history, old_width, new_history, new_width):
    """Function-preserving channel duplication for integer width expansions.

    Histories are right aligned. New older history inputs are zero initialized.
    Concatenation channel maps must preserve the fine/coarse branch boundary.
    """
    if new_width < old_width or new_width % old_width:
        raise ValueError('new width must be an integer multiple of old width')
    model=CausalPixelPredictor(new_history,new_width)
    result=model.state_dict()
    for name,module in model.named_modules():
        if not isinstance(module,nn.Conv2d):continue
        w=old_state[name+'.weight'];b=old_state[name+'.bias']
        out_index=torch.arange(module.out_channels)%w.shape[0]
        if name=='fine.0':
            value=torch.zeros_like(result[name+'.weight'])
            common=min(old_history,new_history)
            value[:,-common:]=w[out_index,-common:]
        else:
            if name=='up.0':
                in_index=torch.cat([torch.arange(new_width)%old_width,
                    old_width+torch.arange(2*new_width)%(2*old_width)])
            else:in_index=torch.arange(module.in_channels)%w.shape[1]
            multiplicity=torch.bincount(in_index,minlength=w.shape[1])
            value=w[out_index][:,in_index]/multiplicity[in_index][None,:,None,None]
        result[name+'.weight']=value
        result[name+'.bias']=b[out_index].clone()
    return result


def bounded_quantize_torch(source,prediction,tolerance=.01):
    """Training rollout implementation; CPU packet decoding is final authority."""
    s,p=source.float(),prediction.float()
    step=2*torch.log1p(torch.tensor(tolerance*.9999,device=s.device))
    anchor=p.abs().clamp_min(torch.finfo(torch.float32).tiny)
    anchor=torch.where(p==0,torch.ones_like(p),anchor)
    q=torch.round(torch.log(s.abs().clamp_min(torch.finfo(torch.float32).tiny)/anchor)/step)
    rec=torch.where(p<0,-anchor,anchor)*torch.exp(q*step)
    rec=torch.where((s-p).abs()<=tolerance*s.abs(),p,rec)
    rec=torch.where(~torch.isfinite(rec)|((s-rec).abs()>tolerance*s.abs()),s,rec)
    return rec
