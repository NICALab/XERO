import math, types
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model_state import build_causal

class IntConv(nn.Module):
    def __init__(self,old,bits,weight_bits=None):
        super().__init__();self.limit=(1<<(bits-1))-1
        weight_limit=(1<<((weight_bits or bits)-1))-1
        w=old.weight.detach().cpu().double()
        scale=torch.pow(2.,torch.ceil(torch.log2(w.abs().amax((1,2,3)).clamp_min(2.**-40)/weight_limit)))
        qw=torch.round(w/scale[:,None,None,None])
        self.acc_bound=int((qw.abs().sum((1,2,3))*self.limit).max())
        assert self.acc_bound<2**31,'Requires a wider accumulator; not this profile'
        self.register_buffer('qw',qw.to(old.weight.device))
        self.register_buffer('scale',scale.to(old.weight.device))
        self.register_buffer('bias',old.bias.detach().double())
        self.stride=old.stride;self.padding=old.padding;self.groups=old.groups
        self.cpu_integer_parts=False
    def enable_cpu_integer_parts(self):
        # A13/W8 needs only TWO exact partials, like the CUDA TensorCore path.
        # lo<=64, hi<=32; each absolute dot bound must remain below 2^24.
        self.cpu_weight8=bool(self.qw.abs().max()<=127)
        if self.cpu_weight8:
            assert float(self.qw.abs().sum((1,2,3)).max())*max(64,(self.limit+64)//128)<2**24
            self.register_buffer('w8',self.qw.float().contiguous(memory_format=torch.channels_last))
            self.cpu_integer_parts=True
            return
        # All three partial convolutions stay below 2^24 even in worst case,
        # so ordinary IEEE FP32 GEMM computes EXACT integer sums here.
        hi=torch.floor((self.qw+64)/128);lo=self.qw-128*hi
        hlim=(self.limit+64)//128
        for w,limit in [(lo,64),(hi,hlim),(lo+hi,64+hlim)]:
            assert float(w.abs().sum((1,2,3)).max())*limit<2**24
        self.register_buffer('wlo',lo.float().contiguous(memory_format=torch.channels_last))
        self.register_buffer('whi',hi.float().contiguous(memory_format=torch.channels_last))
        self.register_buffer('wsum',(lo+hi).float().contiguous(memory_format=torch.channels_last))
        self.cpu_integer_parts=True
    def forward(self,x):
        # Dynamic scalar scale is a power of two; no hidden calibration values.
        # This synchronization is diagnostic only, not a deployment proposal.
        value=float(x.float().abs().max())
        scale=2.**math.ceil(math.log2(max(value,2.**-40)/self.limit))
        qx=torch.round(x.float()/scale).clamp(-self.limit,self.limit)
        if x.device.type=='cpu' and self.cpu_integer_parts:
            hi=torch.floor((qx+64)/128);lo=qx-128*hi
            lo=lo.contiguous(memory_format=torch.channels_last);hi=hi.contiguous(memory_format=torch.channels_last)
            if self.cpu_weight8:
                ll=F.conv2d(lo,self.w8,None,self.stride,self.padding,groups=self.groups).double()
                hh=F.conv2d(hi,self.w8,None,self.stride,self.padding,groups=self.groups).double()
                dots=ll+128*hh
            else:
                ll=F.conv2d(lo,self.wlo,None,self.stride,self.padding,groups=self.groups).double()
                hh=F.conv2d(hi,self.whi,None,self.stride,self.padding,groups=self.groups).double()
                ss=F.conv2d(lo+hi,self.wsum,None,self.stride,self.padding,groups=self.groups).double()
                dots=-127*ll+16256*hh+128*ss
        else:dots=F.conv2d(qx.double(),self.qw,None,self.stride,self.padding,groups=self.groups)
        return (dots*(self.scale*scale)[None,:,None,None]+self.bias[None,:,None,None]).float()


class CanonicalActivation(nn.Module):
    def __init__(self,kind,device):
        super().__init__();self.kind=kind
        x=torch.arange(-65536,65537,dtype=torch.float64)/4096
        # Universal (not trained/per-model) activation table on a fixed grid.
        vals=torch.round((x/(1+torch.exp(-x)))*4096).to(torch.int32)
        self.register_buffer('table',vals.to(device))
    def forward(self,x):
        if self.kind=='silu':
            q=torch.round(x.float()*4096).clamp(-65536,65536).long()+65536
            y=self.table[q].float()/4096
            y=torch.where(x>=16,x,y);y=torch.where(x<=-16,torch.zeros_like(y),y)
        else:y=torch.where(x>0,x,x*.1)
        return torch.round(y.float()*4096)/4096


def convert(m,bits,weight_bits=None):
    device=next(m.parameters(),next(m.buffers(),torch.empty(0))).device
    for name,child in list(m.named_children()):
        if isinstance(child,nn.Conv2d):setattr(m,name,IntConv(child,bits,weight_bits))
        elif isinstance(child,nn.SiLU):setattr(m,name,CanonicalActivation('silu',device))
        elif isinstance(child,nn.LeakyReLU):setattr(m,name,CanonicalActivation('leaky',device))
        else:convert(child,bits,weight_bits)


def conditioning_tolerance(codec):
    """Feature conditioning selected by the serialized model profile.

    This scale is separate from the final correction tolerance.
    """
    value=float(np.float32(getattr(codec,'_conditioning_tolerance',codec.tolerance)))
    if not math.isfinite(value) or not 0<value<1:raise ValueError('Invalid conditioning tolerance')
    return value


def canonical_preprocessing(codec,mean,std):
    mean=float(np.float32(mean));inv=float(np.float32(1/float(np.float32(std))))
    tolerance=conditioning_tolerance(codec)
    def prior(self,p,m,s):
        raw=self.prior((p.float()-mean)*inv).float()
        zero=torch.zeros_like(raw[:,:1]);small=torch.full_like(zero,-20.)
        mask=torch.cat([small,small,small,zero,raw[:,:1],small,small,small],1)
        return mask.permute(0,2,3,1).unsqueeze(1).contiguous(),raw[:,1:].permute(0,2,3,1).unsqueeze(1).contiguous()
    def refine(self,p,mask,sr,m,s):
        p=p.float();innovation=torch.where(mask.bool(),sr.float()-p,0.)
        scale=(tolerance*p.abs()).clamp_min(1e-6) if self.conditioning=='relative' else std
        # Diagnostic strict divide; deployment must implement/qualify the same
        # result without assuming GPU approximate reciprocal is interchangeable.
        feature=(innovation.double()/scale.double()).float() if torch.is_tensor(scale) else innovation*inv
        inputs=[(p-mean)*inv,mask,feature]
        if self.lattice_context:inputs.append(self.grid(p).float())
        output=self.refiner(torch.cat(inputs,1));proposal=p+output[:,:1]*scale
        post=mask*sr.float()+(1-mask)*proposal
        return post,output[:,1:].permute(0,2,3,1).unsqueeze(1).contiguous()
    codec.prior_logits=types.MethodType(prior,codec);codec.refine=types.MethodType(refine,codec)
    return lambda x:(x-float(np.float32(mean)))*inv
