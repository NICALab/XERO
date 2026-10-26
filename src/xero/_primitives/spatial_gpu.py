"""Wavefront-parallel spatial quantization for training (CPU codec remains reference).

Pixels on 3*y+x have no dependencies on one another for the twelve causal
neighbors. Nine prior wavefronts retain reconstruction state in registers.
Double precision and disabled FMA approximate the CPU arithmetic closely;
equivalence and bound checks are required before enabling this in training.
"""
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _wavefront(S,P,C,Q,R,POST,H:tl.constexpr,W:tl.constexpr,BH:tl.constexpr,STEP:tl.constexpr,MAP:tl.constexpr):
    batch=tl.program_id(0);rows=tl.arange(0,BH);base=batch*H*W
    v1=tl.full((BH,),0,tl.float32)
    v2=tl.full((BH,),0,tl.float32)
    v3=tl.full((BH,),0,tl.float32)
    v4=tl.full((BH,),0,tl.float32)
    v5=tl.full((BH,),0,tl.float32)
    v6=tl.full((BH,),0,tl.float32)
    v7=tl.full((BH,),0,tl.float32)
    v8=tl.full((BH,),0,tl.float32)
    v9=tl.full((BH,),0,tl.float32)
    dys:tl.constexpr=(0,0,0,-1,-1,-1,-1,-1,-2,-2,-2,-3)
    dxs:tl.constexpr=(-1,-2,-3,-2,-1,0,1,2,-1,0,1,0)
    eps=tl.full((),.01,tl.float64);step=tl.full((),STEP,tl.float64)
    for wave in range(3*(H-1)+W):
        cols=wave-3*rows;valid=(rows<H)&(cols>=0)&(cols<W)
        if MAP:value=tl.load(C+batch*13*H*W+12*H*W+rows*W+cols,mask=valid,other=0)
        else:value=tl.full((BH,),0,tl.float64)+tl.load(C+12)
        for k in tl.static_range(12):
            dy=dys[k];dx=dxs[k];delay=-3*dy-dx
            if delay==1:previous=v1
            elif delay==2:previous=v2
            elif delay==3:previous=v3
            elif delay==4:previous=v4
            elif delay==5:previous=v5
            elif delay==6:previous=v6
            elif delay==7:previous=v7
            else:previous=v9
            neighbor=(rows+dy>=0)&(cols+dx>=0)&(cols+dx<W)&valid
            reconstructed=tl.gather(previous,tl.maximum(rows+dy,0),axis=0).to(tl.float64)
            neural=tl.load(P+base+(rows+dy)*W+cols+dx,mask=neighbor,other=0).to(tl.float64)
            if MAP:weight=tl.load(C+batch*13*H*W+k*H*W+rows*W+cols,mask=valid,other=0)
            else:weight=tl.load(C+k)
            value=value+weight*tl.where(neighbor,reconstructed-neural,0.)
        neural=tl.load(P+base+rows*W+cols,mask=valid,other=0).to(tl.float64)
        predicted=(neural+value).to(tl.float32)
        source=tl.load(S+base+rows*W+cols,mask=valid,other=0).to(tl.float64)
        p64=predicted.to(tl.float64);anchor=tl.where(p64==0,1.,tl.abs(p64))
        needed=tl.abs(source-p64)>eps*tl.abs(source)
        raw=libdevice.nearbyint(libdevice.log(tl.abs(source)/anchor)/step)
        bad_raw=(raw<=-32768)|(raw>32767)|(raw!=raw)
        symbol=tl.where(needed,tl.minimum(tl.maximum(raw,-32768),32767).to(tl.int32),0)
        safe_symbol=tl.where(bad_raw,0,symbol)
        restored=(tl.where(p64<0,-anchor,anchor)*libdevice.exp(safe_symbol.to(tl.float64)*step)).to(tl.float32)
        restored=tl.where(symbol==0,predicted,restored)
        bad=(needed&bad_raw)|(restored!=restored)|(tl.abs(restored)==float('inf'))|(tl.abs(source-restored.to(tl.float64))>eps*tl.abs(source))
        restored=tl.where(bad,source.to(tl.float32),restored);symbol=tl.where(bad,-32768,symbol)
        tl.store(Q+base+rows*W+cols,symbol.to(tl.int16),mask=valid)
        tl.store(R+base+rows*W+cols,restored,mask=valid)
        tl.store(POST+base+rows*W+cols,predicted,mask=valid)
        v9=v8;v8=v7;v7=v6;v6=v5;v5=v4;v4=v3;v3=v2;v2=v1;v1=tl.where(valid,restored,0.)


def spatial_quantize_gpu(source,prediction,weights):
    if source.device.type!='cuda' or source.device!=prediction.device:raise ValueError('matching CUDA tensors required')
    if source.shape!=prediction.shape or source.ndim!=4 or source.shape[1]!=1:raise ValueError('B1HW required')
    s=source.detach().float().contiguous();p=prediction.detach().float().contiguous()
    c=torch.as_tensor(weights,device=p.device,dtype=torch.float64)
    b,_,h,w=s.shape
    if c.shape not in ((13,),(b,13,h,w)):raise ValueError('13 weights or B13HW map required')
    c=c.contiguous()
    if h>512:raise ValueError('training wavefront currently supports H<=512')
    q=torch.empty_like(s,dtype=torch.int16);r=torch.empty_like(s);post=torch.empty_like(s)
    with torch.cuda.device(s.device):
        _wavefront[(b,)](s,p,c,q,r,post,h,w,triton.next_power_of_2(h),float(2*np.log1p(.01*.9999)),c.ndim==4,
            num_warps=4,enable_fp_fusion=False)
    return q,r,post
