"""Exact deployed forward; reverse causal adjoint with symbols held fixed.

This differentiates the continuous reconstruction recurrence, not discrete symbol
selection or float32 rounding. Rate's symbol-selection surrogate is separate.
The reference Python replay is deliberately small-patch-only. CUDA uses one
reverse wavefront loop, rather than one Python/autograd node per image pixel.
"""
import math
import numpy as np
import torch
from torch import nn
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from .spatial_gpu import spatial_quantize_gpu
from .spatial_training import OFFSETS
from .context_rans import spatial_quantize,spatial_predictions

STEP=float(2*np.log1p(.01*.9999))


@triton.jit
def _reverse(Q,POST,C,GP,GR,A,GN,H:tl.constexpr,W:tl.constexpr,BH:tl.constexpr,FULL:tl.constexpr,STEP:tl.constexpr,MAP:tl.constexpr):
    batch=tl.program_id(0);rows=tl.arange(0,BH);base=batch*H*W
    v1=tl.full((BH,),0,tl.float64);v2=tl.full((BH,),0,tl.float64)
    v3=tl.full((BH,),0,tl.float64);v4=tl.full((BH,),0,tl.float64)
    v5=tl.full((BH,),0,tl.float64);v6=tl.full((BH,),0,tl.float64)
    v7=tl.full((BH,),0,tl.float64);v8=tl.full((BH,),0,tl.float64)
    v9=tl.full((BH,),0,tl.float64)
    dys:tl.constexpr=(0,0,0,-1,-1,-1,-1,-1,-2,-2,-2,-3)
    dxs:tl.constexpr=(-1,-2,-3,-2,-1,0,1,2,-1,0,1,0)
    for index in range(3*(H-1)+W):
        wave=3*(H-1)+W-1-index;cols=wave-3*rows
        valid=(rows<H)&(cols>=0)&(cols<W)
        downstream=tl.full((BH,),0,tl.float64)
        for k in tl.static_range(12):
            dy=dys[k];dx=dxs[k];delay=-3*dy-dx
            if delay==1:later=v1
            elif delay==2:later=v2
            elif delay==3:later=v3
            elif delay==4:later=v4
            elif delay==5:later=v5
            elif delay==6:later=v6
            elif delay==7:later=v7
            else:later=v9
            neighbor=valid&(rows-dy<H)&(cols-dx>=0)&(cols-dx<W)
            adj=tl.gather(later,tl.minimum(rows-dy,BH-1),axis=0)
            if MAP:weight=tl.load(C+batch*13*H*W+k*H*W+(rows-dy)*W+cols-dx,mask=neighbor,other=0)
            else:weight=tl.load(C+k)
            downstream+=weight*tl.where(neighbor,adj,0.)
        address=base+rows*W+cols
        gp=tl.load(GP+address,mask=valid,other=0).to(tl.float64)
        if FULL:
            symbol=tl.load(Q+address,mask=valid,other=0).to(tl.int32)
            # Float casts are straight-through; exact-source escapes have no
            # derivative w.r.t. their prediction under fixed symbol selection.
            alpha=tl.where(symbol==-32768,0.,libdevice.exp(symbol.to(tl.float64)*STEP))
            post=tl.load(POST+address,mask=valid,other=0)
            alpha=tl.where((post==0)&(symbol!=0),0.,alpha)
            gr=tl.load(GR+address,mask=valid,other=0).to(tl.float64)
            adjoint=gp+alpha*(gr+downstream)
        else:adjoint=gp
        tl.store(A+address,adjoint,mask=valid)
        tl.store(GN+address,adjoint-downstream,mask=valid)
        v9=v8;v8=v7;v7=v6;v6=v5;v5=v4;v4=v3;v3=v2;v2=v1;v1=tl.where(valid,adjoint,0.)


def frozen_symbol_replay(prediction,source,q,weights,full=True):
    """Autograd reference, no float casts, original quantizer's q held fixed."""
    b,_,h,w=prediction.shape
    rows=[];posts=[]
    for y in range(h):
        row=[];postrow=[]
        for x in range(w):
            p=prediction[:,:,y,x]+(weights[:,12:13,y,x] if weights.ndim==4 else weights[12])
            for k,(dy,dx) in enumerate(OFFSETS):
                yy,xx=y+dy,x+dx
                if 0<=yy<h and 0<=xx<w:
                    neighbor=row[xx] if yy==y else rows[yy][xx]
                    if not full:neighbor=neighbor.detach()
                    weight=weights[:,k:k+1,y,x] if weights.ndim==4 else weights[k]
                    p=p+weight*(neighbor-prediction[:,:,yy,xx])
            symbol=q[:,:,y,x]
            gain=torch.exp(symbol.to(p.dtype)*STEP)
            restored=torch.where((p==0)&(symbol!=0),gain,p*gain)
            r=torch.where(symbol==-32768,source[:,:,y,x],restored)
            row.append(r);postrow.append(p)
        rows.append(row);posts.append(torch.stack(postrow,-1))
    return torch.stack(posts,-2),torch.stack([torch.stack(row,-1) for row in rows],-2)


def neighbor_slices(h,w,dy,dx):
    y0,y1=max(0,-dy),min(h,h-dy);x0,x1=max(0,-dx),min(w,w-dx)
    return (slice(y0,y1),slice(x0,x1)),(slice(y0+dy,y1+dy),slice(x0+dx,x1+dx))


class _Spatial(torch.autograd.Function):
    @staticmethod
    def forward(ctx,source,prediction,weights,mode):
        s=source.detach().float().contiguous();p=prediction.detach().float().contiguous()
        c=weights.detach().double().contiguous()
        if p.is_cuda:q,r,post=spatial_quantize_gpu(s,p,c)
        else:
            if c.ndim==4:raise ValueError('adaptive spatial forward requires CUDA; frozen replay is the CPU gradient reference')
            pairs=[spatial_quantize(a[0],b[0],gain=1.,weights=c.numpy()) for a,b in zip(s.numpy(),p.numpy())]
            q=torch.from_numpy(np.stack([v[0] for v in pairs])[:,None])
            r=torch.from_numpy(np.stack([v[1] for v in pairs])[:,None])
            post=torch.from_numpy(np.stack([spatial_predictions(a[0],rr[0],c.numpy()) for a,rr in zip(p.numpy(),r.numpy())])[:,None])
        ctx.save_for_backward(s,p,c,q,r,post);ctx.full=mode=='recursive';ctx.input_dtype=prediction.dtype
        ctx.mark_non_differentiable(q)
        return post,r,q

    @staticmethod
    def backward(ctx,gpost,grec,gq):
        s,p,c,q,r,post=ctx.saved_tensors
        gp=torch.zeros_like(p) if gpost is None else gpost.contiguous()
        gr=torch.zeros_like(p) if grec is None else grec.contiguous()
        if p.is_cuda:
            b,_,h,w=p.shape;adj=torch.empty_like(p,dtype=torch.float64);gn=torch.empty_like(p)
            _reverse[(b,)](q,post,c,gp,gr,adj,gn,h,w,triton.next_power_of_2(h),ctx.full,STEP,c.ndim==4,
                num_warps=4,enable_fp_fusion=False)
            error=r.double()-p.double();cw=[]
            for dy,dx in OFFSETS:
                dest,src=neighbor_slices(h,w,dy,dx)
                value=adj[...,dest[0],dest[1]]*error[...,src[0],src[1]]
                if c.ndim==4:
                    channel=torch.zeros_like(adj);channel[...,dest[0],dest[1]]=value;cw.append(channel)
                else:cw.append(value.sum())
            gc=torch.cat(cw+[adj],1) if c.ndim==4 else torch.stack(cw+[adj.sum()])
        else:
            with torch.enable_grad():
                pp=p.double().requires_grad_();cc=c.detach().requires_grad_()
                post,rec=frozen_symbol_replay(pp,s.double(),q,cc,ctx.full)
                loss=(post*gp.double()).sum()
                if ctx.full:loss=loss+(rec*gr.double()).sum()
                gn,gc=torch.autograd.grad(loss,(pp,cc))
        return None,gn.to(ctx.input_dtype),gc,None


def adaptive_spatial(source,prediction,weights,mode='recursive'):
    if weights.shape!=(source.shape[0],13,*source.shape[-2:]):raise ValueError('B13HW coefficients required')
    if mode not in ('recursive','detached'):raise ValueError('invalid backward mode')
    return _Spatial.apply(source,prediction,weights,mode)


class DifferentiableSpatial(nn.Module):
    def __init__(self,weights,mode='recursive'):
        super().__init__()
        if mode not in ('recursive','detached'):raise ValueError('invalid spatial backward mode')
        values=torch.as_tensor(weights,dtype=torch.float64)
        if values.shape!=(13,) or not torch.isfinite(values).all():raise ValueError('13 finite spatial weights required')
        self.weights=nn.Parameter(values.clone());self.mode=mode

    def forward(self,source,prediction):
        # Reference apply_symbol has a special p==0 branch. It is treated as a
        # discontinuity; training data are positive, and forward still falls
        # back to exact source for every unrepresentable output.
        return _Spatial.apply(source,prediction,self.weights,self.mode)
