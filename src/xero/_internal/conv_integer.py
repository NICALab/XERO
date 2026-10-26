import torch
from torch.nn import functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def maximum_kernel(X,P,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    v=tl.load(X+i,i<N,other=0).to(tl.float32)
    tl.store(P+tl.program_id(0),tl.max(tl.abs(v)))


@triton.jit
def scale_kernel(P,S,N:tl.constexpr,BITS:tl.constexpr,B:tl.constexpr):
    i=tl.arange(0,B);v=tl.max(tl.load(P+i,i<N,other=0))
    v=tl.maximum(v,2.**-40)
    exp=((v.to(tl.int32,bitcast=True)>>23)&255)-127
    scale=((exp+2-BITS+127)<<23).to(tl.float32,bitcast=True)
    scale=tl.where(v>scale*((1<<(BITS-1))-1),scale*2,scale)
    tl.store(S,scale)


@triton.jit
def quantize_kernel(X,S,Q,H:tl.constexpr,W:tl.constexpr,C:tl.constexpr,
                    XS1:tl.constexpr,XS2:tl.constexpr,XS3:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);c=i%C;x=(i//C)%W;y=i//(C*W)
    v=tl.load(X+c*XS1+y*XS2+x*XS3,i<H*W*C,other=0).to(tl.float32)
    q=libdevice.nearbyint(v/tl.load(S)).to(tl.int16)
    tl.store(Q+i,q,i<H*W*C)


@triton.jit
def epilogue_kernel(D,S,WS,BIAS,TABLE,Y,N:tl.constexpr,C:tl.constexpr,KIND:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);c=i%C
    d=tl.load(D+i,i<N,other=0).to(tl.float64)
    scale=tl.load(S).to(tl.float64)*tl.load(WS+c).to(tl.float64)
    v=(d*scale+tl.load(BIAS+c).to(tl.float64)).to(tl.float32)
    if KIND==1:
        j=libdevice.nearbyint(v*4096)
        j=tl.minimum(tl.maximum(j,-65536.),65536.).to(tl.int32)+65536
        table=tl.load(TABLE+j).to(tl.float32)/4096
        v=tl.where(v>=16,v,tl.where(v<=-16,0.,table))
    elif KIND==2:v=tl.where(v>0,v,v*0.1)
    if KIND:v=libdevice.nearbyint(v*4096)/4096
    tl.store(Y+i,v,i<N)


@triton.jit
def kernel(X,W,Y,H:tl.constexpr,WIDTH:tl.constexpr,CI:tl.constexpr,CO:tl.constexpr,KH:tl.constexpr,
           XS1:tl.constexpr,XS2:tl.constexpr,XS3:tl.constexpr,BITS:tl.constexpr,
           BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,
           S,WS,BIAS,TABLE,KIND:tl.constexpr,FUSED:tl.constexpr,WEIGHT8:tl.constexpr):
    m=tl.program_id(0)*BM+tl.arange(0,BM);n=tl.program_id(1)*BN+tl.arange(0,BN);k=tl.arange(0,BK)
    a=tl.full((BM,BN),0,tl.int32)
    if BITS>8:
        b=tl.full((BM,BN),0,tl.int32);c=tl.full((BM,BN),0,tl.int32)
    for part in range(tl.cdiv(KH*KH*CI,BK)):
        kval=part*BK+k;ci=kval%CI;kx=(kval//CI)%KH;ky=kval//(CI*KH)
        yy=m[:,None]//WIDTH+ky[None,:]-KH//2;xx=m[:,None]%WIDTH+kx[None,:]-KH//2
        xv=tl.load(X+ci[None,:]*XS1+yy*XS2+xx*XS3,(m[:,None]<H*WIDTH)&(kval[None,:]<KH*KH*CI)&(yy>=0)&(yy<H)&(xx>=0)&(xx<WIDTH),other=0)
        wv=tl.load(W+kval[:,None]*CO+n[None,:],(kval[:,None]<KH*KH*CI)&(n[None,:]<CO),other=0)
        if BITS==10:
            # 64 * 511^2 < 2^24: every partial sum/product is an exactly
            # representable FP32 integer. Reset MMA accumulator each block,
            # then accumulate partial integer dots in INT32.
            partdot=tl.dot(xv.to(tl.float16),wv.to(tl.float16),out_dtype=tl.float32)
            a+=partdot.to(tl.int32)
        elif BITS<=8:
            a=tl.dot(xv.to(tl.int8),wv.to(tl.int8),a,out_dtype=tl.int32)
        else:
            xhi=(xv.to(tl.int32)+64)>>7;xlo=xv.to(tl.int32)-(xhi<<7)
            if WEIGHT8:
                a=tl.dot(xlo.to(tl.int8),wv.to(tl.int8),a,out_dtype=tl.int32)
                b=tl.dot(xhi.to(tl.int8),wv.to(tl.int8),b,out_dtype=tl.int32)
            else:
                whi=(wv.to(tl.int32)+64)>>7;wlo=wv.to(tl.int32)-(whi<<7)
                a=tl.dot(xlo.to(tl.int8),wlo.to(tl.int8),a,out_dtype=tl.int32)
                b=tl.dot(xhi.to(tl.int8),whi.to(tl.int8),b,out_dtype=tl.int32)
                c=tl.dot((xlo+xhi).to(tl.int8),(wlo+whi).to(tl.int8),c,out_dtype=tl.int32)
    if BITS>10:
        if WEIGHT8:a=a+128*b
        else:a=-127*a+16256*b+128*c
    if FUSED:
        scale=tl.load(S).to(tl.float64)*tl.load(WS+n,n<CO,other=0).to(tl.float64)
        v=(a.to(tl.float64)*scale[None,:]+tl.load(BIAS+n,n<CO,other=0).to(tl.float64)[None,:]).to(tl.float32)
        if KIND==1:
            j=libdevice.nearbyint(v*4096);j=tl.minimum(tl.maximum(j,-65536.),65536.).to(tl.int32)+65536
            table=tl.load(TABLE+j).to(tl.float32)/4096
            v=tl.where(v>=16,v,tl.where(v<=-16,0.,table))
        elif KIND==2:v=tl.where(v>0,v,v*0.1)
        if KIND:v=libdevice.nearbyint(v*4096)/4096
        tl.store(Y+m[:,None]*CO+n[None,:],v,(m[:,None]<H*WIDTH)&(n[None,:]<CO))
    else:tl.store(Y+m[:,None]*CO+n[None,:],a,(m[:,None]<H*WIDTH)&(n[None,:]<CO))


def conv(x,w,bits=12,config=(64,32,32,4),epilogue=None,weight8=False):
    h,width=x.shape[-2:];kh,kw,ci,co=w.shape
    assert x.shape[0]==1 and kh==kw and x.shape[1]==ci
    y=torch.empty((1,co,h,width),device=x.device,dtype=torch.float32 if epilogue is not None else torch.int32,memory_format=torch.channels_last)
    bm,bn,bk,nw=config
    args=epilogue if epilogue is not None else (x,x,x,x,0)
    kernel[(triton.cdiv(h*width,bm),triton.cdiv(co,bn))](x,w,y,h,width,ci,co,kh,*x.stride()[1:],bits,bm,bn,bk,*args,epilogue is not None,weight8,num_warps=nw,enable_fp_fusion=False)
    return y


class FastIntConv(torch.nn.Module):
    """GPU exact-dot counterpart of IntConv; no CPU scalar synchronization."""
    def __init__(self,oracle,bits):
        super().__init__();self.bits=bits;self.limit=oracle.limit
        self.kind=0
        self.weight8=bool(oracle.qw.abs().max()<=127)
        assert oracle.stride==(1,1) and oracle.groups==1
        self.register_buffer('w',oracle.qw.to(torch.int16).permute(2,3,1,0).contiguous())
        self.register_buffer('scales',oracle.scale.clone());self.register_buffer('bias',oracle.bias.clone())
        self.register_buffer('table',torch.zeros(1,device=oracle.qw.device,dtype=torch.int32))
    def forward(self,x):
        assert x.is_contiguous() or x.is_contiguous(memory_format=torch.channels_last)
        n=x.numel();parts=triton.cdiv(n,4096)
        partial=torch.empty(parts,device=x.device,dtype=torch.float32);scale=torch.empty((),device=x.device,dtype=torch.float32)
        maximum_kernel[(parts,)](x,partial,n,4096)
        scale_kernel[(1,)](partial,scale,parts,self.bits,triton.next_power_of_2(parts))
        q=torch.empty_like(x,dtype=torch.int16,memory_format=torch.channels_last)
        quantize_kernel[(triton.cdiv(n,1024),)](x,scale,q,*x.shape[-2:],x.shape[1],*x.stride()[1:],1024,enable_fp_fusion=False)
        co=self.w.shape[-1];bn=min(32,max(16,triton.next_power_of_2(co)))
        return conv(q,self.w,self.bits,(128,bn,32,4),(scale,self.scales,self.bias,self.table,self.kind),self.weight8)


def replace_with_fast(module,bits,oracle_type):
    children=list(module.named_children());skip=set()
    for i,(name,child) in enumerate(children):
        if name in skip:continue
        if isinstance(child,oracle_type):
            fast=FastIntConv(child,bits)
            if isinstance(module,torch.nn.Sequential) and i+1<len(children):
                aname,activation=children[i+1]
                if type(activation).__name__=='CanonicalActivation':
                    fast.kind=1 if activation.kind=='silu' else 2
                    fast.table=activation.table
                    setattr(module,aname,torch.nn.Identity());skip.add(aname)
            setattr(module,name,fast)
        else:replace_with_fast(child,bits,oracle_type)
