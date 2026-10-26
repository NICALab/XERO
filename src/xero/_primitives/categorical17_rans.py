"""Direct17-category location coding. No bypass values, no autoregression."""
import ctypes as ct
from functools import lru_cache
import struct
import numpy as np
import torch
from .pixel_rans import _lib

CPU_MAGIC=b'FSC17CPU';GPU_MAGIC=b'SRAN1701'
CPU_HEADER=struct.Struct('<8sQQ');GPU_HEADER=struct.Struct('<8sQQQ')

def frequencies(prob):
    f=(prob*4079).floor().to(torch.int32)+1
    return f.scatter_add(-1,prob.argmax(-1,keepdim=True),(4096-f.sum(-1,keepdim=True)).int())

def probabilities(logits):
    p=logits.float().softmax(-1)
    return p+(frequencies(p).float()/4096-p).detach()

def cpu_cdf(logits):
    f=frequencies(logits.float().softmax(-1)).reshape(-1,17)
    return torch.cat([torch.zeros_like(f[:,:1]),f.cumsum(-1)],-1).cpu().numpy().astype('<u2')

def gpu_cdf(logits):
    from .seed_rans_gpu import extension
    return extension().cdf17(logits.float().softmax(-1).reshape(-1,17).contiguous())

@lru_cache(None)
def library():
    lib=_lib();ptr=ct.c_void_p;n=ct.c_int64
    lib.encode_categorical17.argtypes=[ptr,n,ptr,ptr,n];lib.encode_categorical17.restype=n
    lib.decode_categorical17.argtypes=[ptr,n,n,ptr,ptr];lib.decode_categorical17.restype=ct.c_int
    return lib

@torch.no_grad()
def encode(labels,logits,backend,streams=32):
    q=labels.reshape(-1).to(torch.int16).contiguous();n=q.numel()
    if logits.shape[-1]!=17 or logits.numel()!=n*17:raise ValueError('one17-way probability per label required')
    if backend=='cpu':
        cdf=cpu_cdf(logits);q=q.cpu().numpy().astype('<i2',copy=False);out=np.empty(2*n+16,dtype=np.uint8)
        size=library().encode_categorical17(q.ctypes.data,n,cdf.ctypes.data,out.ctypes.data,out.size)
        if size<0:raise ValueError(f'categorical encode failure{size}')
        return CPU_HEADER.pack(CPU_MAGIC,n,size)+out[:size].tobytes()
    if backend!='gpu':raise ValueError('unknown categorical backend')
    from .seed_rans_gpu import extension
    cdf=gpu_cdf(logits);streams=max(1,min(streams,max(1,(n+2047)//2048)))
    slots,lengths,status=extension().encode17(q,cdf,streams);meta=torch.stack([lengths,status]).cpu().numpy()
    if meta[1].any():raise ValueError('GPU categorical encode failed')
    total=int(meta[0].sum(dtype=np.int64));offsets=lengths.long().cumsum(0)-lengths
    payload=extension().pack(slots,lengths,offsets,total)
    return GPU_HEADER.pack(GPU_MAGIC,n,streams,total)+meta[0].astype('<u4',copy=False).tobytes()+payload.cpu().numpy().tobytes()

@torch.no_grad()
def decode(packet,logits):
    if len(packet)<8:raise ValueError('short categorical packet')
    if logits.shape[-1]!=17:raise ValueError('17-way probabilities required')
    n=logits.numel()//17
    if packet[:8]==CPU_MAGIC:
        if len(packet)<CPU_HEADER.size:raise ValueError('short CPU categorical header')
        _,count,size=CPU_HEADER.unpack_from(packet)
        if count!=n or len(packet)!=CPU_HEADER.size+size:raise ValueError('CPU categorical length mismatch')
        cdf=cpu_cdf(logits);data=np.frombuffer(packet,dtype='uint8',offset=CPU_HEADER.size);out=np.empty(n,dtype='<i2')
        code=library().decode_categorical17(data.ctypes.data,data.size,n,cdf.ctypes.data,out.ctypes.data)
        if code:raise ValueError(f'CPU categorical decode failure{code}')
        return torch.from_numpy(out).to(logits.device)
    if packet[:8]!=GPU_MAGIC or len(packet)<GPU_HEADER.size:raise ValueError('unknown categorical packet')
    _,count,streams,total=GPU_HEADER.unpack_from(packet)
    if count!=n or not 0<streams<=65536 or len(packet)!=GPU_HEADER.size+4*streams+total:raise ValueError('bad categorical GPU framing')
    lens=np.frombuffer(packet,dtype='<u4',count=streams,offset=GPU_HEADER.size).astype('int64')
    if (lens<4).any() or (lens>np.iinfo('int32').max).any() or int(lens.sum())!=total:raise ValueError('bad categorical stream lengths')
    device=logits.device;lengths=torch.from_numpy(lens.astype('int32')).to(device);offsets=lengths.long().cumsum(0)-lengths
    data=torch.from_numpy(np.frombuffer(packet,dtype='uint8',offset=GPU_HEADER.size+4*streams).copy()).to(device)
    from .seed_rans_gpu import extension
    out,status=extension().decode17(data,lengths,offsets,gpu_cdf(logits))
    if bool(status.any()):raise ValueError('GPU categorical decode failure')
    return out
