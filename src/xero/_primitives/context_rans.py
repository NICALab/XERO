"""Causal-context rANS serialization for bounded correction symbols (CPU)."""
import ctypes as ct
import functools
import os
from pathlib import Path
import struct
import subprocess
import numpy as np
import zstandard as zstd

_HEADER=struct.Struct('<4sBIIIQQQ')

def context_count(mode):
    if mode in (2,3,4,5,6,7):return 5**mode
    if mode==8:return 3**7
    if mode==9:return 3**9
    raise ValueError('context mode 2..9 required')

@functools.lru_cache(None)
def _lib():
    src=Path(__file__).parent/'csrc/context_rans_cpu.cpp'
    shared=src.with_suffix('.so')
    if not shared.exists() or shared.stat().st_mtime<src.stat().st_mtime:
        temporary=shared.with_name(shared.name+f'.{os.getpid()}.tmp')
        subprocess.run(['g++','-O3','-shared','-fPIC','-std=c++17',str(src),'-o',str(temporary)],check=True)
        os.replace(temporary,shared)
    lib=ct.CDLL(str(shared))
    ptr=ct.c_void_p;i64=ct.c_int64;i32=ct.c_int
    lib.fiarc_context_counts.argtypes=[ptr,i64,i32,i32,i32,ptr]
    lib.fiarc_context_counts.restype=None
    lib.fiarc_context_encode.argtypes=[ptr,i64,i32,i32,i32,ptr,ptr,i64]
    lib.fiarc_context_encode.restype=i64
    lib.fiarc_context_decode.argtypes=[ptr,i64,i64,i32,i32,i32,ptr,ptr,i64,ptr]
    lib.fiarc_context_decode.restype=i32
    lib.fiarc_spatial_quantize.argtypes=[ptr,ptr,i32,i32,ct.c_double,ct.c_double,ct.c_double,ptr,ptr]
    lib.fiarc_spatial_quantize.restype=None
    lib.fiarc_spatial_restore.argtypes=[ptr,ptr,i32,i32,ct.c_double,ct.c_double,ptr,i64,ptr]
    lib.fiarc_spatial_restore.restype=i32
    lib.fiarc_spatial_quantize_linear.argtypes=lib.fiarc_spatial_quantize.argtypes+[ptr]
    lib.fiarc_spatial_quantize_linear.restype=None
    lib.fiarc_spatial_quantize_min_magnitude.argtypes=lib.fiarc_spatial_quantize_linear.argtypes
    lib.fiarc_spatial_quantize_min_magnitude.restype=None
    lib.fiarc_spatial_restore_linear.argtypes=lib.fiarc_spatial_restore.argtypes+[ptr]
    lib.fiarc_spatial_restore_linear.restype=i32
    lib.fiarc_spatial_quantize_map.argtypes=lib.fiarc_spatial_quantize_linear.argtypes
    lib.fiarc_spatial_quantize_map.restype=None
    lib.fiarc_spatial_restore_map.argtypes=lib.fiarc_spatial_restore_linear.argtypes
    lib.fiarc_spatial_restore_map.restype=i32
    lib.fiarc_spatial_predictions_linear.argtypes=[ptr,ptr,i32,i32,ptr,ptr]
    lib.fiarc_spatial_predictions_linear.restype=None
    return lib

def validate_cdf(cdf,mode):
    raw=np.asarray(cdf)
    if raw.shape!=(context_count(mode),9):raise ValueError('invalid CDF shape')
    if not np.isfinite(raw).all() or np.any(raw!=np.floor(raw)):raise ValueError('integer CDF required')
    if np.any(raw[:,0]!=0) or np.any(raw[:,-1]!=4096) or np.any(np.diff(raw.astype('int64'),axis=1)<=0):
        raise ValueError('invalid rANS CDF')
    return np.ascontiguousarray(raw,dtype='<u2')

def encode_context_rans(symbols,mode=4,cdf=None):
    q=np.ascontiguousarray(symbols,dtype='<i2')
    if q.ndim!=3:raise ValueError('TYX array required')
    t,h,w=q.shape
    external=cdf is not None;lib=_lib()
    if external:cdf=validate_cdf(cdf,mode)
    else:
        counts=np.zeros((context_count(mode),8),dtype=np.uint64)
        lib.fiarc_context_counts(q.ctypes.data,q.size,h,w,mode,counts.ctypes.data)
        scaled=counts.astype('float64')/np.maximum(counts.sum(1,keepdims=True),1)*4088
        frequency=np.floor(scaled).astype('int64')+1
        # Put rounding remainder into the most common symbol; all symbols stay legal.
        largest=counts.argmax(1)
        frequency[np.arange(len(frequency)),largest]+=4096-frequency.sum(1)
        cdf=np.concatenate([np.zeros((len(counts),1),dtype='uint16'),np.cumsum(frequency,axis=1).astype('uint16')],axis=1)
    output=np.empty(q.size*2+16,dtype='uint8')
    n=lib.fiarc_context_encode(q.ctypes.data,q.size,h,w,mode,cdf.ctypes.data,output.ctypes.data,output.size)
    if n<0:raise RuntimeError(f'rANS encode status {n}')
    compressor=zstd.ZstdCompressor(level=3)
    tables=b'' if external else compressor.compress(cdf.astype('<u2').tobytes())
    escaped=compressor.compress(q[(q<-3)|(q>3)].tobytes())
    data=output[:n].tobytes()
    return _HEADER.pack(b'FCR2' if external else b'FCR1',mode,t,h,w,len(tables),len(escaped),len(data))+tables+escaped+data

def decode_context_rans(packet,cdf=None):
    magic,mode,t,h,w,nt,ne,nb=_HEADER.unpack_from(packet)
    if magic not in (b'FCR1',b'FCR2'):raise ValueError('invalid rANS header')
    nc=context_count(mode)
    off=_HEADER.size
    if len(packet)!=off+nt+ne+nb:raise ValueError('rANS size mismatch')
    decomp=zstd.ZstdDecompressor()
    if magic==b'FCR2':
        if nt or cdf is None:raise ValueError('external CDF required, no embedded table permitted')
        cdf=validate_cdf(cdf,mode)
    else:cdf=validate_cdf(np.frombuffer(decomp.decompress(packet[off:off+nt]),dtype='<u2').reshape(nc,9),mode)
    escapes=np.frombuffer(decomp.decompress(packet[off+nt:off+nt+ne]),dtype='<i2')
    data=np.frombuffer(packet[off+nt+ne:],dtype='uint8')
    q=np.empty((t,h,w),dtype='<i2')
    status=_lib().fiarc_context_decode(data.ctypes.data,len(data),q.size,h,w,mode,cdf.ctypes.data,escapes.ctypes.data,escapes.size,q.ctypes.data)
    if status:raise RuntimeError(f'rANS decode status {status}')
    return q

def packet_code_costs(packet,cdf=None):
    """Measured CDF cross-entropy by symbol, plus actual packet side bytes.

    These are diagnostics, not gradients or source compression bounds.
    """
    q=decode_context_rans(packet,cdf=cdf)
    magic,mode,t,h,w,nt,ne,nb=_HEADER.unpack_from(packet)
    off=_HEADER.size;nc=context_count(mode)
    if magic==b'FCR2':cdf=validate_cdf(cdf,mode)
    else:cdf=np.frombuffer(zstd.ZstdDecompressor().decompress(packet[off:off+nt]),dtype='<u2').reshape(nc,9)
    counts=np.zeros((nc,8),dtype=np.uint64)
    _lib().fiarc_context_counts(q.ctypes.data,q.size,h,w,mode,counts.ctypes.data)
    lengths=-np.log2(np.diff(cdf.astype('float64'),axis=1)/4096)
    observed=counts.sum(0);bits=(counts*lengths).sum(0)
    result=[]
    for label,indices in [('zero',[3]),('abs1',[2,4]),('abs2',[1,5]),('abs3',[0,6]),('escape',[7])]:
        n=int(observed[indices].sum());total=float(bits[indices].sum())
        result.append(dict(category=label,count=n,mean_cdf_bits=total/n if n else None))
    return dict(symbols=int(q.size),actual_packet_bpv=8*len(packet)/q.size,cdf_cross_entropy_bpv=float(bits.sum()/q.size),
        tables_bytes=nt,large_symbols_bytes=ne,rans_bytes=nb,header_bytes=_HEADER.size,categories=result)


def spatial_quantize(source,prediction,gain=.5,tolerance=.01,step_scale=1.,weights=None,selection='nearest'):
    s=np.ascontiguousarray(source,dtype='float32');p=np.ascontiguousarray(prediction,dtype='float32')
    if s.ndim!=2 or s.shape!=p.shape:raise ValueError('matching 2D images required')
    if not np.isfinite(s).all() or not np.isfinite(p).all():raise ValueError('finite source and prediction required')
    q=np.empty(s.shape,dtype='int16');r=np.empty_like(s)
    if not np.isfinite(step_scale) or step_scale<=0:raise ValueError('positive finite step scale required')
    if selection not in ('nearest','min_magnitude'):raise ValueError('unknown symbol selection')
    if not np.isfinite(tolerance) or not 0<tolerance<1:raise ValueError('tolerance must be in (0,1)')
    h,w=s.shape;step=2*np.log1p(tolerance*.9999)*step_scale
    if selection=='min_magnitude':
        coef=None if weights is None else np.ascontiguousarray(weights,dtype='float64')
        if coef is not None and (coef.shape!=(13,) or not np.isfinite(coef).all()):raise ValueError('13 finite spatial coefficients required')
        _lib().fiarc_spatial_quantize_min_magnitude(s.ctypes.data,p.ctypes.data,h,w,gain,step,tolerance,q.ctypes.data,r.ctypes.data,None if coef is None else coef.ctypes.data)
    elif weights is None:
        _lib().fiarc_spatial_quantize(s.ctypes.data,p.ctypes.data,h,w,gain,step,tolerance,q.ctypes.data,r.ctypes.data)
    else:
        coef=np.ascontiguousarray(weights,dtype='float64')
        if coef.shape not in ((13,),(13,h,w)) or not np.isfinite(coef).all():raise ValueError('13 finite coefficients or 13HW map required')
        fn=_lib().fiarc_spatial_quantize_map if coef.ndim==3 else _lib().fiarc_spatial_quantize_linear
        fn(s.ctypes.data,p.ctypes.data,h,w,gain,step,tolerance,q.ctypes.data,r.ctypes.data,coef.ctypes.data)
    return q,r


def spatial_restore(prediction,q,escapes,gain=.5,tolerance=.01,step_scale=1.,weights=None):
    p=np.ascontiguousarray(prediction,dtype='float32');q=np.ascontiguousarray(q,dtype='int16')
    e=np.ascontiguousarray(escapes,dtype='float32');r=np.empty_like(p)
    if not np.isfinite(step_scale) or step_scale<=0:raise ValueError('positive finite step scale required')
    h,w=p.shape;step=2*np.log1p(tolerance*.9999)*step_scale
    if weights is None:
        status=_lib().fiarc_spatial_restore(p.ctypes.data,q.ctypes.data,h,w,gain,step,e.ctypes.data,e.size,r.ctypes.data)
    else:
        coef=np.ascontiguousarray(weights,dtype='float64')
        if coef.shape not in ((13,),(13,h,w)) or not np.isfinite(coef).all():raise ValueError('13 finite coefficients or 13HW map required')
        fn=_lib().fiarc_spatial_restore_map if coef.ndim==3 else _lib().fiarc_spatial_restore_linear
        status=fn(p.ctypes.data,q.ctypes.data,h,w,gain,step,e.ctypes.data,e.size,r.ctypes.data,coef.ctypes.data)
    if status:raise ValueError('spatial escape count mismatch')
    return r


def spatial_predictions(prediction,reconstruction,weights):
    """Exact deployed predictions from already determined causal recon values."""
    p=np.ascontiguousarray(prediction,dtype='float32');r=np.ascontiguousarray(reconstruction,dtype='float32')
    coef=np.ascontiguousarray(weights,dtype='float64')
    if p.ndim!=2 or p.shape!=r.shape or coef.shape!=(13,):raise ValueError('invalid spatial shapes')
    if not np.isfinite(p).all() or not np.isfinite(r).all() or not np.isfinite(coef).all():raise ValueError('finite arrays required')
    out=np.empty_like(p);h,w=p.shape
    _lib().fiarc_spatial_predictions_linear(p.ctypes.data,r.ctypes.data,h,w,coef.ctypes.data,out.ctypes.data)
    return out
