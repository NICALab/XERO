"""Per-symbol conditional rANS: caller must reproduce CDFs from decoded context.

The CDF array is NOT stored in each packet. Four-group codec reconstructs it
from model weights and previous corrected groups. This module alone is not a
self-contained image codec. Native CPU path is the initial correctness baseline.
"""
import ctypes as ct
import functools
import os
from pathlib import Path
import struct
import subprocess
import numpy as np
import zstandard as zstd

HEADER=struct.Struct('<4sQQQ')


@functools.lru_cache(None)
def _lib():
    src=Path(__file__).parent/'csrc/pixel_rans_cpu.cpp';shared=src.with_suffix('.so')
    if not shared.exists() or shared.stat().st_mtime<src.stat().st_mtime:
        tmp=shared.with_name(shared.name+f'.{os.getpid()}.tmp')
        subprocess.run(['g++','-O3','-shared','-fPIC','-std=c++17',str(src),'-o',str(tmp)],check=True)
        os.replace(tmp,shared)
    lib=ct.CDLL(str(shared));ptr=ct.c_void_p;n=ct.c_int64
    lib.encode_pixel_rans.argtypes=[ptr,n,ptr,ptr,n];lib.encode_pixel_rans.restype=n
    lib.decode_pixel_rans.argtypes=[ptr,n,n,ptr,ptr,n,ptr];lib.decode_pixel_rans.restype=ct.c_int
    return lib


def validate_cdf(cdf,n):
    c=np.asarray(cdf)
    if c.shape!=(n,9) or not np.isfinite(c).all():raise ValueError('finite N9 CDF required')
    if (c!=np.floor(c)).any() or (c[:,0]!=0).any() or (c[:,-1]!=4096).any() or (np.diff(c.astype(np.int64),axis=-1)<=0).any():
        raise ValueError('invalid CDF')
    return np.ascontiguousarray(c,dtype='<u2')


def encode(q,cdf):
    q=np.ascontiguousarray(q,dtype='<i2').ravel();c=validate_cdf(cdf,q.size)
    out=np.empty(q.size*2+16,dtype=np.uint8)
    n=_lib().encode_pixel_rans(q.ctypes.data,q.size,c.ctypes.data,out.ctypes.data,out.size)
    if n<0:raise RuntimeError(f'pixel-rANS encode {n}')
    side=zstd.ZstdCompressor(level=3).compress(q[(q<-3)|(q>3)].tobytes())
    return HEADER.pack(b'FPR1',q.size,len(side),n)+side+out[:n].tobytes()


def decode(packet,cdf):
    magic,n,ne,nb=HEADER.unpack_from(packet)
    if magic!=b'FPR1' or len(packet)!=HEADER.size+ne+nb:raise ValueError('invalid pixel-rANS packet')
    c=validate_cdf(cdf,n);off=HEADER.size
    side=np.frombuffer(zstd.ZstdDecompressor().decompress(packet[off:off+ne]),dtype='<i2')
    data=np.frombuffer(packet[off+ne:],dtype=np.uint8);q=np.empty(n,dtype='<i2')
    status=_lib().decode_pixel_rans(data.ctypes.data,data.size,n,c.ctypes.data,side.ctypes.data,side.size,q.ctypes.data)
    if status:raise RuntimeError(f'pixel-rANS decode {status}')
    return q
