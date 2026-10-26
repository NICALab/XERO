"""Per-pixel12bit rANS, independent GPU streams. Not autoregressive.

This initial implementation deliberately validates/serializes synchronously.
Lengths, stream states, bypass values and headers are all counted in bytes.
Untrusted byte packets are checked before CUDA dereferences stream offsets.
"""
from functools import lru_cache
import os
from pathlib import Path
import site
import struct
import numpy as np
import torch
from torch.utils.cpp_extension import load
import zstandard as zstd
from .models.neural_symbol_entropy import NeuralSymbolEntropy
from .seed_packet_format import pack_packet,packet_layout

HEADER=struct.Struct('<8sQQQQQ');MAGIC=b'SRAN1201'


@lru_cache(maxsize=16)
def division_table(device):
    """Exact reciprocal division for0<=state<2^31 andfreq1..4096.

    f>=2, k=ceil(log2f), m=ceil(2^(31+k)/f). Rounding error in x*m
    is <1/2^k<=1/f, so cannot cross the next integer quotient. f=1
    uses x directly. LUT is fixed algorithmic data, not transmitted sideinfo.
    """
    entries=[0,0]
    for f in range(2,4097):
        k=(f-1).bit_length();m=((1<<(31+k))+f-1)//f
        entries.append(m|((k-1)<<32))
    return torch.tensor(entries,dtype=torch.int64,device=device)


@lru_cache(None)
def extension():
    major,minor=torch.cuda.get_device_capability();os.environ['TORCH_CUDA_ARCH_LIST']=f'{major}.{minor}'
    root=Path(__file__).parent/'csrc';includes=[]
    for p in site.getsitepackages():includes.extend(str(q) for q in (Path(p)/'nvidia').glob('cu*/include') if (q/'cusparse.h').exists())
    return load(name='fiarc_seed_rans_cuda',sources=[str(root/'seed_rans_gpu.cpp'),str(root/'seed_rans_gpu.cu')],
        extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--threads=4'],extra_include_paths=includes,verbose=False)


def cdf_tensor(logits):
    probabilities=logits.float().softmax(-1).reshape(-1,8).contiguous()
    return extension().probabilities_cdf(probabilities)


@torch.no_grad()
def encode(q,cdf,streams=128):
    q=q.reshape(-1).to(torch.int16).contiguous();n=q.numel();streams=max(1,min(streams,max(1,(n+255)//256)))
    if os.environ.get('FIARC_RANS_RECIPROCAL','1')=='1':
        slots,lengths,status,counts=extension().encode_reciprocal(q,cdf,streams,division_table(q.device))
    else:slots,lengths,status,counts=extension().encode(q,cdf,streams,False,True)
    # One small synchronized metadata transfer, not separate status reduction,
    # total-length reduction and a later duplicate lengths transfer.
    metadata=torch.stack([lengths,status,counts]).cpu().numpy()
    if metadata[1].any():raise RuntimeError('GPU rANS encode failure')
    offsets=lengths.long().cumsum(0)-lengths
    total=int(metadata[0].sum(dtype=np.int64));payload=extension().pack(slots,lengths,offsets,total)
    # Count is produced while the encoder already visits each symbol, then
    # transferred with its existing metadata. No additional GPU search/sync
    # for the overwhelmingly common empty bypass stream.
    nb=int(metadata[2].sum(dtype=np.int64))
    bypass=q[(q<-3)|(q>3)].cpu().numpy().astype('<i2',copy=False) if nb else np.empty(0,dtype='<i2')
    if len(bypass)!=nb:raise RuntimeError('GPU bypass count mismatch')
    side=zstd.ZstdCompressor(level=3).compress(bypass.tobytes())
    return pack_packet(n,streams,total,len(bypass),metadata[0],payload.cpu().numpy().tobytes(),side,os.environ.get('FIARC_RANS_COMPACT','0')=='1')


@torch.no_grad()
def encode_many(items):
    """Same packets as encode(), shared metadata and packed-payload transfers.

    Independent encoder-side alphabets only: every symbol and CDF must already
    be available. Decoder order/causality and bitstream formats do not change.
    Rare bypasses retain the reference path; no unproven empty-bypass shortcut.
    """
    from .packet_bypass import EMPTY_ZSTD
    if not items:return []
    prepared=[];metas=[];device=items[0][0].device
    for q,cdf,streams in items:
        if q.device!=device or cdf.device!=device:raise ValueError('batched rANS requires one device')
        q=q.reshape(-1).to(torch.int16).contiguous();n=q.numel();streams=max(1,min(streams,max(1,(n+255)//256)))
        if os.environ.get('FIARC_RANS_RECIPROCAL','1')=='1':
            slots,lengths,status,counts=extension().encode_reciprocal(q,cdf,streams,division_table(device))
        else:slots,lengths,status,counts=extension().encode(q,cdf,streams,False,True)
        prepared.append((q,n,streams,slots,lengths));metas.append(torch.stack([lengths,status,counts]).reshape(-1))
    metadata=torch.cat(metas).cpu().numpy();at=0;payloads=[];records=[]
    for q,n,streams,slots,lengths in prepared:
        row=metadata[at:at+3*streams].reshape(3,streams);at+=3*streams
        if row[1].any():raise RuntimeError('GPU batched rANS encode failure')
        total=int(row[0].sum(dtype=np.int64));nb=int(row[2].sum(dtype=np.int64))
        payloads.append(extension().pack(slots,lengths,lengths.long().cumsum(0)-lengths,total))
        records.append((q,n,streams,total,nb,row[0]))
    packed=torch.cat(payloads).cpu().numpy().tobytes();at=0;packets=[]
    for q,n,streams,total,nb,lengths in records:
        if nb:
            bypass=q[(q<-3)|(q>3)].cpu().numpy().astype('<i2',copy=False)
            if len(bypass)!=nb:raise RuntimeError('GPU batched bypass count mismatch')
            side=zstd.ZstdCompressor(level=3).compress(bypass.tobytes())
        else:side=EMPTY_ZSTD
        packets.append(pack_packet(n,streams,total,nb,lengths,packed[at:at+total],side,os.environ.get('FIARC_RANS_COMPACT','0')=='1'))
        at+=total
    return packets


@torch.no_grad()
def decode(packet,cdf):
    n,streams,total,ne,nb,header_size,itemsize,dtype=packet_layout(packet)
    if cdf.shape!=(n,9):raise ValueError('bad GPU rANS shape/header')
    lens=np.frombuffer(packet,dtype=dtype,count=streams,offset=header_size).astype('int64')
    if (lens<4).any() or int(lens.sum())!=total or (lens>np.iinfo('int32').max).any():raise ValueError('invalid stream table')
    start=header_size+streams*itemsize;device=cdf.device
    data=torch.from_numpy(np.frombuffer(packet,dtype='uint8',count=total,offset=start).copy()).to(device)
    lengths=torch.from_numpy(lens.astype('int32')).to(device);offsets=lengths.long().cumsum(0)-lengths
    out,status=extension().decode(data,lengths,offsets,cdf)
    if bool(status.any()):raise ValueError('GPU rANS decode failure')
    raw=zstd.ZstdDecompressor().decompress(packet[start+total:]) if ne else b''
    if len(raw)!=nb*2 or int((out==-32768).sum())!=nb:raise ValueError('invalid bypass length')
    out[out==-32768]=torch.from_numpy(np.frombuffer(raw,dtype='<i2').copy()).to(device)
    return out
