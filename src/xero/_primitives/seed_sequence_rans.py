"""Persistent entropy states across frames; decode needs only current CDF.

This primitive does not change the temporal predictor or fidelity quantizer.
New magic is mandatory: legacy full-packet stream partitions are different.
"""
from functools import lru_cache
import os
import site
from pathlib import Path
import struct
import numpy as np
import torch
from torch.utils.cpp_extension import load
import zstandard as zstd
from .seed_rans_gpu import extension,division_table

HEADER=struct.Struct('<8sIIIIII');MAGIC=b'SRANSG01'
GOLOMB_MAGIC=b'SRANSG02'


@lru_cache(None)
def decoder_extension():
    major,minor=torch.cuda.get_device_capability();os.environ['TORCH_CUDA_ARCH_LIST']=f'{major}.{minor}'
    root=Path(__file__).parent/'csrc'
    includes=[]
    for p in site.getsitepackages():includes.extend(str(q) for q in (Path(p)/'nvidia').glob('cu*/include') if (q/'cusparse.h').exists())
    return load(name='fiarc_seed_sequence_decode_cuda',sources=[str(root/'seed_sequence_decode.cpp'),str(root/'seed_sequence_decode.cu')],
        extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--threads=4'],extra_include_paths=includes,verbose=False)


def stream_order(items,streams):
    """Per-stream chronological segments; zero-cost dummy padding if uneven."""
    counts=[q.numel() for q,_ in items];frames=len(items)
    if len(set(counts))==1 and counts[0]%streams==0:
        n=counts[0]//streams
        q=torch.stack([q for q,_ in items]).reshape(frames,streams,n).permute(1,0,2).contiguous().reshape(-1)
        cdf=torch.stack([c for _,c in items]).reshape(frames,streams,n,9).permute(1,0,2,3).contiguous().reshape(-1,9)
        return q,cdf
    # General diagnostic fallback. Real fixed-FOV phases use the fast path.
    sizes=[sum(n*(s+1)//streams-n*s//streams for n in counts) for s in range(streams)];width=max(sizes)
    device=items[0][0].device;dummy=torch.tensor([0,0,0,0,4096,4096,4096,4096,4096],device=device,dtype=torch.int32)
    qs=[];cs=[]
    for s,size in enumerate(sizes):
        qparts=[q[n*s//streams:n*(s+1)//streams] for (q,_),n in zip(items,counts)]
        cparts=[c[n*s//streams:n*(s+1)//streams] for (_,c),n in zip(items,counts)]
        qs.append(torch.cat([*qparts,torch.zeros(width-size,device=device,dtype=torch.int16)]))
        cs.append(torch.cat([*cparts,dummy.expand(width-size,9)]))
    return torch.stack(qs).reshape(-1),torch.stack(cs).reshape(-1,9)


@torch.no_grad()
def encode_sequence(items,streams=64):
    if not items or len(items)>256:raise ValueError('requires1..256frames')
    if not 0<streams<=65536:raise ValueError('invalid requested streams')
    device=items[0][0].device;clean=[]
    for q,cdf in items:
        q=q.reshape(-1).to(torch.int16).contiguous()
        if not q.is_cuda or q.device!=device or cdf.device!=device or cdf.dtype!=torch.int32 or cdf.shape!=(q.numel(),9):raise ValueError('invalid same-device symbols/CDF')
        clean.append((q,cdf.contiguous()))
    counts=[q.numel() for q,_ in clean];n=sum(counts)
    streams=min(streams,max(1,(max(counts)+255)//256))
    q,cdf=stream_order(clean,streams)
    slots,lengths,status,bypass_counts=extension().encode_reciprocal(q,cdf,streams,division_table(device))
    metadata=torch.stack([lengths,status,bypass_counts]).cpu().numpy()
    if metadata[1].any():raise RuntimeError('sequence encoder failed')
    total=int(metadata[0].sum(dtype=np.int64));nb=int(metadata[2].sum(dtype=np.int64))
    payload=extension().pack(slots,lengths,lengths.long().cumsum(0)-lengths,total).cpu().numpy().tobytes()
    magic=MAGIC
    if nb:
        chronological=torch.cat([v for v,_ in clean]);bypass=chronological[(chronological<-3)|(chronological>3)].cpu().numpy().astype('<i2',copy=False)
        if len(bypass)!=nb:raise RuntimeError('sequence bypass count mismatch')
        side=zstd.ZstdCompressor(level=3).compress(bypass.tobytes())
        if os.environ.get('FIARC_SEQUENCE_GOLOMB','0')=='1':
            from .seed_bypass_codec import encode as golomb_encode
            candidate=golomb_encode(bypass.tobytes())
            if len(candidate)<len(side):side=candidate;magic=GOLOMB_MAGIC
    else:side=b''
    if max(n,total,len(side),nb,*counts)>0xffffffff:raise ValueError('sequence too large for framing')
    return HEADER.pack(magic,len(items),streams,n,total,len(side),nb)+np.asarray(counts,dtype='<u4').tobytes()+metadata[0].astype('<u4',copy=False).tobytes()+payload+side


class SequenceDecoder:
    def __init__(self,packet,device):
        if len(packet)<HEADER.size:raise ValueError('short sequence header')
        magic,frames,streams,n,total,side,nb=HEADER.unpack_from(packet)
        if magic not in (MAGIC,GOLOMB_MAGIC) or not 0<frames<=256 or not 0<streams<=65536 or nb>n or ((not side)!=(not nb)):raise ValueError('invalid sequence header')
        start=HEADER.size+4*(frames+streams)
        if len(packet)!=start+total+side:raise ValueError('invalid sequence packet length')
        self.counts=np.frombuffer(packet,dtype='<u4',count=frames,offset=HEADER.size).astype('int64').tolist()
        lens=np.frombuffer(packet,dtype='<u4',count=streams,offset=HEADER.size+4*frames).astype('int64')
        if sum(self.counts)!=n or (lens<4).any() or (lens>0x7fffffff).any() or int(lens.sum())!=total:raise ValueError('invalid sequence lengths')
        if magic==GOLOMB_MAGIC:
            from .seed_bypass_codec import decode as golomb_decode
            raw=golomb_decode(packet[start+total:],nb)
        else:raw=zstd.ZstdDecompressor().decompress(packet[start+total:]) if side else b''
        if len(raw)!=nb*2:raise ValueError('invalid sequence bypass')
        self.data=torch.from_numpy(np.frombuffer(packet,dtype='uint8',count=total,offset=start).copy()).to(device)
        self.lengths=torch.from_numpy(lens.astype('int32')).to(device);self.offsets=self.lengths.long().cumsum(0)-self.lengths
        self.states=torch.zeros_like(self.lengths);self.cursors=torch.zeros_like(self.lengths)
        self.bypass=torch.from_numpy(np.frombuffer(raw,dtype='<i2').copy()).to(device)
        self.index=0;self.bypass_at=0

    @torch.no_grad()
    def decode_frame(self,cdf):
        if self.index>=len(self.counts):raise ValueError('sequence already finished')
        if cdf.device!=self.data.device or cdf.dtype!=torch.int32 or cdf.shape!=(self.counts[self.index],9):raise ValueError('wrong current-frame CDF')
        last=self.index+1==len(self.counts)
        q,status=decoder_extension().decode_frame(self.data,self.lengths,self.offsets,cdf.contiguous(),self.states,self.cursors,self.index==0,last)
        if bool(status.any()):raise ValueError('sequence entropy state/cursor failure')
        rare=q==-32768;count=int(rare.sum())
        if self.bypass_at+count>self.bypass.numel():raise ValueError('exhausted bypass')
        if count:q[rare]=self.bypass[self.bypass_at:self.bypass_at+count]
        self.bypass_at+=count;self.index+=1
        if last and self.bypass_at!=self.bypass.numel():raise ValueError('unused bypass values')
        return q
