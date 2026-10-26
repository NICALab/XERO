"""Grouped entropy container, unchanged framewise learned-seed predictions."""
import struct
import numpy as np
import torch
import zstandard as zstd
from ficompress.bounded_quantizer_cuda import bounded_symbols
from ficompress.models.grouped_spatial_codec import restore_symbols,ESCAPE
from ficompress.checkerboard_entropy import adjusted_logits,even_indices,odd_indices
from ficompress.seed_rans_gpu import cdf_tensor
from ficompress.seed_sequence_rans import encode_sequence,SequenceDecoder

HEADER=struct.Struct('<8sIII6Q');MAGIC=b'FSEEDSG1'


def validate_codec(codec):
    if codec.selection!='lattice' or codec.grid_stride%2 or getattr(codec,'seed_entropy_offsets',None) is None or getattr(codec,'final_entropy_offsets',None) is None:
        raise ValueError('grouped codec requires even lattice stride and both conditional probability tables')


def phase_indices(value,odd):return (odd_indices if odd else even_indices)(tuple(value.shape),value.device)
def select(value,index):return value.reshape(-1).index_select(0,index)
def select_logits(value,index):return value.reshape(-1,8).index_select(0,index)


def final_even_cdf(logits,mask,index):
    cdf=cdf_tensor(select_logits(logits,index));seed=select(mask,index)
    # Decoded seeds already determine these pixels. Their placeholder is a
    # zero-cost no-op, not a probabilistic q0 and not another source symbol.
    dummy=torch.tensor([0,0,0,0,4096,4096,4096,4096,4096],device=cdf.device,dtype=torch.int32)
    return torch.where(seed[:,None],dummy,cdf).contiguous()


@torch.no_grad()
def prepare_frame(codec,source,pred,mean,std,mode='learned'):
    validate_codec(codec)
    if source.shape!=pred.shape or source.shape[:2]!=(1,1):raise ValueError('single grayscale frame required')
    sq,sr=bounded_symbols(source,pred,codec.seed_step,codec.seed_tolerance)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        mask=codec.select(source,pred,sq,mean,std,mode).bool()
        _,sl=codec.prior_logits(pred,mean,std);post,fl=codec.refine(pred,mask.float(),sr,mean,std)
    q,rec=bounded_symbols(source,post,codec.final_step,codec.tolerance)
    if codec.even_refinement_gain:
        post=codec.refine_from_even(post,rec,mask);q,rec=bounded_symbols(source,post,codec.final_step,codec.tolerance)
    stride=codec.grid_stride;transmitted=torch.where(mask,sq,0);low=transmitted[...,::stride,::stride]
    empty=torch.zeros_like(low,dtype=torch.bool)
    seed_logits=adjusted_logits(sl[...,::stride,::stride,:],empty,torch.zeros_like(low),low,codec.seed_entropy_offsets)
    final_logits=adjusted_logits(fl,mask,sq,q,codec.final_entropy_offsets)
    phases=[]
    for odd in (False,True):
        index=phase_indices(low,odd);phases.append((select(low,index),cdf_tensor(select_logits(seed_logits,index))))
    final_q=torch.where(mask,0,q)
    for odd in (False,True):
        index=phase_indices(q,odd)
        cdf=cdf_tensor(select_logits(final_logits,index)) if odd else final_even_cdf(final_logits,mask,index)
        phases.append((select(final_q,index),cdf))
    return dict(phases=phases,source=source,seed_escape=(sq==ESCAPE)&mask,final_escape=(q==ESCAPE)&~mask),rec


@torch.no_grad()
def encode_group(prepared,streams=64):
    if not prepared or len(prepared)>256:raise ValueError('requires1..256frames')
    shape=prepared[0]['source'].shape
    if any(v['source'].shape!=shape for v in prepared):raise ValueError('inconsistent group shapes')
    packets=[encode_sequence([v['phases'][phase] for v in prepared],streams) for phase in range(4)]
    source=torch.stack([v['source'] for v in prepared]);zc=zstd.ZstdCompressor(level=3)
    for key in ('seed_escape','final_escape'):
        values=source[torch.stack([v[key] for v in prepared])].cpu().numpy().astype('<f4',copy=False)
        packets.append(zc.compress(values.tobytes()))
    return HEADER.pack(MAGIC,len(prepared),shape[-2],shape[-1],*[len(v) for v in packets])+b''.join(packets)


class GroupDecoder:
    def __init__(self,packet,device):
        if len(packet)<HEADER.size:raise ValueError('short group header')
        magic,self.frames,self.h,self.w,*lengths=HEADER.unpack_from(packet)
        if magic!=MAGIC or not 0<self.frames<=256 or not self.h or not self.w or HEADER.size+sum(lengths)!=len(packet):raise ValueError('invalid group header')
        at=HEADER.size;parts=[]
        for n in lengths:parts.append(packet[at:at+n]);at+=n
        self.phases=[SequenceDecoder(part,device) for part in parts[:4]]
        if any(len(v.counts)!=self.frames for v in self.phases):raise ValueError('group phase count mismatch')
        self.escape=[];self.escape_at=[0,0];self.index=0
        for packet in parts[4:]:
            raw=zstd.ZstdDecompressor().decompress(packet)
            if len(raw)%4:raise ValueError('invalid float32 escape stream')
            self.escape.append(torch.from_numpy(np.frombuffer(raw,dtype='<f4').copy()).to(device))

    def take_escape(self,q,stage):
        n=int((q==ESCAPE).sum());at=self.escape_at[stage]
        if at+n>self.escape[stage].numel():raise ValueError('exhausted float32 escape stream')
        self.escape_at[stage]+=n
        return self.escape[stage][at:at+n]

    @torch.no_grad()
    def decode_frame(self,codec,pred,mean,std):
        validate_codec(codec)
        if self.index>=self.frames or pred.shape!=(1,1,self.h,self.w):raise ValueError('invalid group frame/shape')
        stride=codec.grid_stride
        with torch.autocast('cuda',dtype=torch.bfloat16):_,sl=codec.prior_logits(pred,mean,std)
        low=torch.zeros_like(pred[...,::stride,::stride],dtype=torch.int16).contiguous();empty=torch.zeros_like(low,dtype=torch.bool)
        low_logits=sl[...,::stride,::stride,:]
        for odd in (False,True):
            index=phase_indices(low,odd)
            logits=low_logits+codec.seed_entropy_offsets[45] if not odd else adjusted_logits(low_logits,empty,torch.zeros_like(low),low,codec.seed_entropy_offsets)
            values=self.phases[int(odd)].decode_frame(cdf_tensor(select_logits(logits,index)))
            low.reshape(-1).index_copy_(0,index,values)
        # low is compact: indexed writes above must not target a reshape copy.
        sq=torch.zeros_like(pred,dtype=torch.int16);sq[...,::stride,::stride]=low
        mask=sq!=0;sr=pred.clone()
        sr[...,::stride,::stride]=restore_symbols(pred[...,::stride,::stride],low,self.take_escape(low,0),codec.seed_step)
        with torch.autocast('cuda',dtype=torch.bfloat16):post,fl=codec.refine(pred,mask.float(),sr,mean,std)
        q=torch.zeros_like(pred,dtype=torch.int16)
        for odd in (False,True):
            index=phase_indices(q,odd)
            logits=fl+codec.final_entropy_offsets[45] if not odd else adjusted_logits(fl,mask,sq,q,codec.final_entropy_offsets)
            cdf=cdf_tensor(select_logits(logits,index)) if odd else final_even_cdf(logits,mask,index)
            q.reshape(-1).index_copy_(0,index,self.phases[2+int(odd)].decode_frame(cdf))
        escapes=self.take_escape(q,1);rec=restore_symbols(post,q,escapes,codec.final_step)
        if codec.even_refinement_gain:
            post=codec.refine_from_even(post,rec,mask);rec=restore_symbols(post,q,escapes,codec.final_step)
        self.index+=1
        if self.index==self.frames and any(at!=v.numel() for at,v in zip(self.escape_at,self.escape)):raise ValueError('unused float32 escapes')
        return rec
