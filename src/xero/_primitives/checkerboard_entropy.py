"""Two-phase entropy coding; all neural predictions remain spatially parallel."""
import struct
import math
from functools import lru_cache
import torch
from torch.nn import functional as F

HEADER=struct.Struct('<8sQ');MAGIC=b'FSCHECK1'
_compiled_adjustment=None

def labels(q):return torch.where((q>=-3)&(q<=3),q.long()+3,7)

def odd_sites(tensor):
    h,w=tensor.shape[-2:]
    return (torch.arange(h,device=tensor.device)[:,None]+torch.arange(w,device=tensor.device)[None,:])%2==1

def checkerboard_context(labels):
    h,w=labels.shape[-2:];p=F.pad(labels,(1,1,1,1),value=3)
    neighbors=[p[...,0:h,1:w+1],p[...,2:h+2,1:w+1],p[...,1:h+1,0:w],p[...,1:h+1,2:w+2]]
    count=sum((n!=3).long() for n in neighbors)
    votes=sum(((n>=4)&(n<=6)).long()-(n<=2).long() for n in neighbors)
    return torch.where(odd_sites(labels),count*9+votes+4,45)

def adjustment(logits,mask,seed_q,q,offsets):
    context=checkerboard_context(labels(torch.where(mask,seed_q,q)))
    return logits+offsets[context]

def enable_compiled_entropy():
    global _compiled_adjustment
    if _compiled_adjustment is None:_compiled_adjustment=torch.compile(adjustment)

def enable_native_entropy():
    global _compiled_adjustment
    from .checkerboard_entropy_cuda import adjustment as native
    _compiled_adjustment=native

def adjusted_logits(logits,mask,seed_q,q,offsets):
    fn=_compiled_adjustment if _compiled_adjustment is not None else adjustment
    return fn(logits,mask,seed_q,q,offsets)


@lru_cache(maxsize=16)
def odd_indices(shape,device):
    # Geometry only, constructed once per shape/device outside timed warmup.
    # CPU selection here has no per-frame CUDA dynamic-size synchronization.
    h,w=shape[-2:];index=torch.arange(math.prod(shape),device='cpu')
    return index[(((index//w)%h)+(index%w))%2==1].to(device)


def fixed_odd_symbols(q,logits):
    index=odd_indices(tuple(q.shape),q.device)
    return q.reshape(-1).index_select(0,index),logits.reshape(-1,8).index_select(0,index)

@lru_cache(maxsize=16)
def even_indices(shape,device):
    h,w=shape[-2:];index=torch.arange(math.prod(shape),device='cpu')
    return index[(((index//w)%h)+(index%w))%2==0].to(device)

def final_symbol_inputs(q,logits,mask,seed_q,offsets,select,odd_seed_free=False,all_sites_present=False):
    adjusted=adjusted_logits(logits,mask,seed_q,q,offsets)
    odd=odd_sites(q);inputs=[]
    for i,phase in enumerate((~odd,odd)):
        if all_sites_present:
            index=(odd_indices if i else even_indices)(tuple(q.shape),q.device)
            values=q.reshape(-1).index_select(0,index),adjusted.reshape(-1,8).index_select(0,index)
        else:values=fixed_odd_symbols(q,adjusted) if i==1 and odd_seed_free else select(q,adjusted,(~mask)&phase)
        inputs.append(values)
    return inputs

def encode_final(q,logits,mask,seed_q,offsets,backend,streams,encode_symbols,select,odd_seed_free=False,all_sites_present=False):
    packets=[encode_symbols(*values,backend,max(1,streams//2)) for values in
        final_symbol_inputs(q,logits,mask,seed_q,offsets,select,odd_seed_free,all_sites_present)]
    return HEADER.pack(MAGIC,len(packets[0]))+b''.join(packets)

def decode_final(packet,logits,mask,seed_q,offsets,backend,decode_symbols,odd_seed_free=False,all_sites_present=False):
    if len(packet)<HEADER.size:raise ValueError('short checkerboard packet')
    magic,n=HEADER.unpack_from(packet)
    if magic!=MAGIC or n<=0 or HEADER.size+n>=len(packet):raise ValueError('bad checkerboard framing')
    packets=[packet[HEADER.size:HEADER.size+n],packet[HEADER.size+n:]]
    q=torch.zeros_like(mask,dtype=torch.int16);odd=odd_sites(mask)
    for index,(data,phase) in enumerate(zip(packets,(~odd,odd))):
        # During first phase context is constant45. Second phase reads only
        # already decoded even neighbors (and transmitted seeds).
        valid=(~mask)&phase
        direct=all_sites_present or (index==1 and odd_seed_free)
        if direct:
            indices=(odd_indices if index else even_indices)(tuple(q.shape),q.device)
            adjusted=adjusted_logits(logits,mask,seed_q,q,offsets).reshape(-1,8).index_select(0,indices)
        else:adjusted=logits[valid]+offsets[45] if index==0 else adjusted_logits(logits,mask,seed_q,q,offsets)[valid]
        decoded=decode_symbols(data,adjusted,backend,logits.device)
        if direct:q.reshape(-1).index_copy_(0,indices,decoded)
        else:q[valid]=decoded
    return q.reshape(-1) if all_sites_present else q[~mask]
