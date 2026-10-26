"""Bit-identical fixed-count uneven stream packing; opt-in release adapter.

Preserve each stream's chronological segments and terminal dummy padding.
Do NOT pad individual frames: that would change the rANS state sequence.
Equal partitions and variable-count inputs retain the released implementation.
"""
from functools import lru_cache
import torch


@lru_cache(maxsize=4)
def _plan(frames, count, streams, device):
    dev=torch.device(device)
    starts=torch.arange(streams,device=dev,dtype=torch.int64)*count//streams
    ends=torch.arange(1,streams+1,device=dev,dtype=torch.int64)*count//streams
    lengths=ends-starts
    # Integer ceiling known on CPU; no GPU scalar/shape synchronization.
    width=frames*((count+streams-1)//streams)
    offset=torch.arange(width,device=dev,dtype=torch.int64)[None,:]
    nonzero=lengths.clamp_min(1)[:,None]
    valid=offset<(frames*lengths)[:,None]
    index=(offset//nonzero)*count+starts[:,None]+offset%nonzero
    index=torch.where(valid,index,0).reshape(-1)
    valid=valid.reshape(-1)
    dummy=torch.tensor([0,0,0,0,4096,4096,4096,4096,4096],device=dev,dtype=torch.int32)
    return index,valid,dummy


def fixed_uneven_order(items,streams):
    """Called only for positive, equal frame counts not divisible by streams."""
    frames=len(items);count=items[0][0].numel()
    if streams<1 or count<1 or any(q.numel()!=count for q,_ in items):
        raise ValueError('Fixed positive counts and streams required')
    index,valid,dummy=_plan(frames,count,streams,str(items[0][0].device))
    q=torch.stack([q for q,_ in items]).reshape(-1)
    cdf=torch.stack([c for _,c in items]).reshape(-1,9)
    q=q.index_select(0,index)
    cdf=cdf.index_select(0,index)
    return torch.where(valid,q,0),torch.where(valid[:,None],cdf,dummy)


def adapter(reference):
    def stream_order(items,streams):
        counts=[q.numel() for q,_ in items]
        if counts and counts[0]>0 and len(set(counts))==1 and counts[0]%streams:
            return fixed_uneven_order(items,streams)
        return reference(items,streams)
    return stream_order


def clear_plan_cache():
    _plan.cache_clear()
