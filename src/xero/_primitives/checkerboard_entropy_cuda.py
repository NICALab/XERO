"""Exact context plus FP32 offset addition; no new probability approximation."""
import torch
from .seed_rans_gpu import extension

@torch.no_grad()
def adjustment(logits,mask,seed_q,q,offsets):
    return extension().checker_adjust(logits.float().contiguous(),mask.bool().contiguous(),
        seed_q.to(torch.int16).contiguous(),q.to(torch.int16).contiguous(),offsets.float().contiguous())
