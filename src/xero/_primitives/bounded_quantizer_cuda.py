"""Fused inference-only hard quantization, checked against the torch reference."""
import torch
from .seed_rans_gpu import extension
from .models.grouped_spatial_codec import STEP

@torch.no_grad()
def bounded_symbols(source,prediction,step=STEP,tolerance=.01):
    return tuple(extension().bounded(source.float().contiguous(),prediction.float().contiguous(),step,tolerance))


@torch.no_grad()
def bounded_symbols_after_even(source,prediction,mask,previous_q,previous_rec,step=STEP,tolerance=.01):
    """Only after a refinement preserving all even/seed predictions exactly."""
    return tuple(extension().bounded_after_even(source.float().contiguous(),prediction.float().contiguous(),mask.bool().contiguous(),
        previous_q.short().contiguous(),previous_rec.float().contiguous(),step,tolerance))


bounded_symbols.after_even=bounded_symbols_after_even
