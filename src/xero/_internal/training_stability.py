"""Explicit, recorded controls for post-collapse training comparisons."""
import math
import torch
from torch.nn import functional as F


def learning_rate(step,total,base=1e-4,floor=2e-5,decay_start=50000):
    if not (math.isfinite(base) and math.isfinite(floor) and 0<floor<=base) or decay_start<0:
        raise ValueError('Invalid learning rate schedule')
    if step<=decay_start or total<=decay_start:return base
    fraction=min((step-decay_start)/(total-decay_start),1.)
    return floor+(base-floor)*.5*(1+math.cos(math.pi*fraction))


def distortion(prediction,target,scale,first,first_weight=1.):
    if not math.isfinite(first_weight) or not 0<first_weight<=1:
        raise ValueError('First-chunk distortion weight must be in(0,1]')
    error=(prediction-target)/scale
    # Preserve the exact historical operation for the default control.
    if first_weight==1.:
        value=F.smooth_l1_loss(error,torch.zeros_like(target))
        return value,value
    per_sample=F.smooth_l1_loss(error,torch.zeros_like(target),reduction='none').flatten(1).mean(1)
    weights=torch.where(first,first_weight,1.)
    return (per_sample*weights).mean(),per_sample.mean()
