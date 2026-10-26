"""Decoder-reproducible open-loop forecast for a jointly encoded chunk.

Previous corrected history is available; current source is NEVER available to
this function. Only after decoding the entire chunk is actual corrected state
committed. The innovation decoder jointly refines all T predicted frames.
"""
import torch


def forecast_chunk(base,history,mean,std,frames,first):
    first=torch.as_tensor(first,device=history.device,dtype=torch.bool)
    if first.ndim==0:first=first.expand(len(history))
    if first.shape!=(len(history),):raise ValueError('one reset flag per sample required')
    local=torch.where(first[:,None,None,None],mean,history)
    predictions=[]
    for i in range(frames):
        p=base((local-mean)/std)*std+mean
        if i==0:p=torch.where(first[:,None,None,None],mean,p)
        predictions.append(p)
        local=torch.cat([local[:,1:],p],1)
    return torch.cat(predictions,1)


def corrected_history(history,reconstruction):
    return torch.cat([history,reconstruction],1)[:,-history.shape[1]:]
