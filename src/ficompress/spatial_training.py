"""Actual causal spatial forward pass plus a detached-state training gradient.

Forward uses the deployed CPU quantizer, including strict float64 error tests.
Backward holds the corrected spatial state fixed. This is a surrogate, not an
exact derivative of hard quantization or of the entropy coder.
"""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch.nn import functional as F
from .context_rans import _lib,spatial_quantize,spatial_predictions

OFFSETS=[(0,-1),(0,-2),(0,-3),(-1,-2),(-1,-1),(-1,0),(-1,1),(-1,2),(-2,-1),(-2,0),(-2,1),(-3,0)]

def detached_spatial_surrogate(prediction,reconstruction,weights,kernel=None):
    if kernel is None:
        array=np.zeros((1,1,4,6),dtype='float32')
        for (dy,dx),weight in zip(OFFSETS,weights):array[0,0,dy+3,dx+3]=float(weight)
        kernel=torch.from_numpy(array).to(prediction.device)
    error=reconstruction.detach().float()-prediction.float()
    return prediction.float()+F.conv2d(F.pad(error,(3,2,3,0)),kernel)+float(weights[12])

class ActualSpatialTraining:
    def __init__(self,weights,workers=4,backend='cpu',gradient_mode='detached'):
        self.weights=np.asarray(weights,dtype='float64')
        if self.weights.shape!=(13,) or not np.isfinite(self.weights).all():raise ValueError('13 finite weights required')
        if backend not in ('cpu','gpu'):raise ValueError('unknown spatial backend')
        if gradient_mode not in ('detached','identity'):raise ValueError('unknown spatial gradient approximation')
        self.gradient_mode=gradient_mode
        self.backend=backend;self.gpu_weights={};self.kernels={}
        if backend=='cpu':_lib()
        self.pool=ThreadPoolExecutor(max_workers=workers) if backend=='cpu' else None

    def _surrogate(self,prediction,r):
        # Alternative local fixed-symbol approximation. This is NOT the exact
        # derivative through discontinuous symbol selection; forward is identical.
        if self.gradient_mode=='identity':return prediction.float()
        device=prediction.device
        if device not in self.kernels:
            a=np.zeros((1,1,4,6),dtype='float32')
            for (dy,dx),w in zip(OFFSETS,self.weights):a[0,0,dy+3,dx+3]=w
            self.kernels[device]=torch.from_numpy(a).to(device)
        return detached_spatial_surrogate(prediction,r,self.weights,self.kernels[device])

    def __call__(self,source,prediction,gradient=False):
        if self.backend=='gpu':
            from .spatial_gpu import spatial_quantize_gpu
            if prediction.device not in self.gpu_weights:self.gpu_weights[prediction.device]=torch.as_tensor(self.weights,device=prediction.device)
            q,r,actual=spatial_quantize_gpu(source,prediction,self.gpu_weights[prediction.device])
            if not gradient:return r
            surrogate=self._surrogate(prediction,r)
            return actual+(surrogate-surrogate.detach()),r,float((q!=0).float().mean())
        s=source.detach().float().cpu().numpy()[:,0]
        p=prediction.detach().float().cpu().numpy()[:,0]
        def one(pair):
            a,b=pair;q,r=spatial_quantize(a,b,gain=1.,weights=self.weights)
            sp=spatial_predictions(b,r,self.weights) if gradient else None
            return q,r,sp
        results=list(self.pool.map(one,zip(s,p)))
        r=torch.from_numpy(np.stack([v[1] for v in results])[:,None]).to(prediction.device)
        if not gradient:return r
        actual=torch.from_numpy(np.stack([v[2] for v in results])[:,None]).to(prediction.device)
        surrogate=self._surrogate(prediction,r)
        output=actual+(surrogate-surrogate.detach())
        fraction=sum(np.count_nonzero(v[0]) for v in results)/s.size
        return output,r,float(fraction)

    def close(self):
        if self.pool:self.pool.shutdown(wait=True)
