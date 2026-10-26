"""Short-stack input adapter. Frozen network/loss/optimizer remain unchanged.

Only absent pre-sequence history is masked. Never wrap time, repeat source frames
as training targets, join recordings, or read another role to fill context.
"""
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch


def causal_indices(target,length):
    indices=target[:,None]-length+1+torch.arange(length,device=target.device)[None]
    return indices.clamp_min(0),indices>=0


def make_cache_class(base):
    class BoundaryCache(base):
        def __init__(self,bank,device,seed=41,patch=128,tiles=32,refresh=1000,batch=32,
                     history=64,rollout=64,temporal=200,generation_offset=0,start_step=0):
            self.bank=bank;self.device=device;self.seed=seed;self.patch=patch;self.tiles=tiles;self.refresh=refresh;self.batch=batch
            self.history=history;self.rollout=rollout;self.temporal=min(temporal,min(m['shape'][0] for m in bank.train))
            self.sy=min(patch+16,min(m['shape'][1] for m in bank.train));self.sx=min(patch+16,min(m['shape'][2] for m in bank.train))
            if min(self.sy,self.sx)<patch or self.temporal<2:raise ValueError('Insufficient native training movie')
            if generation_offset<0:raise ValueError('Negative cache generation')
            self.mean=bank.normalization()[0]
            self.pool=ThreadPoolExecutor(1);self.generation=generation_offset+start_step//refresh
            self.rng=torch.Generator(device=device).manual_seed(seed+start_step)
            self.ready=self.pool.submit(self.load,self.generation);self.slab=None;self.meta=[];self.load_seconds=0

        def next(self,step):
            wait=0.;refreshed=False
            if self.slab is None or (step-1)%self.refresh==0:
                tic=time.monotonic();cpu,self.meta,self.load_seconds=self.ready.result();wait=time.monotonic()-tic
                self.slab=cpu.to(self.device,non_blocking=True);self.generation+=1
                self.ready=self.pool.submit(self.load,self.generation);refreshed=True
            b,p,g,dev=self.batch,self.patch,self.rng,self.device
            slot=torch.randint(self.tiles,(b,),device=dev,generator=g)
            target=torch.randint(1,self.temporal,(b,),device=dev,generator=g)
            first=torch.rand(b,device=dev,generator=g)<.01
            target=torch.where(first,0,target);self.boundary_target=target
            tt,valid=causal_indices(target,self.history+self.rollout+1)
            y=torch.randint(self.sy-p+1,(b,),device=dev,generator=g);x=torch.randint(self.sx-p+1,(b,),device=dev,generator=g)
            yy=y[:,None]+torch.arange(p,device=dev);xx=x[:,None]+torch.arange(p,device=dev)
            yy=torch.where(torch.rand(b,device=dev,generator=g)[:,None]<.5,yy.flip(1),yy)
            xx=torch.where(torch.rand(b,device=dev,generator=g)[:,None]<.5,xx.flip(1),xx)
            seq=self.slab[slot[:,None,None,None],tt[:,:,None,None],yy[:,None,:,None],xx[:,None,None,:]]
            seq=torch.where(valid[:,:,None,None],seq,self.mean)
            return seq,wait,refreshed
    return BoundaryCache


@torch.no_grad()
def history_for_target(seq,target_index,history,rollout,mean,std,tol,predictor,net,codec,
                       forecast,bounded,advance,autocast):
    hist=seq[:,:history]
    _,hist=bounded(hist,torch.full_like(hist,mean),codec.final_step,tol)
    for k in range(rollout):
        source_index=target_index-rollout+k
        valid=source_index>=0;first=source_index==0
        with autocast():
            pred=forecast(predictor,hist,mean,std,1,first)
            v=net(seq[:,history+k:history+k+1],pred,mean,std,'learned',False)
        next_hist=advance(hist,v['rec'])
        # Deployment repeats the first decoded frame to initialize valid history.
        next_hist=torch.where(first[:,None,None,None],v['rec'].expand_as(next_hist),next_hist)
        hist=torch.where(valid[:,None,None,None],next_hist,hist)
    return hist,seq[:,-1:],target_index==0
