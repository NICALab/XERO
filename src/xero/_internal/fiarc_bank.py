"""Frozen-teacher manifest and balanced, double-buffered GPU training tile cache."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import h5py
import numpy as np
import torch

def resumed_generation(offset,start_step,refresh):
    if offset<0 or start_step<0 or refresh<1:raise ValueError('Invalid resumed cache position')
    return offset+start_step//refresh

class TeacherBank:
    def __init__(self,path,allow_smoke=False):
        self.path=Path(path);self.manifest=json.loads((self.path/'complete.json').read_text())
        if self.manifest['plan'].get('smoke_only') and not allow_smoke:raise ValueError('Software-smoke bank is not a production training source')
        self.movies=self.manifest['movies'];self.train=[m for m in self.movies if m['role']=='train']
        self.calibration=[m for m in self.movies if m['role']=='calibration'];self.test=[m for m in self.movies if m['role']=='test']
        if not self.train or not self.calibration:raise ValueError('Train and calibration windows required')
        for m in self.movies:
            with h5py.File(m['path']) as f:
                d=f['data'];assert list(d.shape)==m['shape'] and d.dtype==np.float32
            if m['checkpoint_sha256']!=self.manifest['plan']['checkpoint_sha256']:raise ValueError('Mixed teachers')
        # Prohibit source-time overlap between roles within a recording.
        for i,a in enumerate(self.movies):
            for b in self.movies[:i]:
                if a['record']==b['record'] and a['role']!=b['role']:
                    ae=a['source_frame_start']+a['shape'][0];be=b['source_frame_start']+b['shape'][0]
                    if max(a['source_frame_start'],b['source_frame_start'])<min(ae,be):raise ValueError('Cross-role source overlap')

    def normalization(self):
        from normalization_stats import statistics
        sums=squares=0.;count=0
        for m in self.train:
            # Exact same samples/reduction order; derived cache avoids repeated HDF reads.
            (s,ss,n),_=statistics(m,self.path/'.normalization_cache_v1')
            sums+=s;squares+=ss;count+=n
        mean=sums/count;std=np.sqrt(max(squares/count-mean*mean,0))
        if not np.isfinite(mean+std) or std<=0:raise ValueError('Invalid training statistics')
        return float(mean),float(std)

class BankTileCache:
    def __init__(self,bank,device,seed=41,patch=128,tiles=32,refresh=1000,batch=32,history=64,rollout=5,temporal=200,generation_offset=0,start_step=0):
        self.bank=bank;self.device=device;self.seed=seed;self.patch=patch;self.tiles=tiles;self.refresh=refresh;self.batch=batch
        self.history=history;self.rollout=rollout;self.temporal=min(temporal,min(m['shape'][0] for m in bank.train))
        self.sy=min(patch+16,min(m['shape'][1] for m in bank.train));self.sx=min(patch+16,min(m['shape'][2] for m in bank.train))
        if min(self.sy,self.sx)<patch or self.temporal<history+rollout+1:raise ValueError('Movie too small for temporal/spatial input')
        if generation_offset<0:raise ValueError('Negative cache generation offset')
        self.pool=ThreadPoolExecutor(1);self.generation=resumed_generation(generation_offset,start_step,refresh)
        self.rng=torch.Generator(device=device).manual_seed(seed+start_step)
        self.ready=self.pool.submit(self.load,self.generation);self.slab=None;self.meta=[];self.load_seconds=0

    def plan(self,generation):
        order=np.random.default_rng(self.seed).permutation(len(self.bank.train));rng=np.random.default_rng(self.seed+generation*10007)
        result=[]
        for slot in range(self.tiles):
            m=self.bank.train[int(order[(generation*self.tiles+slot)%len(order)])]
            t=int(rng.integers(m['shape'][0]-self.temporal+1));y=int(rng.integers(m['shape'][1]-self.sy+1));x=int(rng.integers(m['shape'][2]-self.sx+1))
            result.append(dict(path=m['path'],record=m['record'],role=m['role'],t=t,y=y,x=x,source_frame_start=m['source_frame_start']+t))
        return result

    def load(self,generation):
        tic=time.monotonic();meta=self.plan(generation)
        cpu=torch.empty((self.tiles,self.temporal,self.sy,self.sx),dtype=torch.float32,pin_memory=True);arr=cpu.numpy()
        for i,m in enumerate(meta):
            with h5py.File(m['path'],'r',rdcc_nbytes=4*1024**2) as f:
                arr[i]=f['data'][m['t']:m['t']+self.temporal,m['y']:m['y']+self.sy,m['x']:m['x']+self.sx]
        if not np.isfinite(arr).all():raise ValueError('Nonfinite frozen target')
        return cpu,meta,time.monotonic()-tic

    def next(self,step):
        wait=0.;refreshed=False
        if self.slab is None or (step-1)%self.refresh==0:
            tic=time.monotonic();cpu,self.meta,self.load_seconds=self.ready.result();wait=time.monotonic()-tic
            self.slab=cpu.to(self.device,non_blocking=True);self.generation+=1;self.ready=self.pool.submit(self.load,self.generation);refreshed=True
        b=self.batch;p=self.patch;length=self.history+self.rollout+1;g=self.rng;dev=self.device
        slot=torch.randint(self.tiles,(b,),device=dev,generator=g);t=torch.randint(self.temporal-length+1,(b,),device=dev,generator=g)
        y=torch.randint(self.sy-p+1,(b,),device=dev,generator=g);x=torch.randint(self.sx-p+1,(b,),device=dev,generator=g)
        yy=y[:,None]+torch.arange(p,device=dev);xx=x[:,None]+torch.arange(p,device=dev)
        yy=torch.where(torch.rand(b,device=dev,generator=g)[:,None]<.5,yy.flip(1),yy)
        xx=torch.where(torch.rand(b,device=dev,generator=g)[:,None]<.5,xx.flip(1),xx)
        seq=self.slab[slot[:,None,None,None],t[:,None,None,None]+torch.arange(length,device=dev)[None,:,None,None],yy[:,None,:,None],xx[:,None,None,:]]
        return seq,wait,refreshed

    def close(self):self.pool.shutdown(wait=True)
