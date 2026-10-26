"""Explicit GPU resource lifetime; fast arithmetic and packing are unchanged."""
import gc
from functools import wraps
import torch

class Graph:
    """CUDA graph with explicit release of its private pool and tensor owners."""
    def __init__(self,fn,args):
        self.inputs=[];self.graph=None;self.output=None
        try:
            self.inputs=[x.clone() for x in args]
            for _ in range(3):fn(*self.inputs)
            torch.cuda.synchronize();self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):self.output=fn(*self.inputs)
        except BaseException:
            self.close();raise
    def __call__(self,*args):
        if self.graph is None:raise RuntimeError('CUDA graph is closed')
        for target,source in zip(self.inputs,args):target.copy_(source)
        self.graph.replay();return self.output
    def close(self):
        if self.graph is not None:self.graph.reset()
        self.graph=None;self.inputs=[];self.output=None

def close_runtime(runtime):
    if runtime is None:return
    # Public clear() may follow an asynchronous replay. Retire device work
    # before destroying pools; this is a lifecycle boundary, not a frame sync.
    torch.cuda.synchronize()
    for graph in getattr(runtime,'_memory_graphs',[]):graph.close()
    codec=getattr(runtime,'codec',None)
    for graph in getattr(codec,'_memory_graphs',[]):graph.close()

def release_caches():
    from fiarc_stream_order import clear_plan_cache
    from xero._primitives.checkerboard_entropy import even_indices,odd_indices
    clear_plan_cache();even_indices.cache_clear();odd_indices.cache_clear()
    gc.collect();torch.cuda.empty_cache()

def release_on_failure(fn):
    @wraps(fn)
    def guarded(self,*args,**kwargs):
        try:return fn(self,*args,**kwargs)
        except BaseException:
            self.clear();raise
    return guarded
