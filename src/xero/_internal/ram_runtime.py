import torch

class PinnedSource:
    def __init__(self,host,group=64):
        if host.device.type!='cpu' or not host.is_pinned() or host.dtype!=torch.float32 or host.ndim!=3:
            raise ValueError('Pinned contiguous float32 TYX CPU source required')
        if not host.is_contiguous() or group!=64:raise ValueError('Use the frozen64-frame entropy group')
        self.host=host;self.group=group;self.n=len(host)
        self.buffers=[torch.empty((min(group,self.n),*host.shape[1:]),device='cuda',dtype=torch.float32) for _ in range(2)]
        self.transfer=torch.cuda.Stream();self.ready=[torch.cuda.Event(),torch.cuda.Event()]
        self.reset()

    def __len__(self):return self.n

    def reset(self):
        self.next=0;self.loaded=[-1,-1]

    def load(self,start,free=None):
        slot=start//self.group%2;count=min(self.group,self.n-start)
        with torch.cuda.stream(self.transfer):
            if free is not None:self.transfer.wait_event(free)
            self.buffers[slot][:count].copy_(self.host[start:start+count],non_blocking=True)
            self.ready[slot].record(self.transfer)
        self.loaded[slot]=start

    def __getitem__(self,index):
        if index!=self.next or not 0<=index<self.n:raise ValueError('Causal ordered source access required')
        self.next+=1;start=index//self.group*self.group;slot=start//self.group%2
        if index==0:self.load(0)
        if self.loaded[slot]!=start:raise RuntimeError('Missing prefetch')
        current=torch.cuda.current_stream()
        if index==start:
            current.wait_event(self.ready[slot])
            if start+self.group<self.n:
                # Previous group's entropy encode already consumed its source.
                free=torch.cuda.Event();free.record(current)
                self.load(start+self.group,free)
        return self.buffers[slot][index-start]


class PinnedOutput:
    def __init__(self,frames,shape):
        self.host=torch.empty((frames,*shape),dtype=torch.float32,pin_memory=True)
        self.buffers=[torch.empty(shape,device='cuda',dtype=torch.float32) for _ in range(2)]
        self.transfer=torch.cuda.Stream();self.done=[torch.cuda.Event(),torch.cuda.Event()]
        self.used=[False,False];self.next=0

    def reset(self):self.used=[False,False];self.next=0

    def put(self,index,rec):
        if index!=self.next or index>=len(self.host):raise ValueError('Unexpected output order')
        self.next+=1;slot=index%2;current=torch.cuda.current_stream()
        if self.used[slot]:current.wait_event(self.done[slot])
        # Own a stable copy; decoder scratch/cudagraph buffers may be reused.
        self.buffers[slot].copy_(rec[0,0])
        ready=torch.cuda.Event();ready.record(current)
        with torch.cuda.stream(self.transfer):
            self.transfer.wait_event(ready)
            self.host[index].copy_(self.buffers[slot],non_blocking=True)
            self.done[slot].record(self.transfer)
        self.used[slot]=True

    def finish(self):
        self.transfer.synchronize()
        if self.next!=len(self.host):raise ValueError('Incomplete RAM output')
