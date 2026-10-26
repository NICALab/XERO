"""Experimental integer CDF and independent CPU sequence decoder.

New arithmetic profile; NOT compatible with old floating-CDF packets.
No source pixels, oracle symbols, or encoder CDFs are used by the reader.
"""
import struct
from functools import lru_cache
import numpy as np
import torch
import zstandard as zstd
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@lru_cache(None)
def exp_table(device):
    # Freeze these integer values in the eventual profile, not backend exp().
    values=np.maximum(1,np.rint(np.exp(-np.arange(257,dtype=np.float64)/16)*65536)).astype('int32')
    return torch.tensor(values,device=device,dtype=torch.int32)


def canonical_cdf(logits):
    v=torch.round(logits.float().reshape(-1,8)*16).to(torch.int32)
    delta=(v.max(-1,keepdim=True).values-v).clamp(0,256)
    weights=exp_table(str(logits.device))[delta.long()]
    freq=torch.div(weights*4088,weights.sum(-1,keepdim=True),rounding_mode='floor').to(torch.int32)+1
    remainder=4096-freq.sum(-1,keepdim=True)
    freq.scatter_add_(1,v.argmax(-1,keepdim=True),remainder.to(torch.int32))
    zero=torch.zeros((len(v),1),device=v.device,dtype=torch.int32)
    return torch.cat((zero,freq.cumsum(-1,dtype=torch.int32)),1).contiguous()


@triton.jit
def cdf_kernel(X,T,C,N:tl.constexpr,ROWS:tl.constexpr):
    r=tl.program_id(0)*ROWS+tl.arange(0,ROWS);k=tl.arange(0,8)
    x=tl.load(X+r[:,None]*8+k[None,:],r[:,None]<N,other=0).to(tl.float32)
    rounded=libdevice.nearbyint(x*16)
    # Make out-of-range conversion explicit: CPU PyTorch uses INT_MIN, while
    # CUDA float->int conversion can saturate positive overflow to INT_MAX.
    rounded=tl.where((rounded>=-2147483648.)&(rounded<2147483648.),rounded,-2147483648.)
    v=rounded.to(tl.int32);maximum=tl.max(v,1)
    # Match canonical_cdf's INT32 subtraction and clamp(0, 256), including
    # overflow for extreme logits. Without the lower clamp the LUT address can
    # become negative and crash CUDA; this does not change valid-range CDFs.
    delta=tl.maximum(tl.minimum(maximum[:,None]-v,256),0)
    w=tl.load(T+delta);total=tl.sum(w,1)
    f=(w*4088)//total[:,None]+1
    rem=4096-tl.sum(f,1)
    best=tl.min(tl.where(v==maximum[:,None],k[None,:],8),1)
    f+=tl.where(k[None,:]==best[:,None],rem[:,None],0)
    c=tl.cumsum(f,1)
    tl.store(C+r*9,0,r<N)
    tl.store(C+r[:,None]*9+k[None,:]+1,c,r[:,None]<N)


def fast_canonical_cdf(logits):
    if logits.device.type=='cpu':return canonical_cdf(logits)
    v=logits.reshape(-1,8).contiguous();n=len(v)
    result=torch.empty((n,9),device=v.device,dtype=torch.int32)
    cdf_kernel[(triton.cdiv(n,32),)](v,exp_table(str(v.device)),result,n,32,enable_fp_fusion=False)
    return result


class CPUSequenceDecoder:
    LOW=1<<23
    HEADER=struct.Struct('<8s6I')
    def __init__(self,packet,device='cpu',accelerated=True):
        if str(device)!='cpu':raise ValueError('Independent CPU reader requires CPU')
        if len(packet)<32:raise ValueError('Short rANS packet')
        magic,nf,ns,n,total,side,nb=self.HEADER.unpack_from(packet)
        if magic not in (b'SRANSG01',b'SRANSG02') or not 0<nf<=256 or not 0<ns<=65536 or nb>n:raise ValueError('Bad rANS header')
        start=32+4*(nf+ns)
        if start+total+side!=len(packet) or bool(side)!=bool(nb):raise ValueError('Bad rANS framing')
        self.counts=np.frombuffer(packet,'<u4',nf,32).tolist()
        lens=np.frombuffer(packet,'<u4',ns,32+4*nf).tolist()
        if sum(self.counts)!=n or min(lens)<4 or sum(lens)!=total:raise ValueError('Bad rANS lengths')
        self.stream=[];self.state=[];self.cursor=[4]*ns;at=start
        for length in lens:
            data=packet[at:at+length];state=struct.unpack_from('<I',data)[0]
            if not self.LOW<=state<2**31:raise ValueError('Invalid initial state')
            self.stream.append(data);self.state.append(state);at+=length
        if nb:
            if magic==b'SRANSG02':
                from ficompress.seed_bypass_codec import decode
                raw=decode(packet[at:],nb)
            else:raw=zstd.ZstdDecompressor().decompress(packet[at:],max_output_size=nb*2)
        else:raw=b''
        if len(raw)!=nb*2:raise ValueError('Invalid rare symbols')
        self.rare=np.frombuffer(raw,'<i2');self.rare_at=0;self.index=0
        self.accelerated=accelerated
        if accelerated:
            import ctypes
            import fast_packet
            self.reader=ctypes.CDLL(str(fast_packet.SO)).decode_frame_cpu
            self.reader.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
            self.reader.restype=ctypes.c_int
            self.data=np.frombuffer(packet,dtype='uint8',count=total,offset=start)
            self.lengths=np.asarray(lens,dtype='uint32')
            self.offsets=np.r_[0,np.cumsum(lens[:-1])].astype('uint64')
            self.state=np.asarray(self.state,dtype='uint32');self.cursor=np.asarray(self.cursor,dtype='uint32')

    def decode_frame(self,cdf):
        if self.index>=len(self.counts):raise ValueError('Reader already ended')
        n=self.counts[self.index];ns=len(self.stream)
        if cdf.device.type!='cpu' or cdf.dtype!=torch.int32 or cdf.shape!=(n,9):raise ValueError('Bad CDF shape')
        cdf=cdf.numpy()
        if np.any(cdf[:,0]!=0) or np.any(cdf[:,-1]!=4096) or np.any(np.diff(cdf,axis=1)<0):raise ValueError('Invalid CDF')
        q=np.empty(n,dtype='int16')
        if self.accelerated:
            cdf=np.ascontiguousarray(cdf)
            status=self.reader(self.data.ctypes.data,self.lengths.ctypes.data,self.offsets.ctypes.data,self.state.ctypes.data,self.cursor.ctypes.data,cdf.ctypes.data,n,ns,q.ctypes.data)
            if status:raise ValueError(f'Independent CPU entropy reader failure: {status}')
        for s in range(0 if self.accelerated else ns):
            state=self.state[s];cursor=self.cursor[s];data=self.stream[s]
            for i in range(n*s//ns,n*(s+1)//ns):
                slot=state&4095;label=int(np.searchsorted(cdf[i],slot,side='right'))-1
                if not 0<=label<8:raise ValueError('Bad entropy label')
                lo=int(cdf[i,label]);freq=int(cdf[i,label+1])-lo
                if freq<=0:raise ValueError('Empty frequency')
                q[i]=-32768 if label==7 else label-3
                state=freq*(state>>12)+slot-lo
                while state<self.LOW:
                    if cursor>=len(data):raise ValueError('Entropy stream exhausted')
                    state=(state<<8)|data[cursor];cursor+=1
            self.state[s]=state;self.cursor[s]=cursor
        rare=q==-32768;count=int(rare.sum())
        if self.rare_at+count>len(self.rare):raise ValueError('Rare stream exhausted')
        q[rare]=self.rare[self.rare_at:self.rare_at+count];self.rare_at+=count;self.index+=1
        if self.index==len(self.counts):
            if self.rare_at!=len(self.rare) or any(v!=self.LOW for v in self.state) or any(c!=len(d) for c,d in zip(self.cursor,self.stream)):raise ValueError('Nonterminal rANS state')
        return torch.from_numpy(q)
