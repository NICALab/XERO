import os, sys, time
import numpy as np
import torch
import triton
import triton.language as tl
from historical_matched import IndependentReader
from integer_codec_speed import IntegerRuntime,RECORD
import sequence_codec
import ficompress.seed_sequence_rans as rans

@triton.jit
def shift_history(H,R,P:tl.constexpr,C:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);valid=i<P
    for c in range(C-1):
        v=tl.load(H+(c+1)*P+i,valid,0);tl.store(H+c*P+i,v,valid)
    tl.store(H+(C-1)*P+i,tl.load(R+i,valid,0),valid)


def advance_in_place(hist,rec,first):
    if first:hist.copy_(rec.expand_as(hist))
    else:shift_history[(triton.cdiv(rec.numel(),256),)](hist,rec,rec.numel(),hist.shape[1],256)


def memory_stream(self,n):
    hist=torch.full((1,self.base.history,*self.shape),self.mean,device='cuda')
    for i in range(n):
        pred=hist[:,-1:] if i==0 else self.predict_graph(hist)
        rec=yield i,pred
        if i+1<n:advance_in_place(hist,rec,i==0)


class MemoryReader(IndependentReader):
    def decoded_frames(self,packets,n):
        hist=torch.full((1,self.base.history,*self.shape),self.mean,device='cuda');k=0
        for packet in packets:
            assert RECORD.unpack_from(packet)[0]+RECORD.size==len(packet)
            group=self.GroupDecoder(packet[RECORD.size:],'cuda')
            assert (group.h,group.w)==self.shape
            for _ in range(group.frames):
                pred=hist[:,-1:] if k==0 else self.forecast(hist)
                rec=group.decode_frame(self.codec,pred,self.mean,self.std)
                yield k,rec
                if k+1<n:advance_in_place(hist,rec,k==0)
                k+=1
        assert k==n


@triton.jit
def _uneven_pack_frame(Q,C,OQ,OC,COUNT:tl.constexpr,STREAMS:tl.constexpr,
                       FRAMES:tl.constexpr,FRAME:tl.constexpr,WIDTH:tl.constexpr,
                       B:tl.constexpr):
    stream=tl.program_id(1)
    pos=tl.program_id(0)*B+tl.arange(0,B)
    begin=stream*COUNT//STREAMS
    length=(stream+1)*COUNT//STREAMS-begin
    valid=pos<length
    src=begin+pos
    dst=stream*WIDTH+FRAME*length+pos
    tl.store(OQ+dst,tl.load(Q+src,valid,0),valid)
    # Keep the released chronological order within each stream. Convert the
    # compact int16 CDF to int32 on store, without a frame-major temporary.
    for col in range(9):
        tl.store(OC+dst*9+col,tl.load(C+src*9+col,valid,0),valid)
    if FRAME==FRAMES-1:
        tail=pos<WIDTH-FRAMES*length
        dst=stream*WIDTH+FRAMES*length+pos
        tl.store(OQ+dst,0,tail)
        for col in range(9):
            tl.store(OC+dst*9+col,0 if col<4 else 4096,tail)


def memory_order(reference):
    def order(items,streams):
        count=items[0][0].numel();frames=len(items)
        if count<1 or any(q.numel()!=count for q,_ in items):return reference(items,streams)
        if count%streams:
            width=frames*((count+streams-1)//streams)
            q=torch.empty(streams*width,device=items[0][0].device,dtype=items[0][0].dtype)
            c=torch.empty((streams*width,9),device=items[0][1].device,dtype=torch.int32)
            # The longest terminal padding is FRAMES slots, even for streams
            # with no symbols. No dense gather indices, stacks or where copies.
            blocks=triton.cdiv(max((count+streams-1)//streams,frames),256)
            for i,(qi,ci) in enumerate(items):
                _uneven_pack_frame[(blocks,streams)](qi,ci,q,c,count,streams,frames,i,width,256)
            return q,c
        n=count//streams
        # Write directly into stream-major storage, avoiding frame-major stack
        # plus its full-size transpose copy. Bit order and packet format unchanged.
        q=torch.empty((streams,frames,n),device=items[0][0].device,dtype=items[0][0].dtype)
        c=torch.empty((streams,frames,n,9),device=items[0][1].device,dtype=torch.int32)
        for i,(qi,ci) in enumerate(items):q[:,i].copy_(qi.reshape(streams,n));c[:,i].copy_(ci.reshape(streams,n,9))
        return q.reshape(-1),c.reshape(-1,9)
    return order


def compact_pending_cdfs():
    # Canonical integer CDFs are0..4096. Keep the64-frame pending cache in
    # signed16 bits, then restore int32 per entropy phase at serialization.
    # This is lossless storage, not another probability quantization.
    module=sys.modules[IntegerRuntime.__mro__[1].__module__]
    prepare=module.prepare_frame
    def compact(*args,**kwargs):
        item,rec=prepare(*args,**kwargs)
        item['phases']=[(q,c.to(torch.int16)) for q,c in item['phases']]
        return item,rec
    module.prepare_frame=compact;sequence_codec.encode_sequence=memory_encode_sequence


@torch.no_grad()
def memory_encode_sequence(items,streams=64):
    """Same SRANSG01 bytes; expand16-bit CDFs directly into packed int32 storage."""
    assert items and len(items)<=256 and 0<streams<=65536
    clean=[];device=items[0][0].device
    for q,c in items:
        q=q.reshape(-1).to(torch.int16).contiguous()
        assert q.is_cuda and c.device==device and c.dtype in (torch.int16,torch.int32) and c.shape==(q.numel(),9)
        clean.append((q,c.contiguous()))
    counts=[q.numel() for q,_ in clean];n=sum(counts);streams=min(streams,max(1,(max(counts)+255)//256))
    q,cdf=rans.stream_order(clean,streams);cdf=cdf.to(torch.int32)
    slots,lengths,status,bypass_counts=rans.extension().encode_reciprocal(q,cdf,streams,rans.division_table(device))
    metadata=torch.stack([lengths,status,bypass_counts]).cpu().numpy()
    del q,cdf
    if metadata[1].any():raise RuntimeError('Sequence encoder failed')
    total=int(metadata[0].sum(dtype=np.int64));nb=int(metadata[2].sum(dtype=np.int64))
    payload=rans.extension().pack(slots,lengths,lengths.long().cumsum(0)-lengths,total).cpu().numpy().tobytes()
    del slots
    if nb:
        chronological=torch.cat([q for q,_ in clean]);bypass=chronological[(chronological<-3)|(chronological>3)].cpu().numpy().astype('<i2',copy=False)
        assert len(bypass)==nb
        side=rans.zstd.ZstdCompressor(level=3).compress(bypass.tobytes())
    else:side=b''
    assert os.environ.get('FIARC_SEQUENCE_GOLOMB','0')!='1'
    return rans.HEADER.pack(rans.MAGIC,len(items),streams,n,total,len(side),nb)+np.asarray(counts,dtype='<u4').tobytes()+metadata[0].astype('<u4',copy=False).tobytes()+payload+side


class SingleGroupSource:
    """One64-frame GPU buffer, refilled after synchronous encode_group.

    All operations use the current stream. The previous group's deferred float
    escapes have already been gathered into bytes before its storage is reused.
    Avoids keeping a second1GB input block on2048-square shared GPUs.
    """
    def __init__(self,host):
        self.host=host;self.buffer=torch.empty((64,*host.shape[1:]),device='cuda',dtype=torch.float32);self.reset()
    def __len__(self):return len(self.host)
    def reset(self):self.next=0
    def __getitem__(self,i):
        assert i==self.next;self.next+=1
        if i%64==0:
            n=min(64,len(self.host)-i);self.buffer[:n].copy_(self.host[i:i+n],non_blocking=True)
        return self.buffer[i%64]


@torch.inference_mode()
def encode_segments(rt,source,k):
    source.reset();torch.cuda.synchronize();tick=time.perf_counter();segments=[]
    for start in range(0,len(source),k):
        count=min(k,len(source)-start)
        class View:
            def __len__(self):return count
            def __getitem__(self,i):return source[start+i]
        packets,_=rt.encode(View(),count,256)
        segments.append((start,count,packets))
    # Include actual byte materialization; indexes separately accounted below.
    blob=b''.join(p for _,_,ps in segments for p in ps)
    torch.cuda.synchronize();return segments,blob,time.perf_counter()-tick


@torch.inference_mode()
def decode_segments(reader,segments,output):
    output.reset();torch.cuda.synchronize();tick=time.perf_counter()
    for start,count,packets in segments:
        for i,y in reader.decoded_frames(packets,count):output.put(start+i,y)
    output.finish();torch.cuda.synchronize();return time.perf_counter()-tick
