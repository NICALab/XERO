import gc, hashlib, json, math, time, types
from pathlib import Path
import numpy as np
import torch
from integer_codec_speed import IntegerRuntime,RECORD
from integer_archive import write_model,serialized_profile,PROFILE,BOUND_PROFILE
from canonical_entropy import fast_canonical_cdf
from large_fov_restart import MemoryReader,memory_stream,SingleGroupSource,encode_segments,decode_segments,compact_pending_cdfs,memory_order
from fiarc_stream_order import adapter
import ficompress.seed_sequence_rans as rans
from ram_runtime import PinnedSource,PinnedOutput
from model_state import export_state
import sequence_codec, memory_safety
_configured=False

def interval(h,w):
    k=1
    while k*h*w*2<=100_000_000:k*=2
    return max(128,k)


class Codec:
    def __init__(self, checkpoint, folder):
        global _configured
        torch.set_num_threads(2);torch._inductor.config.compile_threads=2
        if not _configured:
            sequence_codec.cdf_tensor=fast_canonical_cdf
            # Direct stream-major INT32 output avoids stack/transpose plus a
            # second full conversion buffer on large FOVs. Packet bytes unchanged.
            rans.stream_order=memory_order(adapter(rans.stream_order))
            compact_pending_cdfs();_configured=True
        self.state=export_state(torch.load(checkpoint,map_location='cpu',weights_only=False))
        self.epsilon=float(self.state['tolerance'])
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.model=self.folder/'model.npz'
        if not self.model.exists():write_model(self.state,self.model)
        expected=PROFILE if self.epsilon==.01 else BOUND_PROFILE
        if serialized_profile(self.model)!=expected:
            raise ValueError('Legacy model in output directory: use a new output directory, never reinterpret existing packets')
        self.rt=self.reader=None;self.shape=None

    def clear(self):
        memory_safety.close_runtime(self.rt);memory_safety.close_runtime(self.reader)
        self.rt=self.reader=None;memory_safety.release_caches()

    def __enter__(self):return self

    def __exit__(self,*exc):self.clear()

    def release_encoder(self):
        memory_safety.close_runtime(self.rt);self.rt=None;memory_safety.release_caches()

    def release_reader(self):
        memory_safety.close_runtime(self.reader);self.reader=None;memory_safety.release_caches()

    def configure_shape(self,shape):
        if self.shape!=shape:
            self.clear();self.shape=shape

    def encoder(self,shape):
        self.configure_shape(shape)
        if self.rt is None:
            self.rt=IntegerRuntime(self.state,shape)
            self.rt.stream=types.MethodType(memory_stream,self.rt)
        return self.rt

    def receiver(self,shape):
        self.configure_shape(shape)
        if self.reader is None:self.reader=MemoryReader(self.model,shape)
        return self.reader

    @torch.inference_mode()
    @memory_safety.release_on_failure
    def process(self,x,warm=True,repeats=1,seek_check=False,restart_override=None):
        x=np.asarray(x,dtype='f4',order='C')
        if x.ndim!=3 or not np.isfinite(x).all():raise ValueError('Invalid target')
        n,h,w=x.shape;shape=(h,w);k=interval(h,w) if restart_override is None else restart_override
        if type(k) is not int or k<128 or k&(k-1):raise ValueError('Invalid restart interval')
        host=torch.empty(x.shape,dtype=torch.float32,pin_memory=True)
        host.copy_(torch.from_numpy(x))
        source=(SingleGroupSource if h*w>=1_000_000 else PinnedSource)(host)
        rt=self.encoder(shape)
        if warm:
            # Warm the exact tail shape as well as the 64-frame entropy packets.
            encode_segments(rt,source,k)
        enc=[];signatures=[]
        for _ in range(repeats):
            segments,blob,seconds=encode_segments(rt,source,k)
            enc.append(seconds);signatures.append(hashlib.sha256(blob).hexdigest())
        if len(set(signatures))!=1:raise AssertionError('Nondeterministic encode')
        del source
        # 2M+ pixel FOVs do not keep encoder + independent decoder together.
        if h*w>=1_000_000:del rt;self.release_encoder()
        reader=self.receiver(shape);output=PinnedOutput(n,shape)
        if warm:decode_segments(reader,segments,output)
        dec=[decode_segments(reader,segments,output) for _ in range(repeats)]
        y=output.host.numpy().copy()
        seek_exact=None
        if seek_check and len(segments)>1:
            start,count,packets=segments[1];single=PinnedOutput(count,shape)
            decode_segments(reader,[(0,count,packets)],single)
            seek_exact=bool(np.array_equal(single.host.numpy(),y[start:start+count]))
            if not seek_exact:raise AssertionError('Independent restart seek mismatch')
            del single
        if h*w>=1_000_000:del reader;self.release_reader()
        sse=energy=0.;bad=0;prefix_bytes=steady_bytes=prefix_vox=steady_vox=0
        for i in range(n):
            a=x[i].astype('f8');e=y[i].astype('f8')-a
            bad+=int(np.count_nonzero(~np.isfinite(y[i])|(np.abs(e)>self.epsilon*np.abs(a))))
            sse+=float(np.square(e).sum());energy+=float(np.square(a).sum())
        if bad:raise AssertionError(f'Original-domain bound violations: {bad}')
        index=[];offset=0
        for start,count,packets in segments:
            index.append(dict(start=start,frames=count,offset=offset,bytes=sum(map(len,packets))))
            for j,packet in enumerate(packets):
                nv=sequence_codec.HEADER.unpack_from(packet,RECORD.size)[1]*h*w
                if j==0:prefix_bytes+=len(packet);prefix_vox+=nv
                else:steady_bytes+=len(packet);steady_vox+=nv
                offset+=len(packet)
        index_bytes=len(json.dumps(index,separators=(',',':')).encode())
        result=dict(shape=list(x.shape),frames=n,voxels=x.size,restart=k,
            payload_bytes=len(blob),index_bytes=index_bytes,prefix_payload_bytes=prefix_bytes,
            prefix_voxels=prefix_vox,steady_payload_bytes=steady_bytes,steady_voxels=steady_vox,
            sse=sse,energy=energy,violations=bad,encode_seconds=float(np.median(enc)),
            decode_seconds=float(np.median(dec)),source_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
            reconstruction_sha256=hashlib.sha256(y.tobytes()).hexdigest(),packet_sha256=signatures[0],
            independent_cuda_decoder=True,seek_exact=seek_exact,index=index)
        del host,output,segments,blob
        return y,result
