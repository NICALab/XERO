import time
import torch
from speed_probe import Sink, RECORD
from model_state import model_profile,build_causal
from sequence_codec import prepare_frame,encode_group,GroupDecoder
from ficompress.models.chunk_forecast import forecast_chunk
from history_policy import advance

class Runtime:
    """Decoder core: accepts serialized weights, geometry and bytes only."""
    def __init__(self, state, shape, compile_model,context_input_profile='canonical_v1'):
        self.state, self.shape = state, tuple(shape)
        self.profile=model_profile(state)
        self.contextual=self.profile=='contextual_future'
        if context_input_profile not in ('canonical_v1','legacy_v0'):raise ValueError('Unknown context input implementation')
        self.context_input_profile=context_input_profile if self.contextual else None
        self.future = self.profile in ('direct_future','hierarchical_future')
        self.gap=state['context_future']['gap'] if self.contextual else state['bidirectional']['gap'] if self.future else 1
        if self.contextual:
            from contextual_future import build as build_contextual
            self.base,self.codec=build_contextual(state,'cuda');self.interp=None
        elif self.future:
            self.base, self.codec, self.interp = build(state, 'cuda')
        else:
            self.base, self.codec = build_causal(state, 'cuda'); self.interp = None
        self.mean, self.std = state['base']['mean'], state['base']['std']
        if self.contextual and context_input_profile=='canonical_v1':
            from canonical_context_runtime import CanonicalContextRuntime
            self.base=CanonicalContextRuntime(self.base,compile_model)
        elif compile_model:self.base=torch.compile(self.base)
        if compile_model:
            if self.future: self.interp = torch.compile(self.interp)
            for name in ('select','prior_logits','refine','refine_from_even'):
                setattr(self.codec, name, torch.compile(getattr(self.codec,name)))

    def stream(self, n):
        if self.contextual:
            from contextual_future import predictions as contextual_predictions
            return contextual_predictions(self.base,self.mean,self.std,n,self.shape,self.gap,'cuda')
        if self.future:
            spec = self.state['bidirectional']
            return predictions(self.base,self.interp,self.mean,self.std,n,self.shape,
                spec['gap'],'cuda',spec.get('topology','direct'))
        def causal():
            hist = torch.full((1,self.base.history,*self.shape),self.mean,device='cuda')
            for i in range(n):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    pred = forecast_chunk(self.base,hist,self.mean,self.std,1,i==0)
                rec = yield i,pred
                hist = advance(hist,rec,i==0,'repeat_first_decoded')
        return causal()

    def precision(self,index,n):
        if self.future: frame_precision(self.codec,self.state,index,n)

    def decoded_frames(self,packets,n):
        """No original/source/encoder tensors or provenance may enter this API."""
        gen = self.stream(n); pair = next(gen); k = 0
        for packet in packets:
            if len(packet)<RECORD.size or RECORD.unpack_from(packet)[0]+RECORD.size!=len(packet):
                raise ValueError('Corrupt packet length')
            reader = GroupDecoder(packet[RECORD.size:],'cuda')
            if (reader.h,reader.w)!=self.shape or not 1<=reader.frames<=64:
                raise ValueError('Corrupt packet geometry')
            for _ in range(reader.frames):
                if k>=n: raise ValueError('Extra frames')
                index,pred = pair; self.precision(index,n)
                rec = reader.decode_frame(self.codec,pred,self.mean,self.std)
                yield index,rec
                k += 1
                try: pair = gen.send(rec)
                except StopIteration:
                    if k!=n: raise ValueError('Early end of reference order')
        if k!=n: raise ValueError('Truncated archive')

    def encode(self,source,n,streams,observer=None):
        # Source and history scale with resident size, NOT with logical n.
        pending=[]; sink=Sink()
        torch.cuda.synchronize(); start=time.perf_counter()
        try:
            gen=self.stream(n); pair=next(gen)
            for k in range(n):
                index,pred=pair; self.precision(index,n)
                item,rec=prepare_frame(self.codec,source[index%len(source)][None,None],pred,self.mean,self.std)
                if observer is not None:observer(index,rec)
                pending.append(item)
                if len(pending)==64 or k==n-1:
                    packet=encode_group(pending,streams)
                    sink.put(RECORD.pack(len(packet))+packet); pending=[]
                try: pair=gen.send(rec)
                except StopIteration:
                    if k!=n-1: raise RuntimeError('Early end of encoder')
            torch.cuda.synchronize()
        finally:
            sink.finish()
        end=time.perf_counter()
        return sink.packets,dict(seconds=end-start,fps=n/(end-start),
            first_packet_seconds=sink.completed[0]-start,
            packet_completion_seconds=[v-start for v in sink.completed])

    def timed_decode(self,packets,n):
        # Simulates a bounded GPU consumer in display order, not full-movie retention.
        output=torch.empty((max(16,self.gap+1),*self.shape),device='cuda',dtype=torch.float32)
        torch.cuda.synchronize(); start=time.perf_counter()
        for index,rec in self.decoded_frames(packets,n): output[index%len(output)].copy_(rec[0,0])
        torch.cuda.synchronize(); seconds=time.perf_counter()-start
        return dict(seconds=seconds,fps=n/seconds,output='bounded GPU ring; no D2H movie transfer')
