import importlib.util, struct
import torch
from integer_archive import load_model
from integer_gpu_reader import prepare
from canonical_entropy import fast_canonical_cdf
import sequence_codec
RECORD=struct.Struct("<Q")

class IndependentReader:
    @torch.inference_mode()
    def __init__(self,model,shape):
        self.shape=shape;self.restart=128
        self.base,self.codec,spec=load_model(model,'cuda');self.mean,self.std=spec['mean'],spec['std']
        self.forecast=prepare(self.base,self.codec,spec,shape,13)
        spec=importlib.util.spec_from_file_location('_historical_integer_packet_reader',sequence_codec.__file__)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.cdf_tensor=fast_canonical_cdf
        self.GroupDecoder=module.GroupDecoder
    def decoded_frames(self,packets,n):
        k=0;hist=None
        for packet in packets:
            if RECORD.unpack_from(packet)[0]+RECORD.size!=len(packet):raise ValueError('Record length')
            reader=self.GroupDecoder(packet[RECORD.size:],'cuda')
            if (reader.h,reader.w)!=self.shape:raise ValueError('Geometry')
            for _ in range(reader.frames):
                first=k==0 or (self.restart>0 and k%self.restart==0)
                if first:hist=torch.full((1,self.base.history,*self.shape),self.mean,device='cuda')
                pred=hist[:,-1:] if first else self.forecast(hist)
                rec=reader.decode_frame(self.codec,pred,self.mean,self.std)
                hist=rec.repeat(1,self.base.history,1,1) if first else torch.cat((hist[:,1:],rec),1)
                yield k,rec;k+=1
        if k!=n:raise ValueError('Incomplete independent decode')
