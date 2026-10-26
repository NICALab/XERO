import torch
from integer_ptq_screen import convert,IntConv,canonical_preprocessing
from conv_integer import replace_with_fast
from sustained_speed import Runtime,RECORD
from memory_safety import Graph

class IntegerRuntime(Runtime):
    def __init__(self,state,shape):
        super().__init__(state,shape,False)
        for n in [self.base,self.codec.prior,self.codec.refiner]:
            convert(n,13,8);replace_with_fast(n,13,IntConv)
        self.codec.select=torch.compile(self.codec.select)
        norm=canonical_preprocessing(self.codec,self.mean,self.std)
        hist=torch.full((1,self.base.history,*shape),self.mean,device='cuda')
        pred=hist[:,-1:].clone();mask=torch.zeros_like(pred);seed=pred.clone()
        self.predict_graph=Graph(lambda h:self.base(norm(h))*self.std+self.mean,[hist])
        prior=self.codec.prior_logits;refine=self.codec.refine;even=self.codec.refine_from_even
        prior_graph=Graph(lambda p:prior(p,self.mean,self.std),[pred])
        refine_graph=Graph(lambda p,m,s:refine(p,m,s,self.mean,self.std),[pred,mask,seed])
        even_graph=Graph(even,[pred,seed,mask.bool()])
        self._memory_graphs=[self.predict_graph,prior_graph,refine_graph,even_graph]
        self.codec.prior_logits=lambda p,m,s:prior_graph(p)
        self.codec.refine=lambda p,m,s,mean,std:refine_graph(p,m,s)
        self.codec.refine_from_even=lambda p,r,m:even_graph(p,r,m)
    def stream(self,n):
        def frames():
            hist=torch.full((1,self.base.history,*self.shape),self.mean,device='cuda')
            for i in range(n):
                if i%128==0:hist.fill_(self.mean)
                p=hist[:,-1:] if i%128==0 else self.predict_graph(hist)
                rec=yield i,p
                hist=rec.repeat(1,self.base.history,1,1) if i%128==0 else torch.cat((hist[:,1:],rec),1)
        return frames()
