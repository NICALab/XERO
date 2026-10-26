import torch
from integer_ptq_screen import IntConv,canonical_preprocessing
from conv_integer import replace_with_fast
from memory_safety import Graph

@torch.inference_mode()
def prepare(base,codec,spec,shape,bits=13):
    for net in [base,codec.prior,codec.refiner]:replace_with_fast(net,bits,IntConv)
    mean,std=spec['mean'],spec['std'];norm=canonical_preprocessing(codec,mean,std)
    hist=torch.full((1,base.history,*shape),mean,device='cuda');pred=hist[:,-1:].clone();mask=torch.zeros_like(pred);seed=pred.clone()
    forecast=Graph(lambda h:base(norm(h))*std+mean,[hist])
    prior=codec.prior_logits;refine=codec.refine;even=codec.refine_from_even
    prior_graph=Graph(lambda p:prior(p,mean,std),[pred])
    refine_graph=Graph(lambda p,m,s:refine(p,m,s,mean,std),[pred,mask,seed])
    even_graph=Graph(even,[pred,seed,mask.bool()])
    codec._memory_graphs=[forecast,prior_graph,refine_graph,even_graph]
    codec.prior_logits=lambda p,m,s:prior_graph(p)
    codec.refine=lambda p,m,s,mu,sd:refine_graph(p,m,s)
    codec.refine_from_even=lambda p,r,m:even_graph(p,r,m)
    return forecast
