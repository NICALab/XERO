"""Sparse transmitted seeds -> parallel refinement -> bounded final symbols.

Only the encoder's seed selector sees source pixels. All other network inputs
are reproduced from temporal prediction and actually quantized seed packets.
Hard forward mask/quantization; surrogate gradients are not fidelity proofs.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from ficompress.models.grouped_spatial_codec import bounded_symbols,restore_symbols,fixed_symbol_surrogate,GroupedSpatialCodec,STEP,ESCAPE
from ficompress.models.cell_seed_locations import cellify,uncellify,location_rate


def block(a,b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.SiLU(),nn.Conv2d(b,b,3,padding=1),nn.SiLU())


class SeedRefiner(nn.Module):
    def __init__(self,width,input_channels=3):
        super().__init__();w=width
        self.fine=block(input_channels,w);self.mid=block(w,2*w);self.low=block(2*w,3*w)
        self.upmid=block(5*w,2*w);self.upfine=block(3*w,w);self.out=nn.Conv2d(w,9,1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
        with torch.no_grad():self.out.bias[1:].copy_(torch.tensor([.002,.006,.03,.922,.03,.006,.002,.002]).log())

    def forward(self,x):
        f=self.fine(x);m=self.mid(F.avg_pool2d(f,2));l=self.low(F.avg_pool2d(m,2))
        u=self.upmid(torch.cat([m,F.interpolate(l,size=m.shape[-2:],mode='bilinear',align_corners=False)],1))
        return self.out(self.upfine(torch.cat([f,F.interpolate(u,size=f.shape[-2:],mode='bilinear',align_corners=False)],1))).float()


def symbol_cost_map(source,post,q,logits,temperature=.15,step=STEP):
    """Hard deployed 12-bit categorical cost + bypass, soft bin gradient."""
    costs=-GroupedSpatialCodec.probabilities(logits).log2()
    labels=torch.where((q>=-3)&(q<=3),q.long()+3,7)
    hard=costs.gather(-1,labels[...,None]).squeeze(-1)+(labels==7).float()*16+(q==ESCAPE).float()*32
    valid=(source>0)&(post>0)&(q!=ESCAPE)
    u=(source.float().clamp_min(1e-12).log()-post.float().clamp_min(1e-12).log())/step
    u=torch.where(valid,u,0.)
    k=torch.arange(-3,4,device=post.device,dtype=torch.float32)
    masses=(torch.sigmoid((u[...,None]-k+.5)/temperature)-torch.sigmoid((u[...,None]-k-.5)/temperature)).clamp_min(0)
    smooth=(masses*costs[...,:7].detach()).sum(-1)+(1-masses.sum(-1)).clamp_min(0)*(costs[...,7].detach()+16)
    smooth=torch.where(valid,smooth,0.)
    return hard+(smooth-smooth.detach())


class SeededSpatialCodec(nn.Module):
    def __init__(self,width=16,selector_width=16,grid_stride=16,gate_temperature=.5,conditioning='std',initial_seed_error_percent=4.,initial_mask_probability=.005,selection='pixel',location_coder='binary',seed_tolerance=.01,cell_position_bias=False,seed_gradient='fixed_symbol',lattice_context=False,conditional_final_entropy=False,even_refinement_gain=0.,even_refinement_weights=None,learn_even_refinement=False,conditional_seed_entropy=False,tolerance=.01):
        super().__init__();self.width=width;self.selector_width=selector_width
        if not math.isfinite(tolerance) or not 0<tolerance<=.5:raise ValueError('invalid final tolerance; supported range (0, .5]')
        self.tolerance=float(tolerance);self.final_step=2*math.log1p(tolerance*.9999)
        if not math.isfinite(even_refinement_gain) or not 0<=even_refinement_gain<=2:raise ValueError('invalid even refinement gain')
        self.even_refinement_gain=float(even_refinement_gain)
        if even_refinement_weights is not None:
            if len(even_refinement_weights)!=4 or not all(math.isfinite(v) and abs(v)<=2 for v in even_refinement_weights):raise ValueError('invalid even refinement weights')
            if not even_refinement_gain:raise ValueError('weighted refinement requires a nonzero gain')
        self.even_refinement_weights=None if even_refinement_weights is None else tuple(float(v) for v in even_refinement_weights)
        if learn_even_refinement and (even_refinement_weights is None or not conditional_final_entropy):raise ValueError('trainable filter requires explicit weights and joint conditional entropy first')
        self.learn_even_refinement=bool(learn_even_refinement)
        self.grid_stride=grid_stride;self.gate_temperature=gate_temperature
        if not 0<seed_tolerance<=tolerance:raise ValueError('seed tolerance must be positive and no looser than final bound')
        self.seed_tolerance=seed_tolerance;self.seed_step=2*math.log1p(seed_tolerance*.9999)
        if seed_gradient not in ('fixed_symbol','detached'):raise ValueError('unknown seed gradient surrogate')
        self.seed_gradient=seed_gradient
        if conditioning not in ('std','relative'):raise ValueError('unknown seed conditioning')
        self.conditioning=conditioning;self.initial_seed_error_percent=initial_seed_error_percent;self.initial_mask_probability=initial_mask_probability
        if selection not in ('pixel','cell','lattice'):raise ValueError('unknown selection geometry')
        self.selection=selection
        if lattice_context and selection!='lattice':raise ValueError('candidate geometry context requires lattice selection')
        self.lattice_context=lattice_context
        if cell_position_bias and selection!='cell':raise ValueError('cell offset bias requires cell selection')
        self.cell_position_bias=cell_position_bias
        if location_coder not in ('binary','categorical'):raise ValueError('unknown location coder')
        self.location_coder=location_coder
        if selection=='cell':
            self.location_prior=nn.Sequential(nn.Conv2d(16,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,17,1))
            nn.init.zeros_(self.location_prior[-1].weight);nn.init.zeros_(self.location_prior[-1].bias)
            with torch.no_grad():self.location_prior[-1].bias[:16].fill_(-math.log(16))
        self.selector=nn.Sequential(block(3,selector_width),nn.Conv2d(selector_width,1,1))
        nn.init.zeros_(self.selector[-1].weight);nn.init.zeros_(self.selector[-1].bias)
        # Start from uncommon large innovations; learned CNN can change rankings.
        self.gate_bias=nn.Parameter(torch.tensor(math.log1p(initial_seed_error_percent)))
        self.prior=nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.SiLU(),nn.Conv2d(8,9,1))
        nn.init.zeros_(self.prior[-1].weight);nn.init.zeros_(self.prior[-1].bias)
        with torch.no_grad():
            self.prior[-1].bias[0]=math.log(initial_mask_probability/(1-initial_mask_probability))
            self.prior[-1].bias[1:].copy_(torch.tensor([.002,.006,.03,.922,.03,.006,.002,.002]).log())
        self.refiner=SeedRefiner(width,4 if lattice_context else 3)
        # Appended after existing child modules: legacy optimizer parameter order
        # remains an exact prefix. Zero initialization preserves every proposal.
        if cell_position_bias:
            self.position_offsets=nn.Embedding(1,16)
            nn.init.zeros_(self.position_offsets.weight)
        self.conditional_final_entropy=conditional_final_entropy
        # Append a child module, not a root Parameter: legacy optimizer names
        # and moments remain an exact prefix when this table is introduced.
        if conditional_final_entropy:
            self.final_entropy=nn.Embedding(46,8)
            nn.init.zeros_(self.final_entropy.weight)
        # Last child preserves all old optimizer name/order prefixes.
        if learn_even_refinement:
            self.even_filter=nn.Embedding(1,4)
            with torch.no_grad():self.even_filter.weight[0].copy_(torch.tensor(self.even_refinement_weights))
        self.conditional_seed_entropy=bool(conditional_seed_entropy)
        if conditional_seed_entropy:
            if selection!='lattice':raise ValueError('conditional seed entropy requires lattice candidates')
            # Last child: existing optimizer names/moments are an exact prefix.
            self.seed_entropy=nn.Embedding(46,8)
            nn.init.zeros_(self.seed_entropy.weight)

    @property
    def config(self):return dict(tolerance=self.tolerance,width=self.width,selector_width=self.selector_width,grid_stride=self.grid_stride,gate_temperature=self.gate_temperature,
        conditioning=self.conditioning,initial_seed_error_percent=self.initial_seed_error_percent,initial_mask_probability=self.initial_mask_probability,selection=self.selection,location_coder=self.location_coder,seed_tolerance=self.seed_tolerance,cell_position_bias=self.cell_position_bias,seed_gradient=self.seed_gradient,lattice_context=self.lattice_context,conditional_final_entropy=self.conditional_final_entropy,even_refinement_gain=self.even_refinement_gain,even_refinement_weights=self.even_refinement_weights,learn_even_refinement=self.learn_even_refinement,conditional_seed_entropy=self.conditional_seed_entropy)

    def location_logits(self,pred,mean,std):
        packed=cellify((pred.float()-mean)/std).permute(0,3,1,2).contiguous()
        logits=self.location_prior(packed).float().permute(0,2,3,1).contiguous()
        valid=cellify(torch.ones_like(pred)).bool()
        return torch.cat([logits[...,:16].masked_fill(~valid,-10000.),logits[...,16:]],-1)

    def prior_logits(self,pred,mean,std):
        raw=self.prior((pred.float()-mean)/std).float()
        # Mask uses the existing eight-symbol coder, labels 3=no and 4=yes.
        z=torch.zeros_like(raw[:,:1]);small=torch.full_like(z,-20.)
        mask=torch.cat([small,small,small,z,raw[:,:1],small,small,small],1)
        return mask.permute(0,2,3,1).unsqueeze(1).contiguous(),raw[:,1:].permute(0,2,3,1).unsqueeze(1).contiguous()

    def grid(self,pred):
        y=torch.arange(pred.shape[-2],device=pred.device);x=torch.arange(pred.shape[-1],device=pred.device)
        return ((y[:,None]%self.grid_stride==0)&(x[None,:]%self.grid_stride==0))[None,None].expand_as(pred)

    def select(self,source,pred,seed_q,mean,std,mode='learned'):
        if mode=='grid':
            # Lattice warm-up opens all gates, but q0 always means no observation
            # in both training and byte decoding (no dense-mask warm-up shift).
            return self.grid(pred).float()*((seed_q!=0).float() if self.selection=='lattice' else 1.)
        if mode=='none':return torch.zeros_like(pred)
        if mode!='learned':raise ValueError('mode must be learned/grid/none')
        scale=(self.tolerance*pred.float().abs()).clamp_min(1e-6) if self.conditioning=='relative' else std
        error=(source.float()-pred.float())/scale
        x=torch.cat([(source.float()-mean)/std,(pred.float()-mean)/std,error],1)
        relative=(source.float()-pred.float()).abs()/(self.tolerance*source.float().abs()).clamp_min(1e-6)
        score=self.selector(x).float()+relative.clamp_max(1000).log1p()-self.gate_bias
        if self.selection=='cell':
            packed=cellify(score);valid=cellify(torch.ones_like(pred)).bool()
            if self.cell_position_bias:packed=packed+self.position_offsets.weight[0]
            logits=torch.cat([packed.masked_fill(~valid,-10000.),torch.zeros_like(packed[...,:1])],-1)
            soft=(logits/self.gate_temperature).softmax(-1)
            hard=F.one_hot(logits.argmax(-1),17).float()
            return uncellify((hard+(soft-soft.detach()))[...,:16],pred.shape)
        soft=torch.sigmoid(score/self.gate_temperature)
        hard=(score>0).float()
        # Hard source-dependent pattern IS transmitted and charged in full.
        mask=(hard+(soft-soft.detach()))*(seed_q!=0).float()
        return mask*self.grid(pred).float() if self.selection=='lattice' else mask

    def refine(self,pred,mask,seed_rec,mean,std):
        # Explicit mask prevents arbitrary unknown entries from leaking source.
        innovation=torch.where(mask.detach().bool(),seed_rec.float()-pred.float(),0.)
        # Keep the gate's STE path although forward uses only known seed values.
        innovation=innovation+((mask-mask.detach())*(seed_rec.float()-pred.float()).detach())
        scale=(self.tolerance*pred.float().abs()).clamp_min(1e-6) if self.conditioning=='relative' else std
        inputs=[(pred.float()-mean)/std,mask,innovation/scale]
        if self.lattice_context:inputs.append(self.grid(pred).float())
        out=self.refiner(torch.cat(inputs,1))
        proposal=pred.float()+out[:,:1]*scale
        post=mask*seed_rec.float()+(1-mask)*proposal
        return post,out[:,1:].permute(0,2,3,1).unsqueeze(1).contiguous()

    def refine_from_even(self,post,even_reconstruction,mask):
        from ficompress.even_correction_refinement import refine_from_even
        weights=self.even_filter.weight[0] if self.learn_even_refinement else self.even_refinement_weights
        return refine_from_even(post,even_reconstruction,mask,self.even_refinement_gain,weights)

    def forward(self,source,pred,mean,std,mode='learned',compute_rate=True,rate_border=0):
        if rate_border<0 or (rate_border and (self.selection!='lattice' or rate_border%self.grid_stride or min(source.shape[-2:])<=2*rate_border)):
            raise ValueError('valid-rate border requires lattice alignment and a nonempty core')
        def rate_mean(value):
            return (value[...,rate_border:-rate_border,rate_border:-rate_border] if rate_border else value).mean()
        seed_q,seed_hard=bounded_symbols(source,pred,self.seed_step,self.seed_tolerance)
        # Optional residual-quantizer STE: the decoded seed approximates source,
        # so its gradient to the temporal anchor is zero. Forward bytes/bounds
        # are unchanged. The seed symbol rate still differentiates w.r.t.pred.
        seed_rec=seed_hard.detach() if self.seed_gradient=='detached' else fixed_symbol_surrogate(pred,source,seed_q,seed_hard,self.seed_step)
        mask=self.select(source,pred,seed_q,mean,std,mode)
        post,final_logits=self.refine(pred,mask,seed_rec,mean,std)
        final_q,hard=bounded_symbols(source,post,self.final_step,self.tolerance)
        rec=fixed_symbol_surrogate(post,source,final_q,hard,self.final_step)
        if self.even_refinement_gain:
            post=self.refine_from_even(post,rec,mask.detach().bool())
            final_q,hard=bounded_symbols(source,post,self.final_step,self.tolerance)
            rec=fixed_symbol_surrogate(post,source,final_q,hard,self.final_step)
        result=dict(rec=rec,post=post,mask=mask,seed_q=seed_q,seed_rec=seed_rec,final_q=final_q,final_logits=final_logits)
        if compute_rate:
            mask_logits,seed_logits=self.prior_logits(pred,mean,std)
            costs=-GroupedSpatialCodec.probabilities(mask_logits).log2()
            if self.selection=='lattice':mask_rate=pred.new_zeros(())
            elif mode=='learned' and self.selection=='cell':mask_rate=location_rate(mask,self.location_logits(pred,mean,std),self.location_coder)
            else:mask_rate=(mask*costs[...,4]+(1-mask)*costs[...,3]).mean() if mode=='learned' else pred.new_zeros(())
            if self.conditional_seed_entropy:
                from ficompress.checkerboard_entropy import checkerboard_context,labels
                stride=self.grid_stride
                low_mask=mask[...,::stride,::stride]
                low_q=seed_q[...,::stride,::stride]
                transmitted=torch.where(low_mask.detach().bool(),low_q,0)
                # Only decoded even candidate symbols condition the odd phase.
                # Context decisions are detached; gate and symbol surrogates
                # retain the same paths as the independent-seed model.
                context=checkerboard_context(labels(transmitted))
                low_logits=seed_logits[...,::stride,::stride,:]+self.seed_entropy(context)
                selected=symbol_cost_map(source[...,::stride,::stride],pred[...,::stride,::stride],low_q,low_logits,step=self.seed_step)
                absent=-GroupedSpatialCodec.probabilities(low_logits)[...,3].log2()
                seed_cost=low_mask*selected+(1-low_mask)*absent
                if rate_border:
                    border=rate_border//stride
                    seed_cost=seed_cost[...,border:-border,border:-border]
                core_voxels=source.shape[0]*source.shape[1]*(source.shape[-2]-2*rate_border)*(source.shape[-1]-2*rate_border)
                seed_rate=seed_cost.sum()/core_voxels
            else:
                seed_rate=rate_mean(mask*symbol_cost_map(source,pred,seed_q,seed_logits,step=self.seed_step))
            if self.selection=='lattice' and not self.conditional_seed_entropy:
                # Unified absence/value alphabet: q0 is paid at disabled candidate
                # sites, without a second mask/presence stream or source leak.
                zero_cost=-GroupedSpatialCodec.probabilities(seed_logits)[...,3].log2()
                seed_rate=seed_rate+rate_mean((self.grid(pred).float()-mask)*zero_cost)
            rate_logits=final_logits
            if self.conditional_final_entropy:
                from ficompress.checkerboard_entropy import checkerboard_context,labels
                # Odd sites see only decoded even symbols; even sites use a
                # constant context. Integer context decisions are detached, not
                # a differentiable promise about neighboring quantizer bins.
                symbols=torch.where(mask.detach().bool(),seed_q,final_q)
                context=checkerboard_context(labels(symbols))
                rate_logits=final_logits+self.final_entropy(context)
            final_rate=rate_mean((1-mask)*symbol_cost_map(source,post,final_q,rate_logits,step=self.final_step))
            result.update(loss=mask_rate+seed_rate+final_rate,mask_rate=mask_rate,seed_rate=seed_rate,final_rate=final_rate)
        return result

    cdf=staticmethod(GroupedSpatialCodec.cdf)
