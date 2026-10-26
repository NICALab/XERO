import json, hashlib
import numpy as np
import torch
from torch import nn
from integer_ptq_screen import build_causal,convert,IntConv,CanonicalActivation
from canonical_entropy import exp_table
from ficompress.models.causal_pixel_predictor import CausalPixelPredictor
from seeded_spatial_codec import SeededSpatialCodec
PROFILE='fiarc_integer_experimental_v1'
BOUND_PROFILE='fiarc_integer_boundscale_v2'
SUPPORTED_PROFILES=(PROFILE,BOUND_PROFILE)

def serialized_profile(path):
    with np.load(path,allow_pickle=False) as f:
        return json.loads(f['metadata'].tobytes())['profile']


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def write_model(state,path,bits=13,wb=8):
    """Serialize decoder-only arrays from an explicitly supplied checkpoint."""
    if path.exists():raise FileExistsError(path)
    base,codec=build_causal(state,'cpu')
    for net in [base,codec.prior,codec.refiner]:convert(net,bits,wb)
    arrays={};nodes={}
    for prefix,net in [('base',base),('prior',codec.prior),('refiner',codec.refiner)]:
        for name,module in net.named_modules():
            key=prefix+'.'+name
            if isinstance(module,IntConv):
                arrays[key+'.q']=module.qw.numpy().astype('<i2')
                arrays[key+'.exponent']=torch.log2(module.scale).numpy().astype('<i2')
                arrays[key+'.bias']=module.bias.numpy().astype('<f4')
                nodes[key]=dict(stride=module.stride,padding=module.padding,groups=module.groups)
            elif isinstance(module,CanonicalActivation) and 'silu_table' not in arrays:
                arrays['silu_table']=module.table.numpy().astype('<i4')
    for name in ['seed_entropy','final_entropy','even_filter']:
        if hasattr(codec,name):arrays[name]=getattr(codec,name).weight.detach().numpy().astype('<f4')
    arrays['exp_table']=exp_table('cpu').numpy().astype('<i4')
    # A new profile prevents old clients from silently using fixed-1% math on
    # new non-1% streams. Preserve the exact old representation at 1%.
    profile=PROFILE if codec.tolerance==.01 else BOUND_PROFILE
    meta=dict(profile=profile,bits=bits,weight_bits=wb,base={k:v for k,v in state['base'].items() if k!='model'},codec=state['seed_config'],nodes=nodes)
    arrays['metadata']=np.frombuffer(json.dumps(meta,sort_keys=True).encode(),dtype='u1')
    np.savez_compressed(path,**arrays)


@torch.no_grad()
def load_model(path,device='cpu'):
    with np.load(path,allow_pickle=False) as f:
        arrays={name:f[name].copy() for name in f.files}
    meta=json.loads(arrays.pop('metadata').tobytes())
    if meta['profile'] not in SUPPORTED_PROFILES or meta['bits'] not in (8,10,12,13):raise ValueError('Unsupported integer profile')
    spec=meta['base'];base=CausalPixelPredictor(spec['history'],spec['width'],input_mode=spec['input_mode']).to(device).eval()
    codec=SeededSpatialCodec(**meta['codec']).to(device).eval()
    codec._conditioning_tolerance=.01 if meta['profile']==PROFILE else codec.tolerance
    def fill(net,prefix):
        for name,child in list(net.named_children()):
            key=prefix+'.'+name
            if isinstance(child,nn.Conv2d):
                node=meta['nodes'][key];q=arrays[key+'.q'];ex=arrays[key+'.exponent'];bias=arrays[key+'.bias']
                if q.shape!=tuple(child.weight.shape) or ex.shape!=(child.out_channels,) or bias.shape!=(child.out_channels,):raise ValueError('Bad model shape')
                if np.max(abs(q.astype('i4')))>((1<<(meta['weight_bits']-1))-1) or np.max(abs(ex))>100 or not np.isfinite(bias).all():raise ValueError('Invalid model numbers')
                m=IntConv.__new__(IntConv);nn.Module.__init__(m);m.limit=(1<<(meta['bits']-1))-1
                m.register_buffer('qw',torch.from_numpy(q.astype('float64')).to(device))
                m.register_buffer('scale',torch.from_numpy(np.exp2(ex.astype('float64'))).to(device))
                m.register_buffer('bias',torch.from_numpy(bias.astype('float64')).to(device))
                m.stride=tuple(node['stride']);m.padding=tuple(node['padding']);m.groups=node['groups'];m.cpu_integer_parts=False
                m.acc_bound=int((m.qw.abs().sum((1,2,3))*m.limit).max())
                if m.acc_bound>=2**31:raise ValueError('Accumulator bound exceeded')
                if device=='cpu':m.enable_cpu_integer_parts()
                setattr(net,name,m)
            elif isinstance(child,(nn.SiLU,nn.LeakyReLU)):
                m=CanonicalActivation('silu' if isinstance(child,nn.SiLU) else 'leaky',device)
                if arrays['silu_table'].shape!=(131073,):raise ValueError('Bad activation table')
                m.table=torch.from_numpy(arrays['silu_table']).to(device);setattr(net,name,m)
            else:fill(child,key)
    fill(base,'base');fill(codec.prior,'prior');fill(codec.refiner,'refiner')
    for name in ['seed_entropy','final_entropy','even_filter']:
        if name in arrays:getattr(codec,name).weight.copy_(torch.from_numpy(arrays[name]).to(device))
    codec.seed_entropy_offsets=codec.seed_entropy.weight.detach();codec.final_entropy_offsets=codec.final_entropy.weight.detach()
    if not np.array_equal(arrays['exp_table'],exp_table('cpu').numpy()):raise ValueError('CDF table profile mismatch')
    return base,codec,spec
