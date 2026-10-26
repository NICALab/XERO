"""One learned full-resolution position or empty per4x4 cell.

Joint17-way probabilities serialized as presence then4 conditional binary planes.
All cells of each plane are independent/parallel; no raster pixel recurrence.
"""
import struct
import torch
from torch.nn import functional as F
from .grouped_spatial_codec import GroupedSpatialCodec
from .. import categorical17_rans

HEADER=struct.Struct('<8s5Q');MAGIC=b'FSCELL01'

def cellify(x):
    h,w=x.shape[-2:]
    return F.pixel_unshuffle(F.pad(x,(0,(-w)%4,0,(-h)%4)),4).permute(0,2,3,1).contiguous()

def uncellify(x,shape):
    return F.pixel_shuffle(x.permute(0,3,1,2).contiguous(),4)[...,:shape[-2],:shape[-1]]

def labels_from_mask(mask):
    cells=cellify(mask.float())
    return torch.where(cells.sum(-1)>.5,cells.argmax(-1),16)

def binary_logits(p0,p1):
    small=torch.full_like(p0,-30.)
    return torch.stack([small,small,small,p0.clamp_min(1e-30).log(),p1.clamp_min(1e-30).log(),small,small,small],-1)

def conditional_tables(logits):
    p=logits.float().softmax(-1);active=p[...,:16]
    presence=binary_logits(p[...,16],active.sum(-1))
    tables=[]
    for level in range(4):
        pairs=active.reshape(*active.shape[:-1],1<<level,2,1<<(3-level)).sum(-1)
        tables.append(binary_logits(pairs[...,0],pairs[...,1]))
    return presence,tables

def location_costs(logits,coding='binary'):
    if coding=='categorical':return -categorical17_rans.probabilities(logits).log2()
    if coding!='binary':raise ValueError('unknown location coding')
    presence,tables=conditional_tables(logits)
    pcost=-GroupedSpatialCodec.probabilities(presence)[...,3:5].log2()
    choices=pcost[...,1,None].expand(*pcost.shape[:-1],16)
    for level,table in enumerate(tables):
        costs=-GroupedSpatialCodec.probabilities(table)[...,3:5].log2()
        choices=choices+costs.reshape(*costs.shape[:-2],-1).repeat_interleave(1<<(3-level),-1)
    return torch.cat([choices,pcost[...,:1]],-1)

def location_rate(mask,logits,coding='binary'):
    positions=cellify(mask);weights=torch.cat([positions,1-positions.sum(-1,keepdim=True)],-1)
    return (weights*location_costs(logits,coding)).sum()/mask.numel()

def encode_locations(mask,logits,encode_symbols,backend,streams,coding='binary',validate=True):
    cells=cellify(mask.float())
    if validate and bool((cells.sum(-1)>1).any()):raise ValueError('more than one seed per cell')
    labels=labels_from_mask(mask);present=labels!=16
    if coding=='categorical':return categorical17_rans.encode(labels,logits,backend,streams)
    if coding!='binary':raise ValueError('unknown location coding')
    presence,tables=conditional_tables(logits)
    count=labels.numel();chunks=max(1,(count+8191)//8192);local_streams=min(streams,1<<(chunks-1).bit_length())
    packets=[encode_symbols(present.to(torch.int16),presence,backend,local_streams)]
    chosen=labels[present]
    for level,table in enumerate(tables):
        prefix=chosen>>(4-level)
        known=table[present].gather(-2,prefix[:,None,None].expand(-1,1,8)).squeeze(-2)
        bit=((chosen>>(3-level))&1).to(torch.int16)
        packets.append(encode_symbols(bit,known,backend,local_streams))
    return HEADER.pack(MAGIC,*(len(p) for p in packets))+b''.join(packets)

def decode_locations(packet,logits,pred,decode_symbols,backend):
    if packet[:8] in (categorical17_rans.CPU_MAGIC,categorical17_rans.GPU_MAGIC):
        labels=categorical17_rans.decode(packet,logits).long().reshape(logits.shape[:-1])
        return mask_from_labels(labels,pred)
    if len(packet)<HEADER.size:raise ValueError('short cell location packet')
    magic,*lengths=HEADER.unpack_from(packet)
    if magic!=MAGIC or HEADER.size+sum(lengths)!=len(packet):raise ValueError('bad cell location framing')
    parts=[];at=HEADER.size
    for n in lengths:parts.append(packet[at:at+n]);at+=n
    presence,tables=conditional_tables(logits)
    flags=decode_symbols(parts[0],presence,backend,pred.device).reshape(logits.shape[:-1])
    if not bool(((flags==0)|(flags==1)).all()):raise ValueError('invalid cell presence')
    present=flags.bool();chosen=torch.zeros(int(present.sum()),device=pred.device,dtype=torch.long)
    for level,table in enumerate(tables):
        known=table[present].gather(-2,chosen[:,None,None].expand(-1,1,8)).squeeze(-2)
        bit=decode_symbols(parts[level+1],known,backend,pred.device).long()
        if not bool(((bit==0)|(bit==1)).all()):raise ValueError('invalid position bit')
        chosen=2*chosen+bit
    labels=torch.full_like(flags,16,dtype=torch.long);labels[present]=chosen
    return mask_from_labels(labels,pred)

def mask_from_labels(labels,pred):
    if not bool(((labels>=0)&(labels<=16)).all()):raise ValueError('invalid cell label')
    positions=F.one_hot(labels,17)[...,:16].float()
    valid=cellify(torch.ones_like(pred)).bool()
    if bool((positions.bool()&~valid).any()):raise ValueError('seed outside original FOV')
    return uncellify(positions,pred.shape).bool()
