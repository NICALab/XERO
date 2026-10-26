"""Self-describing compact rANS framing; entropy states/payload unchanged."""
import struct
import numpy as np

LEGACY=struct.Struct('<8sQQQQQ');MAGIC=b'SRAN1201'
COMPACT=struct.Struct('<8sIHIII');COMPACT_MAGIC=b'SRAN1202'


def pack_packet(n,streams,total,nb,lengths,payload,side,compact=False):
    lengths=np.asarray(lengths)
    if len(lengths)!=streams or (lengths<4).any() or len(payload)!=total or int(lengths.sum(dtype=np.int64))!=total:raise ValueError('inconsistent packet layout')
    if compact and 0<streams<65536 and max(n,total,len(side),nb)<=0xffffffff and lengths.max(initial=0)<=65535:
        side=side if nb else b''
        return COMPACT.pack(COMPACT_MAGIC,n,streams,total,len(side),nb)+lengths.astype('<u2',copy=False).tobytes()+payload+side
    return LEGACY.pack(MAGIC,n,streams,total,len(side),nb)+lengths.astype('<u4',copy=False).tobytes()+payload+side


def packet_layout(packet):
    if packet.startswith(COMPACT_MAGIC):
        header,itemsize,dtype=COMPACT,2,'<u2'
    elif packet.startswith(MAGIC):header,itemsize,dtype=LEGACY,4,'<u4'
    else:raise ValueError('unknown rANS framing')
    if len(packet)<header.size:raise ValueError('short rANS header')
    _,n,streams,total,side,bypass=header.unpack_from(packet)
    if not 0<streams<=65536 or bypass>n:raise ValueError('invalid rANS counts')
    if header is COMPACT and ((not side)!= (not bypass)):raise ValueError('invalid compact bypass presence')
    if len(packet)!=header.size+streams*itemsize+total+side:raise ValueError('invalid packet length')
    return n,streams,total,side,bypass,header.size,itemsize,dtype


def expand_legacy(packet):
    """Test/audit identity: restore legacy envelope, same exact coder payload."""
    n,streams,total,side,nb,header,size,dtype=packet_layout(packet)
    lengths=np.frombuffer(packet,dtype=dtype,count=streams,offset=header)
    at=header+streams*size
    if (lengths<4).any() or int(lengths.sum(dtype=np.int64))!=total:raise ValueError('invalid stream table')
    bypass=packet[at+total:]
    if not side:
        import zstandard as zstd
        bypass=zstd.ZstdCompressor(level=3).compress(b'')
    return pack_packet(n,streams,total,nb,lengths,packet[at:at+total],bypass)
