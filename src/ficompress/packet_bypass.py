"""Conservative bypass-presence inspection of locally encoded packets.

No bypass symbols implies no float32 ESCAPE values. This reuses CPU metadata
already returned by rANS, instead of another GPU nonzero/D2H synchronization.
Unknown/malformed packets never authorize skipping escape extraction.
"""
import struct
import zstandard as zstd

GPU=struct.Struct('<8sQQQQQ');CPU=struct.Struct('<4sQQQ');PAIR=struct.Struct('<8sQ')
EMPTY_ZSTD=zstd.ZstdCompressor(level=3).compress(b'')

def packet_has_bypass(packet,_depth=0):
    if _depth>1:return True
    if packet.startswith((b'SRAN1201',b'SRAN1202')):
        from .seed_packet_format import packet_layout
        try:return packet_layout(packet)[4]!=0
        except ValueError:return True
    if packet.startswith(b'FPR1') and len(packet)>=CPU.size:
        _,n,side,payload=CPU.unpack_from(packet)
        if len(packet)!=CPU.size+side+payload:return True
        try:count=zstd.frame_content_size(packet[CPU.size:CPU.size+side])
        except zstd.ZstdError:return True
        return count!=0  # unknown size is conservative, not an empty certificate
    if packet.startswith(b'FSCHECK1') and len(packet)>=PAIR.size:
        _,n=PAIR.unpack_from(packet)
        if n<=0 or PAIR.size+n>=len(packet):return True
        return packet_has_bypass(packet[PAIR.size:PAIR.size+n],_depth+1) or packet_has_bypass(packet[PAIR.size+n:],_depth+1)
    return True
