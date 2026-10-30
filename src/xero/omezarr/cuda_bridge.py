"""Optional in-process NVIDIA backend using the existing CLI integer decoder."""
from collections import deque
import json
from pathlib import Path
import struct
import tempfile
import time
import numpy as np


class CudaSession:
    def __init__(self, config, backend):
        from ..runtime import cuda
        cuda(backend)
        import torch
        from .codec import unpack_model
        from large_fov_restart import MemoryReader
        self.backend=backend;self.errors=deque(maxlen=1);self.reader=None
        self.directory=tempfile.TemporaryDirectory(prefix='xero-cuda-model-')
        try:
            model=Path(self.directory.name)/'model.npz'
            model.write_bytes(unpack_model(config['model_zlib_base64'],config['model_sha256']))
            with np.load(model,allow_pickle=False) as data:
                meta=json.loads(data['metadata'].tobytes())
            if meta['profile']!=config['profile'] or meta['codec']['tolerance']!=config['tolerance']:
                raise ValueError('Model/configuration mismatch')
            with torch.inference_mode():
                self.reader=MemoryReader(model,(config['height'],config['width']))
        except Exception:
            self.close();raise

    def close(self):
        from memory_safety import close_runtime,release_caches
        close_runtime(self.reader);self.reader=None;release_caches();self.directory.cleanup()

    def decode(self, raw, selection):
        import torch
        from .codec import unpack_header,sha
        torch.cuda.set_device(self.backend)
        header=unpack_header(raw);size=struct.unpack_from('<I',raw,40)[0];offset=44+size
        packets=[];end=0
        for g in header['groups']:
            if g['start']!=end or not 0<g['frames']<=64 or g['bytes']<=0:
                raise ValueError('Invalid packet index')
            packet=raw[offset:offset+g['bytes']]
            if len(packet)!=g['bytes'] or sha(packet)!=g['sha256']:
                raise ValueError('Packet integrity failure')
            if end<selection[0][1]:packets.append(struct.pack('<Q',len(packet))+packet)
            end+=g['frames'];offset+=g['bytes']
        if end!=header['frames'] or offset!=len(raw):raise ValueError('Incomplete chunk')
        count=sum(struct.unpack_from('<I',p,16)[0] for p in packets)
        out=np.empty(tuple(len(range(*s)) for s in selection),dtype=np.float32)
        t,y,x=selection;written=0;started=time.perf_counter()
        with torch.inference_mode():
            for frame,data in self.reader.decoded_frames(packets,count):
                if t[0]<=frame<t[1] and (frame-t[0])%t[2]==0:
                    out[written]=data[0,0,y[0]:y[1]:y[2],x[0]:x[1]:x[2]].cpu().numpy();written+=1
        if written!=len(out):raise ValueError('Incomplete output')
        return out,dict(processed_frames=count,packets=len(packets),core_seconds=time.perf_counter()-started)
