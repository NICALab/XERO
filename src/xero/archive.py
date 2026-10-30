"""Streaming file interface to the released integer decoder and packet format."""
import hashlib
import json
import math
import tempfile
from pathlib import Path
import types
import numpy as np
from .io import source, bound_violations
from .runtime import cuda


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def manifest(folder):
    folder = Path(folder)
    ome = (folder/'zarr.json').exists()
    if ome:
        from .omezarr.source import Source
        m = Source(folder).manifest
    else:
        m = json.loads((folder/'manifest.json').read_text())
    if m.get('profile') not in ('fiarc_integer_experimental_v1','fiarc_integer_boundscale_v2'):
        raise ValueError('Unsupported XERO archive profile')
    n,h,w = m['shape']; k = m['restart']
    if any(type(v) is not int or v<1 for v in (n,h,w,k)) or min(h,w)<4 or k<128 or k&(k-1):
        raise ValueError('Invalid geometry/restart')
    if m.get('history') != 'repeat_first_decoded' or not 0 < m['tolerance'] <= .5:
        raise ValueError('Invalid history/bound')
    if not ome and (m['model'] != 'model.npz' or digest(folder/'model.npz') != m['model_sha256']):
        raise ValueError('Model integrity failure')
    end=0
    for g in m['groups']:
        if Path(g['file']).name != g['file'] or '\\' in g['file'] or not g['file'].startswith('group_'):
            raise ValueError('Unsafe packet path')
        if g['start'] != end or not 1 <= g['frames'] <= 64 or end//k != (end+g['frames']-1)//k:
            raise ValueError('Invalid packet order/restart')
        end += g['frames']
    if end != n:
        raise ValueError('Incomplete archive index')
    return m


def encode(input, checkpoint, output, key='data', restart=None, device='cuda:0',
           axis='t', output_format='ome-zarr'):
    """Encode once, audit pixels, and write restart chunks directly to OME-Zarr."""
    from contextlib import nullcontext
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if output_format not in ('ome-zarr', 'native') or axis not in ('t', 'z'):
        raise ValueError('Specify format ome-zarr/native and axis t/z')
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output_format == 'ome-zarr':
        from .omezarr.writer import RestartWriter

    cuda(device)
    import torch
    import reassessment_codec as rc
    from ram_runtime import PinnedSource
    from large_fov_restart import SingleGroupSource, encode_segments
    from integer_archive import serialized_profile
    from release_utils import write_json

    with source(input,key) as array, torch.inference_mode():
        n,h,w = array.shape
        k = rc.interval(h,w) if restart is None else restart
        if type(k) is not int or k < 128 or k & (k-1):
            raise ValueError('Restart must be a power of two >=128')
        if output_format == 'native':
            output.mkdir(parents=True)
            (output/'INCOMPLETE').write_text('Only archives with manifest.json are complete.\n')
        # Only the small exported model is temporary. Compressed image packets
        # go straight to their final chunk, after the independent pixel audit.
        model_context = (tempfile.TemporaryDirectory(prefix='xero-encode-model-')
                         if output_format == 'ome-zarr' else nullcontext(output))
        with model_context as model_folder, rc.Codec(checkpoint,model_folder) as codec:
            model = Path(model_folder)/'model.npz'
            m = dict(profile=serialized_profile(model), shape=[n,h,w], restart=k,
                model='model.npz', model_sha256=digest(model), groups=[],
                history='repeat_first_decoded', tolerance=codec.epsilon)
            writer = (RestartWriter(output,m,model.read_bytes(),axis=axis)
                      if output_format == 'ome-zarr' else None)
            groups = m['groups']
            for start in range(0,n,k):
                x = np.array(array[start:min(n,start+k)],dtype='float32',order='C',copy=True)
                if not np.isfinite(x).all():
                    raise ValueError('Nonfinite input is not supported')
                host = torch.empty(x.shape,dtype=torch.float32,pin_memory=True)
                host.copy_(torch.from_numpy(x))
                src = (SingleGroupSource if h*w >= 1_000_000 else PinnedSource)(host)
                rt = codec.encoder((h,w))
                segments,blob,_ = encode_segments(rt,src,k)
                del src,host,blob,rt
                if h*w >= 1_000_000:
                    codec.release_encoder()
                packets = segments[0][2]
                reader = codec.receiver((h,w))
                count = 0
                for i,rec in reader.decoded_frames(packets,len(x)):
                    count += 1
                    y = rec[0,0].cpu().numpy()
                    if bound_violations(x[i],y,codec.epsilon):
                        raise RuntimeError(f'Bound validation failed at frame {start+i}')
                if count != len(x):
                    raise RuntimeError('Incomplete independent verification')
                del reader,rec,y
                if h*w >= 1_000_000:
                    codec.release_reader()
                offset = start
                records = []
                for wrapped in packets:
                    payload = memoryview(wrapped)[8:]
                    _,frames,ph,pw,*_ = rc.sequence_codec.HEADER.unpack_from(payload)
                    if (ph,pw) != (h,w):
                        raise ValueError('Packet geometry mismatch')
                    name = f'group_{len(groups):06d}.bin'
                    g = dict(file=name,start=offset,frames=frames,bytes=len(payload),
                             sha256=hashlib.sha256(payload).hexdigest())
                    groups.append(g)
                    if writer is None:
                        (output/name).write_bytes(payload)
                    else:
                        records.append((g,payload))
                    offset += frames
                if writer is not None:
                    writer.append(start,records)
                del records,payload,wrapped,segments,packets,x
                print(json.dumps(dict(encoded=min(n,start+k),total=n,violations=0)),flush=True)
            report = dict(shape=[n,h,w],source_dtype=str(array.dtype),
                source_bytes=n*h*w*array.dtype.itemsize,float32_bytes=n*h*w*4,
                tolerance=m['tolerance'],verified_pixels=n*h*w,violations=0)
            if writer is not None:
                return writer.finish(report)
            write_json(output/'manifest.json',m)
            total = sum(g['bytes'] for g in groups)+model.stat().st_size+(output/'manifest.json').stat().st_size
            result = dict(report,compressed_bytes=total,
                compression_ratio=report['source_bytes']/total,uint16_equivalent_ratio=n*h*w*2/total)
            write_json(output/'encode_report.json',result)
            (output/'INCOMPLETE').unlink()
            return result


def frames(folder, start=0, stop=None, device='cuda:0'):
    """Stream native or OME-Zarr packets through the same CUDA decoder."""
    folder = Path(folder)
    if (folder/'zarr.json').exists():
        from .omezarr.source import Source
        ome = Source(folder)
        # Only the small compressed model is temporary; never decoded images.
        with tempfile.TemporaryDirectory(prefix='xero-model-') as temp:
            model = Path(temp)/'model.npz';model.write_bytes(ome.model)
            yield from _frames(folder,start,stop,device,ome,model)
    else:
        yield from _frames(folder,start,stop,device)


def _frames(folder, start=0, stop=None, device='cuda:0', ome=None, model=None):
    """Yield (absolute index, float32 image). Finish each consumed entropy packet."""
    cuda(device)
    import torch
    from large_fov_restart import MemoryReader
    from integer_archive import serialized_profile,load_model
    from memory_safety import close_runtime,release_caches
    folder=Path(folder);m=manifest(folder);n,h,w=m['shape'];k=m['restart']
    stop=n if stop is None else stop
    if not 0<=start<stop<=n:raise ValueError('Invalid zero-based half-open range')
    model = folder/'model.npz' if model is None else model
    if serialized_profile(model)!=m['profile']:raise ValueError('Model profile mismatch')
    # Check the embedded bound before allocating neural GPU workspaces.
    with np.load(model,allow_pickle=False) as f:
        meta=json.loads(f['metadata'].tobytes())
    if meta['codec']['tolerance'] != m['tolerance']: raise ValueError('Model/manifest bound mismatch')
    reader=None
    with torch.inference_mode():
        try:
            reader=MemoryReader(model,(h,w))
            for origin in range(start//k*k,stop,k):
                selected=[g for g in m['groups'] if origin<=g['start']<min(origin+k,stop)]
                def packets():
                    import struct
                    for g in selected:
                        if ome is not None:
                            raw=ome.packet(g)
                        else:
                            path=folder/g['file']
                            if path.stat().st_size!=g['bytes']: raise ValueError('Packet size mismatch')
                            raw=path.read_bytes()
                        if hashlib.sha256(raw).hexdigest()!=g['sha256']:raise ValueError('Packet integrity failure')
                        header=__import__('sequence_codec').HEADER.unpack_from(raw)
                        if header[:4]!=(b'FSEEDSG1',g['frames'],h,w):raise ValueError('Packet header mismatch')
                        yield struct.pack('<Q',len(raw))+raw
                count=sum(g['frames'] for g in selected)
                for i,rec in reader.decoded_frames(packets(),count):
                    absolute=origin+i
                    if start<=absolute<stop:
                        yield absolute,rec[0,0].cpu().numpy().copy()
        finally:
            close_runtime(reader)
            reader=None
            release_caches()


def decode(folder, output, start=0, stop=None, device='cuda:0'):
    import h5py
    import tifffile
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    m=manifest(folder);stop=m['shape'][0] if stop is None else stop
    if not 0<=start<stop<=m['shape'][0]:raise ValueError('Invalid frame range')
    if output.suffix.lower() not in ('.h5','.hdf5','.tif','.tiff'):raise ValueError('Output must be HDF5 or TIFF')
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_name(output.name+'.partial')
    if tmp.exists():raise FileExistsError(tmp)
    count=0
    if output.suffix.lower() in ('.h5','.hdf5'):
        with h5py.File(tmp,'x') as f:
            d=f.create_dataset('data',shape=(stop-start,*m['shape'][1:]),dtype='float32',chunks=(1,*m['shape'][1:]))
            for i,frame in frames(folder,start,stop,device):d[i-start]=frame;count+=1
    else:
        with tifffile.TiffWriter(tmp,bigtiff=True) as f:
            for i,frame in frames(folder,start,stop,device):
                f.write(frame,contiguous=True,photometric='minisblack',metadata=None);count+=1
    if count!=stop-start:raise RuntimeError('Incomplete decode')
    tmp.replace(output)
    return dict(output=str(output),frames=count,dtype='float32')


def verify(folder,input,key='data',device='cuda:0'):
    m=manifest(folder);bad=count=0
    with source(input,key) as x:
        if list(x.shape)!=m['shape']:raise ValueError('Source geometry differs')
        for i,y in frames(folder,device=device):
            bad+=bound_violations(x[i],y,m['tolerance']);count+=1
    if bad:raise RuntimeError(f'{bad} pixel-bound violations')
    return dict(frames=count,violations=bad,tolerance=m['tolerance'])
