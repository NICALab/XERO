"""Exact existing sparse training statistics, cached outside the bank manifest."""
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from release_utils import write_json

STRIDES=(32,16,16)


def identity(movie):
    if movie['role']!='train':raise ValueError('Normalization is training-only')
    path=Path(movie['path']);st=path.stat()
    return dict(path=str(path),shape=movie['shape'],source_sha256=movie.get('sha256'),
        checkpoint_sha256=movie['checkpoint_sha256'],role='train',strides=list(STRIDES),
        version=1,arithmetic='numpy_float64_original_sparse_array_reduction',
        stat=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns])


def checksum(payload):
    return hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()


def statistics(movie,cache_dir=None):
    """Return (sum, squared-sum, count), hit; never approximate/change sample sites."""
    ident=identity(movie)
    path=(Path(cache_dir)/(hashlib.sha256(ident['path'].encode()).hexdigest()+'.json')) if cache_dir else None
    expected=int(np.prod([(n+s-1)//s for n,s in zip(movie['shape'],STRIDES,strict=True)]))
    if path is not None and path.exists():
        try:
            saved=json.loads(path.read_text());payload=saved['payload'];values=payload['statistics']
            if (saved['sha256']==checksum(payload) and payload['identity']==ident
                    and len(values)==3 and values[2]==expected
                    and np.isfinite(values).all() and values[1]>=0):
                return (np.float64(values[0]),np.float64(values[1]),int(values[2])),True
        except (KeyError,TypeError,ValueError):
            pass # Derived cache only: recompute stale/corrupt metadata from source.
    with h5py.File(movie['path']) as f:
        d=f['data']
        if list(d.shape)!=movie['shape'] or d.dtype!=np.float32:
            raise ValueError('Canonical training movie geometry/dtype changed')
        x=np.asarray(d[::32,::16,::16],dtype='f8')
    if not np.isfinite(x).all():raise ValueError('Nonfinite normalization source')
    if identity(movie)!=ident:raise ValueError('Training movie changed during statistics read')
    values=(x.sum(),(x*x).sum(),x.size)
    if path is not None:
        payload=dict(identity=ident,statistics=[float(values[0]),float(values[1]),int(values[2])])
        write_json(path,dict(payload=payload,sha256=checksum(payload)))
    return values,False
