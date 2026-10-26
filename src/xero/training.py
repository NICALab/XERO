"""User-facing preparation for the preserved causal training implementation."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .io import source


def prepare(inputs, output, key='data'):
    """Make contiguous 80/10/10 splits independently within each recording."""
    import h5py
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    # Validate before creating any output. No clipping or intensity rescaling.
    for path in inputs:
        with source(path,key) as x:
            if x.shape[0]<20:raise ValueError('At least 20 frames/slices per recording are required')
    output.mkdir(parents=True)
    movies=[]
    identity=hashlib.sha256(b'XERO user-supplied frozen input v1').hexdigest()
    for record,path in enumerate(inputs):
        with source(path,key) as x:
            n,h,w=x.shape;edges=[0,int(n*.8),int(n*.9),n]
            for role,lo,hi in zip(['train','calibration','test'],edges[:-1],edges[1:]):
                target=(output/f'record{record:03d}_{role}.h5').resolve()
                with h5py.File(target,'x') as f:
                    d=f.create_dataset('data',shape=(hi-lo,h,w),dtype='float32',chunks=(1,min(h,256),min(w,256)))
                    for i in range(lo,hi):
                        a=np.asarray(x[i],dtype='float32')
                        if not np.isfinite(a).all():raise ValueError('Nonfinite source')
                        d[i-lo]=a
                movies.append(dict(path=str(target),record=str(record),role=role,source_frame_start=lo,
                    shape=[hi-lo,h,w],checkpoint_sha256=identity))
    result=dict(plan=dict(checkpoint_sha256=identity,split='Per-record contiguous 80/10/10; no overlapping targets',
        note='Supplied intensities are frozen compression targets. No denoising performed here.'),movies=movies)
    (output/'complete.json').write_text(json.dumps(result,indent=2)+'\n')
    return dict(bank=str(output),recordings=len(inputs),movies=len(movies))


def train(args):
    import sys
    from .runtime import setup
    setup()
    import fiarc_train
    old=sys.argv
    # The production numerical objective is retained. Integer conversion and
    # source-domain auditing occur when encoding, not in this float trainer.
    sys.argv=['xero train',*args]
    try:
        fiarc_train.main()
    finally:
        sys.argv=old
