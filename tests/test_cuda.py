"""Opt in with XERO_RUN_CUDA_TESTS=1; never grab a GPU in default tests."""
import os
from pathlib import Path
import numpy as np
import pytest
pytestmark=[pytest.mark.cuda,pytest.mark.skipif(os.environ.get('XERO_RUN_CUDA_TESTS')!='1',reason='explicit CUDA opt-in required')]


def test_odd_geometry_roundtrip(tmp_path):
    from xero.runtime import cuda
    cuda()
    import torch
    from xero.archive import encode,frames,verify
    # Tiny trained artificial-data fixture; not an untrained zero-output head
    # and not a claim of biological-data compression performance.
    cp=Path(__file__).resolve().parents[1]/'tutorials/demo_model.pt'
    rng=np.random.default_rng(2);x=rng.uniform(90,110,(130,17,21)).astype('f4')
    x[0,0,:3]=[0,-1,1e-8]
    inp=tmp_path/'input.npy';np.save(inp,x)
    out=tmp_path/'archive';r=encode(inp,cp,out,restart=128)
    assert r['violations']==0
    full=np.stack([y for _,y in frames(out)])
    seek=np.stack([y for _,y in frames(out,127,130)])
    assert np.array_equal(full[127:130].view('u4'),seek.view('u4'))
    assert verify(out,inp)['violations']==0
    with pytest.raises(FileExistsError):encode(inp,cp,out,restart=128)
    packet=out/'group_000000.bin';b=bytearray(packet.read_bytes());b[-1]^=1;packet.write_bytes(b)
    with pytest.raises(ValueError,match='integrity'):list(frames(out,0,1))
