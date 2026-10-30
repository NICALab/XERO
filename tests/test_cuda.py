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
    out=tmp_path/'archive';r=encode(inp,cp,out,restart=128,output_format='native')
    assert r['violations']==0
    full=np.stack([y for _,y in frames(out)])
    seek=np.stack([y for _,y in frames(out,127,130)])
    assert np.array_equal(full[127:130].view('u4'),seek.view('u4'))
    assert verify(out,inp)['violations']==0
    from xero.omezarr import repack_archive, open_image
    from xero.omezarr.source import Source
    direct=tmp_path/'direct.ome.zarr'
    result=encode(inp,cp,direct,restart=128,axis='z')
    assert result['violations']==0 and result['format']=='xero.ome_zarr.v1'
    assert open_image(direct).metadata.dimension_names==('z','y','x')
    assert verify(direct,inp)['violations']==0
    direct_full=np.stack([y for _,y in frames(direct)])
    assert np.array_equal(full.view('u4'),direct_full.view('u4'))
    direct_seek=np.stack([y for _,y in frames(direct,127,130)])
    assert np.array_equal(seek.view('u4'),direct_seek.view('u4'))
    s=Source(direct)
    assert s.model==(out/'model.npz').read_bytes()
    for g in s.manifest['groups']:
        assert s.packet(g)==(out/g['file']).read_bytes()
    repacked=tmp_path/'repacked.ome.zarr';repack_archive(out,repacked,axis='z')
    for p in (direct/'0/c').rglob('*'):
        if p.is_file():assert p.read_bytes()==(repacked/p.relative_to(direct)).read_bytes()
    with pytest.raises(FileExistsError):encode(inp,cp,out,restart=128)
    packet=out/'group_000000.bin';b=bytearray(packet.read_bytes());b[-1]^=1;packet.write_bytes(b)
    with pytest.raises(ValueError,match='integrity'):list(frames(out,0,1))
