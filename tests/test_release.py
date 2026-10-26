import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import h5py
import pytest
from xero.io import source,bound_violations
from xero.training import prepare


def test_bound_signed_zero_and_nonfinite():
    a=np.array([0,100,-100],dtype='f4')
    assert bound_violations(a,[0,101,-99],.01)==0
    assert bound_violations(a,[.001,101.01,-99],.01)==2
    assert bound_violations(a,[0,np.nan,-100],.01)==1


def test_shapes_and_dtype(tmp_path):
    p=tmp_path/'x.npy';np.save(p,np.ones((20,8,9),dtype='uint16'))
    with source(p) as x:assert x.shape==(20,8,9)
    np.save(p,np.ones((2,3,8,9),dtype='f4'))
    with pytest.raises(ValueError):
        with source(p):pass
    np.save(p,np.ones((20,8,9),dtype='f8'))
    with pytest.raises(ValueError):
        with source(p):pass


def test_bank_is_disjoint_and_preserves_values(tmp_path):
    p=tmp_path/'input.npy';x=np.arange(40*8*9,dtype='f4').reshape(40,8,9);np.save(p,x)
    out=tmp_path/'bank';prepare([p],out)
    m=json.loads((out/'complete.json').read_text())
    recovered=[];last=0
    for row in m['movies']:
        assert row['source_frame_start']==last
        with h5py.File(row['path']) as f:recovered.append(f['data'][:])
        last+=row['shape'][0]
    assert np.array_equal(np.concatenate(recovered),x)
    with pytest.raises(FileExistsError):prepare([p],out)


def test_cli_help():
    r=subprocess.run([sys.executable,'-m','xero','--help'],capture_output=True,text=True)
    assert r.returncode==0 and 'encode' in r.stdout and 'verify' in r.stdout


def test_no_workstation_paths():
    root=Path(__file__).resolve().parents[1]
    for path in (root/'src').rglob('*.py'):
        text=path.read_text()
        assert '/home/minho' not in text and '/media/HDD' not in text,path


def test_restart_budget():
    pytest.importorskip('torch');pytest.importorskip('triton')
    from xero.runtime import setup
    setup()
    from reassessment_codec import interval
    for h,w in [(49,121),(496,496),(512,512),(2048,2048)]:
        k=interval(h,w)
        assert k>=128 and not k&(k-1) and k*h*w*2>100_000_000
        if k>128:assert (k//2)*h*w*2<=100_000_000
