"""Optional Java/OpenCL checks against a CUDA-decoded tutorial HDF5 file.

Not run by pytest: requires Java, ImageJ's ij.jar and an available OpenCL GPU.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile
import h5py
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--jar',type=Path,required=True)
    p.add_argument('--ij',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--gpu',default='0')
    p.add_argument('--output',type=Path,default=Path('outputs/imagej_checks.json'))
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    cp=os.pathsep.join(str(x.resolve()) for x in [a.jar,a.ij])
    rows={}
    def java(name,*args):
        r=subprocess.run(['java','-Djava.awt.headless=true','-cp',cp,'org.fiarc.strictjava.'+name,*map(str,args)],capture_output=True,text=True,check=True)
        return r.stdout
    for name,args in [('cpu',[]),('gpu',['gpu',a.gpu]),('gpu_tiled',['gpu',a.gpu,'32'])]:
        rows[name]=java('FiARC_Java_Check',*args);print(rows[name],flush=True)
    with tempfile.TemporaryDirectory(prefix='xero-imagej-test-') as temp:
        temp=Path(temp)
        with zipfile.ZipFile(root/'imagej/examples/XERO_demo.zip') as z:
            assert all(n.startswith('XERO_demo/') and '..' not in Path(n).parts for n in z.namelist())
            z.extractall(temp)
        with h5py.File(a.reference) as f:reference=f['data'][:]
        for mode in ['cpu','gpu']:
            for label,first,last in [('full',1,160),('restart_seek',129,131)]:
                out=temp/(mode+'_'+label)
                text=java('FiARCStandaloneSmoke',temp/'XERO_demo',first,last,mode,out,a.gpu)
                result=np.fromfile(out/'imagej.f32',dtype='<f4').reshape(last-first+1,64,64)
                expected=np.asarray(reference[first-1:last],dtype='<f4')
                assert np.array_equal(result.view('u4'),expected.view('u4')), (mode,label)
                row=json.loads(text);row['matches_cuda']=True;rows[mode+'_'+label]=row
                print(mode,label,'bit-exact',flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
