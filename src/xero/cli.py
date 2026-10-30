"""Small command-line surface; use xero COMMAND --help for full options."""
import argparse
import json
import sys


def main():
    if len(sys.argv)>1 and sys.argv[1]=='train':
        from .training import train
        return train(sys.argv[2:])
    p=argparse.ArgumentParser(prog='xero',description='Error-bounded microscopy compression. Bounds are relative to the supplied input, not an unseen clean image.')
    s=p.add_subparsers(dest='command',required=True)
    prep=s.add_parser('prepare',help='Make train/calibration/test banks from frozen inputs')
    prep.add_argument('--input',nargs='+',required=True);prep.add_argument('--output',required=True);prep.add_argument('--key',default='data')
    s.add_parser('train',help='Train a causal codec (xero train --help)')
    enc=s.add_parser('encode',help='Encode and independently verify every pixel')
    enc.add_argument('--input',required=True);enc.add_argument('--checkpoint',required=True);enc.add_argument('--output',required=True)
    enc.add_argument('--key',default='data');enc.add_argument('--restart',type=int);enc.add_argument('--device',default='cuda:0')
    dec=s.add_parser('decode',help='Stream an archive to float32 HDF5 or BigTIFF')
    dec.add_argument('--archive',required=True);dec.add_argument('--output',required=True)
    dec.add_argument('--start',type=int,default=0);dec.add_argument('--stop',type=int);dec.add_argument('--device',default='cuda:0')
    ver=s.add_parser('verify',help='Check decoded pixels against the original input')
    ver.add_argument('--archive',required=True);ver.add_argument('--input',required=True);ver.add_argument('--key',default='data');ver.add_argument('--device',default='cuda:0')
    info=s.add_parser('inspect',help='Inspect archive metadata and index without a GPU')
    info.add_argument('--archive',required=True)
    rep=s.add_parser('repack',help='Repackage an existing archive as OME-Zarr without re-encoding')
    rep.add_argument('--archive',required=True);rep.add_argument('--output',required=True)
    rep.add_argument('--axis',choices=['t','z'],required=True)
    a=vars(p.parse_args());cmd=a.pop('command')
    if cmd=='repack':
        from .omezarr import repack_archive
        result=repack_archive(a['archive'],a['output'],axis=a['axis'])
    elif cmd=='prepare':
        from .training import prepare
        a['inputs']=a.pop('input');result=prepare(**a)
    else:
        from . import archive
        if cmd=='inspect':result=archive.manifest(a['archive'])
        else:
            if 'archive' in a:a['folder']=a.pop('archive')
            result=getattr(archive,cmd)(**a)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
