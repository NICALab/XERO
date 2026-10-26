"""Generate a small, reproducible positive fluorescence-like movie; no download."""
import argparse
from pathlib import Path
import numpy as np
import h5py


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/demo/input.h5');p.add_argument('--frames',type=int,default=160);p.add_argument('--height',type=int,default=64);p.add_argument('--width',type=int,default=64);a=p.parse_args()
    if min(a.frames,a.height,a.width)<4:raise ValueError('Dimensions must be >=4')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    yy,xx=np.mgrid[:a.height,:a.width];rng=np.random.default_rng(7)
    with h5py.File(out,'x') as f:
        d=f.create_dataset('data',shape=(a.frames,a.height,a.width),dtype='float32')
        for t in range(a.frames):
            frame=np.full((a.height,a.width),100,dtype='float64')
            for j,(fy,fx) in enumerate([(.3,.3),(.65,.7),(.7,.25)]):
                amp=30+80*max(0,np.sin(t*.12+j*2))**3
                frame+=amp*np.exp(-((yy-fy*a.height)**2+(xx-fx*a.width)**2)/20)
            d[t]=(frame+rng.normal(0,.1,frame.shape)).astype('float32')
    print(out)


if __name__=='__main__':main()
