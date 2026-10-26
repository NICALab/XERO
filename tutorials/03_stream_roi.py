"""Analyze decoded frames immediately, without writing a decompressed movie."""
import argparse
import csv
from pathlib import Path
from xero.archive import frames,manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--output',default='outputs/roi.csv');p.add_argument('--roi',type=int,nargs=4,required=True,metavar=('Y0','Y1','X0','X1'));p.add_argument('--device',default='cuda:0');a=p.parse_args()
    _,h,w=manifest(a.archive)['shape'];y0,y1,x0,x1=a.roi
    if not (0<=y0<y1<=h and 0<=x0<x1<=w):raise ValueError('ROI lies outside the image')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['frame_zero_based','mean_intensity'])
        for i,frame in frames(a.archive,device=a.device):
            writer.writerow([i,float(frame[y0:y1,x0:x1].mean())])


if __name__=='__main__':main()
