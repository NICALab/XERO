"""Repackage a compressed archive and read one small ROI without a full TIFF."""
import argparse
from xero.omezarr import repack_archive, open_image, close_decoders

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--archive',required=True)
p.add_argument('--output',required=True)
p.add_argument('--axis',choices=['t','z'],required=True)
p.add_argument('--frame',type=int,default=0)
a=p.parse_args()
print(repack_archive(a.archive,a.output,axis=a.axis))
try:
    image=open_image(a.output)
    roi=image[a.frame,:min(64,image.shape[1]),:min(64,image.shape[2])]
    print('Array:',image.shape,'ROI mean:',float(roi.mean()))
finally:
    close_decoders()
