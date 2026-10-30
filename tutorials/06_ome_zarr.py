"""Read one small ROI from XERO OME-Zarr without exporting a full TIFF."""
import argparse
from xero.omezarr import open_image, close_decoders

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--archive',required=True)
p.add_argument('--frame',type=int,default=0)
a=p.parse_args()
try:
    image=open_image(a.archive)
    roi=image[a.frame,:min(64,image.shape[1]),:min(64,image.shape[2])]
    print('Array:',image.shape,'ROI mean:',float(roi.mean()))
finally:
    close_decoders()
