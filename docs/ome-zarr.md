# Open XERO OME-Zarr images

XERO uses **Zarr v3, OME-NGFF 0.5 and the `xero.restart_v1` codec**.
Open the same compressed image in the browser, ImageJ, napari or Python.

## napari

```bash
python -m pip install "napari[pyqt6]" "xero-microscopy[napari] @ https://xero-viewer.pages.dev/downloads/xero_microscopy-0.3.0-py3-none-any.whl"
```

Java 8 or newer is required for CPU/OpenCL decoding. The wheel includes the
decoder JAR. If Java is not on your PATH, set `XERO_JAVA` to a Java executable
(for example, Fiji's bundled runtime).

Download an [OpenScope calcium](https://xero-viewer.pages.dev/downloads/XERO_OpenScope_OME_Zarr.zip)
or [spinal-cord voltage](https://xero-viewer.pages.dev/downloads/XERO_Spinal_Cord_OME_Zarr.zip)
example, unzip it, then open the `.ome.zarr` folder in napari and choose **XERO**.
Alternatively:

```python
import napari

viewer = napari.Viewer()
viewer.open('XERO_OpenScope.ome.zarr', plugin='xero-microscopy')
napari.run()
```

CPU is the default. Set `XERO_ZARR_BACKEND=gpu:0` before launching napari to
select OpenCL GPU 0. Frames are decoded on demand through a lazy Dask array;
no fully decompressed volume is written to disk.

## Python

For array access without the napari GUI:

```bash
python -m pip install "xero-microscopy[zarr] @ https://xero-viewer.pages.dev/downloads/xero_microscopy-0.3.0-py3-none-any.whl"
```

```python
import os
os.environ['XERO_ZARR_BACKEND'] = 'gpu:0'  # omit for CPU

import zarr
image = zarr.open('XERO_OpenScope.ome.zarr', mode='r')['0']
frame = image[100]
crop = image[100:104, 50:150, 80:180]

from xero.omezarr import close_decoders
close_decoders()  # optional explicit resource release
```

Installing the package registers the codec with Zarr. Other OME-Zarr viewers
still need the XERO codec to reconstruct pixels.

For a source/editable installation, download the
[decoder JAR](https://xero-viewer.pages.dev/downloads/XERO_ImageJ.jar) and set
`XERO_DECODER_JAR` to its path. On a machine configured for the CUDA CLI,
`XERO_ZARR_BACKEND=cuda:0` uses the native CUDA decoder without Java.

To download a compressed subset:

```bash
python tutorials/05_download_compressed.py --dataset openscope \
  --start 100 --end 200 --output downloads/openscope.ome.zarr
```

Complete restart chunks are downloaded; `download.json` records the requested
local range. See [frame indexing](download.md).

## ImageJ and browser

In ImageJ, select the `.ome.zarr` folder in
**Compression (XERO) → Open compressed image**. Choose CPU or OpenCL GPU
decoding and a frame range. Selected frames are loaded into RAM with progress
and cancellation controls.

The [browser viewer](https://xero-viewer.pages.dev/xero/) reads compressed
packet ranges from the OME-Zarr container and decodes them locally using
WebGPU or CPU, with asynchronous prefetch.

## Memory and random access

The Python reader reuses one decoder and model per process. Switching models
releases the previous decoder. Seeking requires prediction context from the
nearest restart; access to arbitrary frames is not constant-cost.

Requested output is limited to 256 MiB per chunk. Zarr also allocates the total
requested array, so read large recordings in small ranges rather than
`image[:]`. napari uses one frame per lazy task.

## Compress your data

```bash
xero encode --input denoised.h5 --checkpoint outputs/model/latest.pt \
  --output outputs/movie.ome.zarr --axis t
# Use --axis z for a structural z-stack.
```

The encoder writes verified restart chunks directly to OME-Zarr. The destination
must not exist. Frame axes are explicit; physical pixel sizes are not guessed.

## Layout and integrity

```text
movie.ome.zarr/
  zarr.json                 OME axes/scales and XERO packet/chunk index
  0/zarr.json               float32 array and codec/model configuration
  0/c/0/0/0                 first independent restart
  0/c/1/0/0                 next independent restart
  browser-model/            optional browser tensor export
```

Dimensions are `[t,y,x]` or `[z,y,x]`, with one resolution level.
Separately encoded channels and planes remain separate images. The model
is stored once in array metadata, not in every chunk.

Checksums detect damaged models, packets and chunks. Missing chunks raise an
error rather than returning a blank frame. Hashes provide integrity, not
source authenticity; open trusted datasets only.

Array reading and compressed export are supported. Assigning new pixels with
`image[...] = ...` does not encode them; use `xero encode`.

[OME-NGFF specification →](https://ngff.openmicroscopy.org/0.5/)
