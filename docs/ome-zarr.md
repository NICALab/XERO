# OME-Zarr and napari

XERO OME-Zarr uses **Zarr v3, OME-NGFF 0.5 and the `xero.restart_v1` codec**.
The original compressed packets and integer model are preserved byte for byte;
repackaging does not retrain the network or change reconstructed pixels.

## Convert an existing archive

```bash
python -m pip install -e '.[zarr]'
xero repack --archive outputs/compressed --output outputs/movie.ome.zarr --axis t
# Use --axis z for a structural z-stack.
```

The destination must not exist. Frame axes are specified explicitly; physical
pixel sizes are not guessed. The Python `repack_archive` API also accepts
`scale` and `units`, in frame–y–x order.

## Read in Python

The [Python wheel](https://xero-viewer.pages.dev/downloads/xero_microscopy-0.2.0-py3-none-any.whl)
includes the decoder JAR. For example:

```bash
python -m pip install 'xero-microscopy[zarr] @ https://xero-viewer.pages.dev/downloads/xero_microscopy-0.2.0-py3-none-any.whl'
```

For a source/editable installation, download the current
[ImageJ decoder JAR](https://xero-viewer.pages.dev/downloads/XERO_ImageJ.jar)
and point `XERO_DECODER_JAR` at it. Java 8 or newer is required; Fiji's Java
runtime can be selected with `XERO_JAVA`. CUDA/PyTorch is not needed for this
reader. CPU is the default; `XERO_ZARR_BACKEND=gpu:0` selects OpenCL GPU 0.
On a machine configured for the CUDA CLI, `XERO_ZARR_BACKEND=cuda:0` uses that
same decoder directly inside Python and does not require Java.

```python
import os
os.environ['XERO_DECODER_JAR'] = '/path/to/XERO_ImageJ.jar'
os.environ['XERO_ZARR_BACKEND'] = 'gpu:0'  # omit for CPU

import zarr
image = zarr.open('outputs/movie.ome.zarr', mode='r')['0']
frame = image[100]                 # float32, decoded on demand
crop = image[100:104, 50:150, 80:180]

from xero.omezarr import close_decoders
close_decoders()                   # optional explicit GPU/resource release
```

Installing the package registers the codec with Zarr. Third-party readers still
need this codec; an arbitrary OME-Zarr viewer cannot decode XERO automatically.

The Java process and model are reused across requests. There is one resident
decoder per Python process. Reads are serialized to bound workspace; switching
models releases the previous decoder. Requested chunk output is limited to
256 MiB. Read large images in small frame ranges rather than `image[:]`.
Zarr allocates the total requested output separately, so this per-chunk limit
does not make a whole-array read of a terabyte recording safe.

To download a compressed subset from the published catalog:

```bash
python tutorials/05_download_compressed.py --dataset openscope \
  --start 100 --end 200 --format ome-zarr --output downloads/openscope.ome.zarr
```

The downloaded subset includes complete restart chunks and records the requested
local frame range in `download.json`. No pixels are decompressed during download.

## napari

```bash
python -m pip install -e '.[napari]'
```

With the decoder settings above, open an `.ome.zarr` folder in napari and
choose **XERO**, or use:

```python
import napari
viewer = napari.Viewer()
viewer.open('outputs/movie.ome.zarr', plugin='xero-microscopy')
napari.run()
```

The reader supplies a lazy Dask array with one frame per task. No fully decoded
volume is written to disk. Seeking still requires prediction context from the
nearest restart; it is not constant-cost access to an arbitrary pixel.

## ImageJ and browser

The updated ImageJ plugin accepts either an original XERO archive or an
`.ome.zarr` folder in **Compression (XERO) → Open compressed image**. The
progress/cancellation controls and CPU/OpenCL choices are unchanged.

The browser adapter reads the OME-Zarr index and requests only the required
packet byte ranges inside restart chunks, with one-packet asynchronous
prefetch. The same qualified WebGPU/WASM numerical kernels are used.

## Layout and integrity

```text
movie.ome.zarr/
  zarr.json                 OME axes/scales and XERO packet/chunk index
  0/zarr.json               float32 array and codec/model configuration
  0/c/0/0/0                 first independent restart
  0/c/1/0/0                 next independent restart
  browser-model/            optional browser tensor export
```

Each chunk wraps unmodified entropy packets. The model is stored once in array
metadata, losslessly compressed, not copied into every restart. Frame dimensions
are `[t,y,x]` or `[z,y,x]`; separately encoded channels/planes remain separate
images. This release does not invent a combined 5-D volume or add a multiscale
pyramid.

Model, packet and restart checksums detect corruption. Missing XERO chunks raise
an error rather than silently returning a blank frame. The native readers verify
every consumed packet; the Zarr codec also verifies the enclosing chunk. Hashes
provide integrity, not source authenticity: open trusted datasets only.

Array reads and byte-preserving export are supported. Assigning new pixels
through `image[...] = ...` is not a neural encoder; use `xero encode` and then
`xero repack`. The original archive format remains readable.

See the [OME-NGFF specification](https://ngff.openmicroscopy.org/0.5/) for the
container conventions.
