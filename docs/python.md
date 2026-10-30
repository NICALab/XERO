# Read XERO OME-Zarr in Python

## Install

Requires Python 3.11+. CPU/OpenCL decoding also requires Java 8+; the decoder JAR is included in the package.

```bash
python -m pip install "xero-microscopy[zarr] @ https://xero-viewer.pages.dev/downloads/xero_microscopy-0.3.0-py3-none-any.whl"
```

If Java is not on your PATH, set `XERO_JAVA` to its executable. For source/editable installations without a bundled decoder, [download the JAR](https://xero-viewer.pages.dev/downloads/XERO_ImageJ.jar) and set `XERO_DECODER_JAR` to its path.

## Read a frame or ROI

Unzip an [example recording](../imagej/examples/README.md), then:

```python
import zarr
from xero.omezarr import close_decoders

image = zarr.open('XERO_OpenScope.ome.zarr', mode='r')['0']
try:
    frame = image[100]
    crop = image[100:104, 50:150, 80:180]
    print(crop.mean())
finally:
    close_decoders()
```

Arrays have `[t, y, x]` or `[z, y, x]` axes. Indices are zero-based; slice stops are exclusive. Installing XERO registers the codec with Zarr.

## Select a decoder

Set the backend before the first read:

```python
import os
os.environ['XERO_ZARR_BACKEND'] = 'gpu:0'
```

| Value | Decoder | Requires |
|---|---|---|
| `cpu` (default) | Java CPU | Java 8+ |
| `gpu:0` | OpenCL GPU 0 | Java 8+ and an OpenCL driver |
| `cuda:0` | Native CUDA GPU 0 | [CUDA CLI installation](installation.md) |

## Memory and access

Read large images in small ranges rather than `image[:]`: Zarr allocates the requested output, and the Java/OpenCL bridge limits each chunk's requested output to 256 MiB. Decoding starts at the preceding restart, so a small crop still requires prediction context. The decoder is reused between reads; `close_decoders()` releases it when finished.

The array is read-only. To compress new pixels, use [`xero encode`](../README.md#quick-start), not `image[...] = ...`.

[Download a compressed subset](download.md) · [OME-Zarr format](ome-zarr.md)
