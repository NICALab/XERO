# XERO for napari

Browse compressed OME-Zarr images with on-demand CPU or OpenCL GPU decoding.

## Install

Requires **Python 3.11+** and **Java 8+** for CPU/OpenCL decoding. The package includes the decoder JAR. If Java is not on your PATH, set `XERO_JAVA` to its executable, such as Fiji's bundled Java.

### Already have napari?

Activate the **same Python/conda environment as napari** and install the reader:

```bash
python -m pip install "xero-microscopy[napari] @ https://xero-viewer.pages.dev/downloads/xero_microscopy-0.3.0-py3-none-any.whl"
```

Restart napari. No additional Qt backend is needed. If your environment uses Python older than 3.11, create a new environment for this installation.

### New napari installation

In a Python 3.11+ environment:

```bash
python -m pip install "napari[pyqt6]" "xero-microscopy[napari] @ https://xero-viewer.pages.dev/downloads/xero_microscopy-0.3.0-py3-none-any.whl"
```

## Open an image

1. Download and unzip an [example recording](../imagej/examples/README.md).
2. Open its `.ome.zarr` folder in napari and select **XERO** when prompted.
3. Browse frames using the dimension slider. Frames are decoded as needed.

To launch from Python:

```python
import napari

viewer = napari.Viewer()
viewer.open('XERO_OpenScope.ome.zarr', plugin='xero-microscopy')
napari.run()
```

CPU is the default. For OpenCL GPU 0, set `XERO_ZARR_BACKEND=gpu:0` in the environment before launching napari, or add the following before the Python example:

```python
import os
os.environ['XERO_ZARR_BACKEND'] = 'gpu:0'
```

[Download other recordings](../docs/download.md) · [Read arrays without napari](../docs/python.md)
