# Tutorials

Run from the repository root. See [CUDA installation](../docs/installation.md) for training/encoding, or [Python reading](../docs/python.md) for CPU/OpenCL access.

| Tutorial | Task | Requires |
|---|---|---|
| [`01_make_demo.py`](01_make_demo.py) | Generate a synthetic movie | NumPy, HDF5 |
| [`02_roundtrip.sh`](02_roundtrip.sh) | Train, encode, decode and verify | NVIDIA CUDA + training extra |
| [`03_stream_roi.py`](03_stream_roi.py) | Stream an ROI signal to CSV | NVIDIA CUDA |
| [`04_package_imagej_example.py`](04_package_imagej_example.py) | Package OME-Zarr as a ZIP | XERO Python package |
| [`05_download_compressed.py`](05_download_compressed.py) | Download a dataset/frame range | Python standard library |
| [`06_ome_zarr.py`](06_ome_zarr.py) | Read an ROI from OME-Zarr | Java 8+ or native CUDA |

## Round trip and ROI analysis

```bash
bash tutorials/02_roundtrip.sh
python tutorials/03_stream_roi.py --archive outputs/demo/movie.ome.zarr \
  --roi 10 30 10 30 --output outputs/demo/roi.csv
python tutorials/06_ome_zarr.py --archive outputs/demo/movie.ome.zarr --frame 100
```

The round trip uses ten training steps and writes to `outputs/demo/`; it demonstrates the workflow, not compression performance. Use a fresh output directory when rerunning.

To skip training, generate the input and use the supplied synthetic-data checkpoint:

```bash
python tutorials/01_make_demo.py
xero encode --input outputs/demo/input.h5 --checkpoint tutorials/demo_model.pt \
  --output outputs/demo/movie.ome.zarr --axis t
```

[Train on your own data](../docs/training.md) · [Download biological examples](../imagej/examples/README.md)
