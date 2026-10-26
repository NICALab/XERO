# Tutorials

Run commands from the repository root after [installation](../docs/installation.md).

| Tutorial | What you learn | Requires |
|---|---|---|
| [`01_make_demo.py`](01_make_demo.py) | Generate a reproducible fluorescence-like single-channel movie | Python, NumPy, HDF5 |
| [`02_roundtrip.sh`](02_roundtrip.sh) | Prepare, train, encode, decode and verify a 1% bound | NVIDIA CUDA + training extras |
| [`03_stream_roi.py`](03_stream_roi.py) | Measure an ROI without saving a decompressed movie | A supported archive + CUDA |
| [`04_package_imagej_example.py`](04_package_imagej_example.py) | Package an archive for sharing with ImageJ users | An encoded archive + Python |
| [ImageJ walkthrough](../imagej/README.md) | Virtual loading, CPU/GPU selection and compatibility checks | ImageJ/Fiji |

```bash
bash tutorials/02_roundtrip.sh
python tutorials/03_stream_roi.py --archive outputs/demo/archive \
  --roi 10 30 10 30 --output outputs/demo/roi.csv
```

The full tutorial generates only artificial data and writes only `outputs/demo/`. Run it in a clean output location; rerunning refuses to overwrite previous results. It trains for ten steps to keep the software demonstration small. **Do not interpret its compression ratio as trained microscopy performance.**

`demo_model.pt` is the trusted 10-step artificial-data checkpoint produced during release validation (optimizer removed). To skip training, run `01_make_demo.py` and then encode its output with `--checkpoint tutorials/demo_model.pt`. This is not a pretrained biological model, and the numerical framework version can affect freshly repeated training.

To inspect a subset without accumulating all frames:

```python
from xero.archive import frames

for index, image in frames("outputs/demo/archive", start=128, stop=160):
    print(index, image[10:30, 10:30].mean())
```

Python ranges are zero-based and stop-exclusive. A consumer retaining every returned image will still accumulate memory; streaming only helps if the consumer releases frames it no longer needs.
