# Tutorials

Run commands from the repository root after [installation](../docs/installation.md).

| Tutorial | What you learn | Requires |
|---|---|---|
| [`01_make_demo.py`](01_make_demo.py) | Generate a reproducible fluorescence-like single-channel movie | Python, NumPy, HDF5 |
| [`02_roundtrip.sh`](02_roundtrip.sh) | Prepare, train, encode, decode and verify a 1% bound | NVIDIA CUDA + training extras |
| [`03_stream_roi.py`](03_stream_roi.py) | Measure an ROI without saving a decompressed movie | A supported archive + CUDA |
| [`04_package_imagej_example.py`](04_package_imagej_example.py) | Package an archive for sharing with ImageJ users | An encoded archive + Python |
| [`05_download_compressed.py`](05_download_compressed.py) | Select a public dataset and frame range; download compressed files only | Python standard library, no GPU |
| [ImageJ walkthrough](../imagej/README.md) | Open biological examples, CPU/GPU selection and compatibility checks | ImageJ/Fiji |

```bash
bash tutorials/02_roundtrip.sh
python tutorials/03_stream_roi.py --archive outputs/demo/archive \
  --roi 10 30 10 30 --output outputs/demo/roi.csv
```

The round-trip tutorial uses synthetic data and ten training steps, writing to `outputs/demo/`. Choose a fresh output directory when rerunning. This is a workflow demonstration, not a compression benchmark.

To skip training, run `01_make_demo.py` and encode its output with `--checkpoint tutorials/demo_model.pt`. This checkpoint is trained on the synthetic example.

To inspect a subset without accumulating all frames:

```python
from xero.archive import frames

for index, image in frames("outputs/demo/archive", start=128, stop=160):
    print(index, image[10:30, 10:30].mean())
```

Python ranges are zero-based and stop-exclusive. A consumer retaining every returned image will still accumulate memory; streaming only helps if the consumer releases frames it no longer needs.
