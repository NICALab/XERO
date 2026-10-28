<div align="center">

<img src="assets/xero-banner.svg" alt="XERO — less data to move, more science to share" width="900">

**Cross-platform Error-bounded Representation of Optical microscopy images**

[![License: MIT](https://img.shields.io/badge/license-MIT-16a085.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](docs/installation.md)
[![Browser viewer](https://img.shields.io/badge/Open-browser%20viewer-0d9488.svg)](https://xero-viewer.pages.dev/xero/)
[![ImageJ](https://img.shields.io/badge/ImageJ-plugin-e18b34.svg)](imagej/README.md)

[**Try the viewer ↗**](https://xero-viewer.pages.dev/xero/) · [Quick start](#quick-start) · [Tutorials](tutorials/README.md) · [ImageJ plugin](imagej/README.md) · [How it works](docs/method.md)

</div>

---

XERO compresses microscopy movies and image stacks with a **pixelwise relative-error bound**, learned temporal prediction, sparse seed updates and an explicitly stored final correction. Compressed data can be decoded when it is needed, without keeping a second, fully decompressed volume on disk.

| Controlled error | Less data to move | Decode where you work |
|---|---|---|
| Audit every reconstructed pixel against the supplied input | Neural prediction plus entropy-coded seed and correction streams | CUDA command line, browser CPU/WebGPU, ImageJ CPU/OpenCL |

## Start with the viewer

**[Open XERO in your browser](https://xero-viewer.pages.dev/xero/)** — choose a dataset and frame range. The viewer supports CPU/WebGPU decoding, contrast controls, zoom, virtual loading, ROI measurements and TIFF export.

Prefer Fiji/ImageJ? **[Download the XERO plugin](https://xero-viewer.pages.dev/downloads/XERO_ImageJ.jar)** ([installation guide](imagej/README.md)) and try the **[OpenScope calcium](https://xero-viewer.pages.dev/downloads/XERO_OpenScope.zip)** or **[spinal-cord voltage](https://xero-viewer.pages.dev/downloads/XERO_Spinal_Cord.zip)** example.

Longer examples: **[OpenScope dendritic calcium](https://xero-viewer.pages.dev/downloads/XERO_OpenScope_Longer.zip)** — **5,120 frames / 99.95 MB**; **[zebrafish spinal-cord voltage](https://xero-viewer.pages.dev/downloads/XERO_Spinal_Cord_Longer.zip)** — **8,192 frames / 99.58 MB**. See [memory requirements](imagej/examples/README.md#longer-examples).

## Download compressed recordings

Select a recording and frame range from the viewer's catalog using the standalone Python downloader:

```bash
python tutorials/05_download_compressed.py --list
python tutorials/05_download_compressed.py --dataset openscope \
  --start 0 --end 128 --output downloads/openscope
python tutorials/05_download_compressed.py --dataset spinal-cord-voltage \
  --start 0 --end 128 --output downloads/spinal_cord
```

Ranges use **zero-based, end-exclusive indices**. [Dataset selection and frame indexing →](docs/download.md)

## Quick start

### 1. Install

The Python CLI requires **Linux with an NVIDIA GPU**, CUDA-enabled PyTorch, a matching CUDA development toolkit (`nvcc`) and a C++17 compiler.

```bash
git clone https://github.com/NICALab/XERO.git
cd XERO
python -m venv .venv
source .venv/bin/activate
# Install a CUDA-enabled PyTorch build appropriate for your GPU and toolkit.
python -m pip install -e '.[train,test]'
xero --help
```

See [installation requirements](docs/installation.md). The first run includes CUDA compilation and initialization.

### 2. Run an end-to-end tutorial

```bash
bash tutorials/02_roundtrip.sh
```

The tutorial generates a synthetic movie, prepares training/calibration/test splits, trains for 10 steps, and verifies an encode/decode round trip. Outputs appear under `outputs/demo/`. This short run demonstrates the workflow, not the paper's compression performance.

### 3. Compress your data

XERO accepts a single-channel sequence in `[frames, y, x]` order: time for movies, or z for structural stacks. Use `float32`, `uint16` or `uint8` input in HDF5 (`data`), NumPy, or memory-mappable uncompressed TIFF. Keep channels/recordings separate; do not flatten unrelated dimensions into time.

```bash
xero prepare --input denoised.h5 --output outputs/bank
xero train --bank outputs/bank --out outputs/model \
  --steps 50000 --tolerance 0.01 --history 64 --rollout 64 \
  --patch 128 --batch 8 --tiles 8 --refresh 500 --compile --skip-short-audits
xero encode --input denoised.h5 --checkpoint outputs/model/latest.pt \
  --output outputs/compressed
xero decode --archive outputs/compressed --output outputs/reconstructed.tif
xero verify --archive outputs/compressed --input denoised.h5
```

`--tolerance 0.01` means **1%**. The bound is fixed by the checkpoint. Adjust batch and patch sizes to your GPU memory, and select a checkpoint using calibration data. [Training guide →](docs/training.md)

## What is guaranteed?

For each finite input pixel `x` and its reconstructed value `x_hat`, encoding independently checks:

```text
abs(x_hat - x) <= epsilon * abs(x)
```

The bound applies to the **input supplied to the compressor**, with exact reconstruction at zero. For denoised input, it bounds compression error, not denoising error. [Numerical contract →](docs/method.md)

## Work directly from compressed data

```bash
# Zero-based, half-open range: frames 100 through 199.
xero decode --archive outputs/compressed --start 100 --stop 200 \
  --output outputs/selected.h5

# Stream frames into an ROI mean; no decompressed movie on disk.
python tutorials/03_stream_roi.py --archive outputs/compressed \
  --roi 10 30 10 30 --output outputs/roi.csv
```

Streaming avoids storing the full decoded recording; memory use still depends on frame size, model size and retained output. [Archive format and random access →](docs/archive.md)

## Repository guide

| Directory | Contents |
|---|---|
| [`src/xero`](src/xero) | Training, compression, streaming decoding and CUDA kernels |
| [`tutorials`](tutorials) | Training round trip, on-demand ROI analysis and compressed-data downloader |
| [`imagej`](imagej) | Plugin downloads, Java/OpenCL sources and build instructions |
| [`tests`](tests) | Input/format checks and optional CUDA integration tests |
| [`docs`](docs) | Installation, scientific contract, training and release validation |

## License and support

XERO source code is available under the [MIT license](LICENSE). Bundled ImageJ dependencies retain their own licenses; see [third-party notices](imagej/THIRD_PARTY_NOTICES.md). Public viewer datasets retain the rights of their original providers and are not relicensed by this repository.

For reproducible bug reports, include your OS, GPU, driver, software versions, archive profile, image dimensions and the complete error message. Do not attach private data or credentials. [Open an issue →](https://github.com/NICALab/XERO/issues)
