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

This repository contains **XERO only**: training, integer-model export, compression, decompression, streaming analysis examples and the ImageJ receiver. Baseline implementations, private imaging datasets and service credentials are not included.

## Start with the viewer

**[Open XERO in your browser](https://xero-viewer.pages.dev/xero/)** — choose a dataset, select a frame range and load it. Use GPU acceleration when available, or the CPU fallback. The viewer provides contrast controls, zoom, virtual loading, ROI measurements and TIFF export. A compatible browser is sufficient; no Python installation is needed.

Prefer Fiji/ImageJ? **[Download and install the XERO plugin](imagej/README.md)**. It reads the same supported compressed archive without a browser or Python process.

## Quick start

### 1. Install

The Python training/encoding/decoding path currently targets **Linux with an NVIDIA GPU**. Install a CUDA-enabled PyTorch build, its matching CUDA development toolkit (`nvcc`), and a C++17 compiler first. PyTorch is deliberately not installed automatically, to avoid replacing your CUDA build.

```bash
git clone https://github.com/NICALab/XERO.git
cd XERO
python -m venv .venv
source .venv/bin/activate
# Install a CUDA-enabled PyTorch build appropriate for your GPU and toolkit.
python -m pip install -e '.[train,test]'
xero --help
```

See [installation and platform requirements](docs/installation.md). Native extensions and CUDA graphs are compiled/initialized on first use; first-run latency is not steady-state decoding speed.

### 2. Run an end-to-end tutorial

```bash
bash tutorials/02_roundtrip.sh
```

The tutorial generates a tiny fluorescence-like movie, prepares disjoint training/calibration/test intervals, trains for **10 demonstration steps**, encodes it, decodes it, and checks the relative-error bound. Outputs appear under `outputs/demo/`. No research dataset or pretrained download is required. The 10-step model demonstrates functionality, **not the paper's compression performance**.

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

`--tolerance 0.01` means **1%**, not 0.01%. The checkpoint fixes the bound; encoding does not silently change it. Adjust patch/batch/cache settings to your hardware. A model trained on one acquisition is not guaranteed to compress another acquisition efficiently. [Training and calibration guide →](docs/training.md)

## What is guaranteed?

For each finite input pixel `x` and its reconstructed value `x_hat`, encoding independently checks:

```text
abs(x_hat - x) <= epsilon * abs(x)
```

There is **no absolute-error floor for dark pixels**; zero must reconstruct as zero. Unsafe cases use exact float32 escape values. The bound is relative to the **input supplied to the compressor**. If that input was denoised, the compression bound does not certify the denoiser's error relative to a clean specimen. XERO is bounded-error compression, not lossless compression.

Integer convolution and fixed probability tables support interoperable decoding. Historical profile IDs are retained inside files for compatibility. This is **not a claim that all operating systems, GPUs or drivers have been exhaustively tested**. Run the ImageJ compatibility check on new hardware. [Numerical contract and limitations →](docs/method.md)

## Work directly from compressed data

```bash
# Zero-based, half-open range: frames 100 through 199.
xero decode --archive outputs/compressed --start 100 --stop 200 \
  --output outputs/selected.h5

# Stream frames into an ROI mean; no decompressed movie on disk.
python tutorials/03_stream_roi.py --archive outputs/compressed \
  --roi 10 30 10 30 --output outputs/roi.csv
```

Memory is bounded by the restart block, neural workspaces and the consumer—not by the entire recording. It still scales with XY and model size; very large frames can exceed GPU memory. [Archive format and random access →](docs/archive.md)

## Repository guide

| Directory | Contents |
|---|---|
| [`src/xero`](src/xero) | CLI, streaming I/O, training interface and current integer runtime |
| [`src/ficompress`](src/ficompress) | Preserved low-level neural/entropy primitives and CUDA kernels |
| [`tutorials`](tutorials) | Synthetic round trip and on-demand ROI analysis |
| [`imagej`](imagej) | Installable JAR, Java/OpenCL sources, build instructions and license notices |
| [`tests`](tests) | Input/format checks and optional CUDA integration tests |
| [`docs`](docs) | Installation, scientific contract, training and release validation |

The `ficompress` namespace and `fiarc_*` format identifiers are compatibility details, not a different algorithm. Existing research files were not renamed or rewritten during release preparation.

## License and support

XERO source code is available under the [MIT license](LICENSE). Bundled ImageJ dependencies retain their own licenses; see [third-party notices](imagej/THIRD_PARTY_NOTICES.md). Public viewer datasets retain the rights of their original providers and are not relicensed by this repository.

For reproducible bug reports, include your OS, GPU, driver, software versions, archive profile, image dimensions and the complete error message. Do not attach private data or credentials. [Open an issue →](https://github.com/NICALab/XERO/issues)
