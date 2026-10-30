# Installation

## Choose your interface

| Interface | Install | Compute |
|---|---|---|
| [Hosted browser](https://xero-viewer.pages.dev/xero/) | None | WebGPU, with browser CPU fallback |
| [ImageJ/Fiji](../imagej/README.md) | One JAR | Java CPU; optional OpenCL GPU |
| [napari](ome-zarr.md#napari) | XERO reader + napari + Java 8+ | CPU; optional OpenCL GPU |
| Python CLI | This package + CUDA-enabled PyTorch/toolchain | NVIDIA CUDA |

## Python environment

Use Python 3.11+ on Linux with CUDA-enabled PyTorch, a matching CUDA development toolkit (`nvcc`), a C++17 compiler and Ninja.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install your compatible CUDA PyTorch build here.
python -m pip install -e '.[train,test,zarr]'
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
nvcc --version
xero --help
```

Tested environment: Python 3.12.13, PyTorch 2.13.0+cu130, CUDA 13.0, Triton 3.7.1 and RTX 5090. Check Triton with `python -c 'import triton'`.

Set `CUDA_HOME` if multiple CUDA toolkits are installed. The first encode/decode compiles CUDA/Triton code and can take minutes; subsequent runs reuse compiler caches. Extension builds default to `MAX_JOBS=2`.

For training, install the `train` extra (Matplotlib and TensorBoard). Supply raw or previously denoised images as input.

## Memory

Start with 32–128 pixel training patches and small batch/tile counts. Encoding preserves the original XY dimensions. GPU workspace grows with image area and model width; host staging is limited to one restart block.

## Common problems

- **Ninja/nvcc not found:** activate the intended environment, install the development tools, check `PATH` and `CUDA_HOME`.
- **No CUDA device:** use the viewer/ImageJ for decoding, or run the CLI on an NVIDIA CUDA machine.
- **Output already exists:** choose a new output directory. XERO does not overwrite an archive; a failed encode leaves an `INCOMPLETE` marker for inspection.
- **TIFF not memory-mappable:** convert to uncompressed TIFF or HDF5.
- **Out of memory:** reduce training batch/cache/patch size. In ImageJ, open a shorter frame range and optionally select the low-VRAM GPU fallback. In the browser, virtual loading is also available. A large native XY may still require more memory.
