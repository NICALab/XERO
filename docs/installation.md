# Installation

## Choose your interface

| Interface | Install | Compute |
|---|---|---|
| [Hosted browser](https://xero-viewer.pages.dev/xero/) | None | WebGPU, with browser CPU fallback |
| [ImageJ/Fiji](../imagej/README.md) | One JAR | Java CPU; optional OpenCL GPU |
| Python CLI | This package + CUDA-enabled PyTorch/toolchain | NVIDIA CUDA |

The Python CLI is not a portable CPU codec. Browser/ImageJ CPU execution is a different implementation of the supported decoding contract, not a Python fallback.

## Python environment

Use Python 3.11 or newer on Linux. Install a CUDA-enabled PyTorch release and a matching **development toolkit** (including `nvcc`), C++17 compiler and Ninja. A GPU driver alone is not a compiler toolkit. Install PyTorch following its official instructions for your GPU; do not replace a functioning CUDA build blindly.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install your compatible CUDA PyTorch build here.
python -m pip install -e '.[train,test]'
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
nvcc --version
xero --help
```

Local release validation uses Python 3.12.13, PyTorch 2.13.0+cu130, CUDA 13.0, Triton 3.7.1 and an RTX 5090. Minimum dependency declarations are not an exhaustive tested-version matrix. Triton comes with the supported Linux CUDA PyTorch installation; check `python -c 'import triton'`.

Set `CUDA_HOME` if multiple CUDA toolkits are installed. XERO sets `MAX_JOBS=2` by default to bound extension-build parallelism. The first encode/decode compiles CUDA/Triton code and captures graphs; it can take minutes. Subsequent runs reuse compiler caches. Do not report this startup period as steady-state FPS.

For `xero train`, install the `train` extra (Matplotlib and TensorBoard). Denoising is upstream of XERO: provide your frozen denoised or raw intensities explicitly. Denoisers and baseline packages are deliberately not installed here.

## Memory

Start with 32–128 pixel training patches and small batch/tile counts, then scale up. Training samples do not need to fill the whole field of view. Encoding preserves the original XY geometry; it does not silently pad it to a multiple of four. Restart blocks limit host staging, and large FOVs release the encoder before independent receiver verification, but GPU workspace still grows with image area and network width.

Do not run the integration test alongside a memory-critical experiment. CUDA tests are opt-in. Windows/macOS users can use browser/ImageJ; this release does not claim native Python CUDA installation on those platforms.

## Common problems

- **Ninja/nvcc not found:** activate the intended environment, install the development tools, check `PATH` and `CUDA_HOME`.
- **No CUDA device:** use the viewer/ImageJ for decoding, or run the CLI on an NVIDIA CUDA machine.
- **Output already exists:** choose a new output directory. XERO does not overwrite an archive; a failed encode leaves an `INCOMPLETE` marker for inspection.
- **TIFF not memory-mappable:** explicitly convert to uncompressed TIFF or HDF5. XERO will not silently load a huge compressed TIFF into RAM.
- **Out of memory:** reduce training batch/cache/patch size; for viewing use virtual mode and/or the ImageJ low-VRAM fallback. A large native XY may still require more memory.
