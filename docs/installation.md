# CUDA CLI installation

Training, encoding and native CUDA decoding require **Python 3.11+ on Linux**, an **NVIDIA GPU**, CUDA-enabled PyTorch, a matching CUDA development toolkit (`nvcc`) and a C++17 compiler.

For viewing without CUDA, use [ImageJ](../imagej/README.md), [napari](../napari/README.md) or the [Python reader](python.md).

## Install

```bash
git clone https://github.com/NICALab/XERO.git
cd XERO
python -m venv .venv
source .venv/bin/activate
# Install a CUDA-enabled PyTorch build matching your GPU and toolkit.
python -m pip install -e '.[train,test]'
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
nvcc --version
xero --help
```

Tested environment: Python 3.12.13, PyTorch 2.13.0+cu130, CUDA 13.0, Triton 3.7.1 and RTX 5090.

The first encode/decode compiles CUDA/Triton code and may take minutes. Subsequent runs reuse compiler caches. Extension builds default to `MAX_JOBS=2`.

## Troubleshooting

- **Ninja/nvcc not found:** check the active environment, `PATH` and `CUDA_HOME`.
- **No CUDA device:** check the PyTorch CUDA build and NVIDIA driver. Non-CUDA viewing options are linked above.
- **Output already exists:** choose a new path. Interrupted encodes leave a separate `.incomplete-*` directory.
- **TIFF not memory-mappable:** convert to uncompressed TIFF or HDF5.
- **Out of memory:** reduce training batch/tile/patch sizes. Encoding retains the original XY size and stages one restart block in host memory; a shorter `--restart` can reduce staging memory, but not full-frame GPU workspace.

[Training guide](training.md) · [Tutorials](../tutorials/README.md)
