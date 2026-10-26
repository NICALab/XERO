"""Load the curated numerical implementation without research-tree imports."""
import os
import sys
from pathlib import Path


def setup():
    folder = str(Path(__file__).parent / '_internal')
    if folder not in sys.path:
        sys.path.insert(0, folder)
    prefix = Path(sys.base_prefix)
    if (prefix / 'bin/nvcc').exists():
        os.environ.setdefault('CUDA_HOME', str(prefix))
        os.environ['PATH'] = str(prefix/'bin') + os.pathsep + os.environ.get('PATH','')
    os.environ.setdefault('MAX_JOBS', '2')


def cuda(device='cuda:0'):
    setup()
    import torch
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('The Python encoder/decoder needs NVIDIA CUDA. Use the browser or ImageJ for CPU/other GPUs.')
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch._inductor.config.compile_threads = 2
