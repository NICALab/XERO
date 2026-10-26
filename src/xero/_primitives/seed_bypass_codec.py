"""Exact alternative for rare signed integer symbols; no learned model."""
from functools import lru_cache
from pathlib import Path
from torch.utils.cpp_extension import load


@lru_cache(None)
def extension():
    return load(name='fiarc_seed_bypass_cpu',sources=[str(Path(__file__).parent/'csrc/seed_bypass_codec.cpp')],
        extra_cflags=['-O3'],with_cuda=False,verbose=False)


def encode(raw):return extension().encode(raw)
def decode(packet,count):return extension().decode(packet,count)
