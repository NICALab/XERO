# Release validation

Tests cover reconstruction accuracy, archive integrity and ImageJ compatibility.

## Codec checks

The 160 × 64 × 64 synthetic round-trip test verified all 655,360 pixels with zero violations of the 1% relative-error bound. Model bytes, entropy packets and decoded pixels matched the reference runtime exactly.

## ImageJ

Automated validation ran on Linux x86_64 with ImageJ 1.54p, Temurin Java 1.8.0_504 and RTX 5090 OpenCL:

- CPU and GPU decoding matched CUDA reference pixels on the 160-frame synthetic fixture.
- GPU decoding matched every pixel of the 128-frame OpenScope and spinal-cord examples.
- A rebased OpenScope excerpt starting at source frame 256 also passed.
- The menu-opening regression checks stack navigation, memory-size calculations and overflow rejection.

See the [validation results](imagej_java8_validation.json) and [build receipt](../imagej/build_receipt.json). Run **Plugins → Compression (XERO) → Check CPU or GPU compatibility** on a new machine.

## Repeat the tests

```bash
python -m pytest tests -q
# Explicitly opt in on a free NVIDIA GPU:
XERO_RUN_CUDA_TESTS=1 python -m pytest tests -q
```

The GPU test checks original odd-sized XY, signed/zero/tiny source values, a restart boundary, exact seek equivalence, source-domain verification, overwrite rejection and corruption rejection. Default tests do not allocate a GPU.

These checks validate the tested configurations, not every hardware/driver combination. Keep encode-time reconstruction verification enabled.
