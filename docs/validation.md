# Release validation

These are software-correctness checks, not new biological performance measurements.

## Codec checks

On the artificial 160 × 64 × 64 tutorial sequence:

- The original research runtime and this release produced identical `model.npz` bytes.
- All serialized entropy packet bytes were identical.
- All decoded float32 pixels were bit-exact.
- Independent reconstruction verified all 655,360 pixels with zero 1% relative-bound violations.

The supplied tutorial checkpoint was trained for ten steps on artificial data. Its compression ratio is not an estimate for biological recordings.

## ImageJ

The packaged JAR was rebuilt from the included Java/OpenCL sources and checked on Linux x86_64, OpenJDK 21 and an NVIDIA RTX 5090. Both CPU and OpenCL GPU paths passed:

- A 49 × 61 / 1% artificial-data fixture against its independent CUDA reference hash.
- The downloadable 160-frame example against CUDA-decoded pixels, including a seek across its 128-frame restart.

The JAR contains the compatibility checks so users can repeat them locally. Windows/macOS native bindings are bundled; those operating systems were not executed as part of this local release validation. Browser testing in the research project is a separate qualification record, not evidence that every ImageJ platform has been tested.

The JAR's byte size, SHA-256 and dependency hashes are recorded in [`imagej/build_receipt.json`](../imagej/build_receipt.json).

## Repeat the tests

```bash
python -m pytest tests -q
# Explicitly opt in on a free NVIDIA GPU:
XERO_RUN_CUDA_TESTS=1 python -m pytest tests -q
```

The GPU test checks original odd-sized XY, signed/zero/tiny source values, a restart boundary, exact seek equivalence, source-domain verification, overwrite rejection and corruption rejection. Default tests do not allocate a GPU.

An untrained degenerate checkpoint failed an audit during stress testing and was rejected without a final manifest. This is documented rather than bypassed: model export alone is not proof of codec qualification. Production checkpoint selection must include actual encode/decode auditing.

## Limits

This validation does not establish universal hardware determinism, optimal compression ratios, a speed comparison against baselines, or security against malicious files. Numerical compatibility is stronger than visual similarity; test new backends against reference pixels and leave the encode-time verification enabled.
