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
- The artificial 160-frame regression fixture against CUDA-decoded pixels, including a seek across its 128-frame restart.

The JAR contains the compatibility checks so users can repeat them locally. Windows/macOS native bindings are bundled; those operating systems were not executed as part of this local release validation. Browser testing in the research project is a separate qualification record, not evidence that every ImageJ platform has been tested.

The JAR's byte size, SHA-256 and dependency hashes are recorded in [`imagej/build_receipt.json`](../imagej/build_receipt.json).

### Java 8 opener regression

A Windows/Fiji Java 8 report exposed a newer-JDK API in the menu opener's
output-memory estimate: `Math.multiplyExact(long, int)` (introduced in Java 9).
The decoder-only compatibility check did not execute this UI path. The estimate
now uses checked `long, long` multiplication, including the final byte-size
factor. Both javac and ECJ builds now use `--release 8`; bytecode targeting alone
does not restrict Java library API availability.

`tests/ImageJOpenRegression.java` exercises the actual menu dialog, ordinary
ImageJ stack opening, reference pixels and stack navigation. It also tests
large byte counts and overflow rejection. This is an
additional opener regression, not a replacement for the codec checks above.

On 2026-10-27, this regression passed on Linux x86_64, ImageJ 1.54p,
Temurin Java 1.8.0_504, using Java CPU and RTX 5090 OpenCL. Both backends
opened the artificial 160-frame regression fixture. Its pixels matched the CUDA reference
bit-for-bit. The 128-frame OpenScope calcium and spinal-cord voltage examples also
opened successfully with GPU decoding, with every pixel bit-exact against CUDA.
A rebased OpenScope excerpt beginning at source frame 256 passed the same check.
These are correctness checks, not throughput benchmarks. Detailed results are in
[`imagej_java8_validation.json`](imagej_java8_validation.json). This does not
claim a local execution test on Windows/Java 1.8.0_172.

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
