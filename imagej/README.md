# XERO for ImageJ / Fiji

**[XERO_ImageJ.jar — open file page](XERO_ImageJ.jar)** · [Source](src) · [Licenses](THIRD_PARTY_NOTICES.md)

On the file page, use **Download raw file** (the download icon). If this repository is private, sign in with an account that has access; a private repository returns 404 to unauthorized visitors.

Open compressed images directly in ImageJ. CPU decoding runs in Java; optional GPU decoding uses OpenCL through the bundled JOCL binding. **No Python, browser, local web server, PyTorch, or separate decoder application is needed.**

## Install

1. Download `XERO_ImageJ.jar` and copy it into Fiji/ImageJ's `plugins` directory.
2. Restart ImageJ, then run **Plugins → Compression (XERO) → Check CPU or GPU compatibility**. Start with CPU, then check GPU if available.
3. Choose **Plugins → Compression (XERO) → Open compressed image** and select the local archive folder containing `manifest.json`, `model.npz` and `group_*.bin`.
4. Use **Auto** compute, choose a frame range, and open it. Selected frames are decoded into an ordinary ImageJ stack. The progress window supports cancellation.

Frame numbers in ImageJ are **one-based and inclusive**. Python uses zero-based, half-open ranges. The plugin opens local/mounted folders, not arbitrary R2 URLs. For web-hosted data use the [browser viewer](https://xero-viewer.pages.dev/xero/).

## Try a small example

Choose one of these short biological recordings:

| Example | Frame size | Frames | Relative-error bound |
|---|---|---:|---:|
| [OpenScope dendritic calcium](examples/XERO_OpenScope.zip) | 496 × 496 | 128 | 1% |
| [Zebrafish spinal-cord voltage](examples/XERO_Spinal_Cord.zip) | 1024 × 148 | 128 | 0.5% |

1. Open the ZIP's GitHub file page, choose **Download raw file**, and unzip it.
2. In **Plugins → Compression (XERO) → Open compressed image**, select the extracted `XERO_OpenScope` or `XERO_Spinal_Cord` folder.
3. Select frames **1–128** and open the stack. Use ImageJ's brightness/contrast controls and frame slider to explore it. GPU is recommended; CPU is supported but can take longer.

Both examples contain the original compressed packets and the required decoder model. No training, Python installation or additional model download is required. [Example details](examples/README.md). To download a different recording or range without decompressing it, use the [Python archive downloader](../docs/download.md).

## Platforms and memory

Use a matching 64-bit Java runtime (Java 8+; Fiji normally supplies one). CPU mode needs no GPU driver. GPU mode needs the GPU vendor's OpenCL driver. Native JOCL bindings for Windows, macOS and Linux are packaged, but packaging is not proof of hardware compatibility. Local release testing is on Linux/RTX 5090; run the bundled compatibility check on your own machine.

Selected frames are loaded into RAM as float32 pixels (4 bytes per pixel). Request a shorter range for large recordings. The opener rejects an estimated output larger than one third of the ImageJ heap, leaving space for decoder workspaces; this is a guard, not a guarantee that every image fits. Neural workspace and temporal history also scale with XY.

**Fast (full frame)** is the default. Optional **Low VRAM (512/256-pixel tiles)** uses the same files and output arithmetic, but moves intermediate state through system RAM and can be substantially slower. It is a fallback, not a speed optimization, and does not reduce total system RAM use. For 2048² images in this mode, an 8-GiB Java heap and sufficient free RAM are recommended, not guaranteed to fit all models.

## Data integrity

The built-in compatibility check compares decoded float32 values with independently generated CUDA reference hashes.

Only open trusted archives. Integrity hashes detect corruption, not authenticity; this is research software, not a security-audited parser for hostile inputs.

## Build from source

Building requires a JDK 9+ and Python (optionally Eclipse ECJ); **end users do not need these build tools**. Both compiler paths use `--release 8` to restrict Java APIs as well as bytecode to Java 8. Download these Maven artifacts into `deps/`, using the filenames shown:

| Filename | Maven coordinates |
|---|---|
| `gson.jar` | `com.google.code.gson:gson:2.11.0` |
| `jocl.jar` | `org.jocl:jocl:2.0.5` |
| `aircompressor.jar` | `io.airlift:aircompressor:0.27` |

Supply ImageJ's `ij.jar` separately; it is not bundled into the plugin.

```bash
python imagej/build.py --deps deps --ij /path/to/ij.jar --out imagej/build
# Optional: --ecj /path/to/ecj.jar instead of javac
```

The source includes the small numerical self-check fixtures and third-party license texts. The synthetic current-format fixture is for compatibility testing, not a microscopy performance benchmark.
