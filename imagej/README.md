# XERO for ImageJ / Fiji

**[Download XERO_ImageJ.jar](XERO_ImageJ.jar?raw=true)** · [Source](src) · [Licenses](THIRD_PARTY_NOTICES.md)

Open compressed images directly in ImageJ. CPU decoding runs in Java; optional GPU decoding uses OpenCL through the bundled JOCL binding. **No Python, browser, local web server, PyTorch, or separate decoder application is needed.**

## Install

1. Download `XERO_ImageJ.jar` and copy it into Fiji/ImageJ's `plugins` directory.
2. Restart ImageJ, then run **Plugins → XERO → Check CPU or GPU compatibility**. Start with CPU, then check GPU if available.
3. Choose **Plugins → XERO → Open compressed image** and select the local archive folder containing `manifest.json`, `model.npz` and `group_*.bin`.
4. Use **Auto** compute and **Virtual stack** for interactive viewing. The progress window supports cancellation.

Frame numbers in ImageJ are **one-based and inclusive**. Python uses zero-based, half-open ranges. The plugin opens local/mounted folders, not arbitrary R2 URLs. For web-hosted data use the [browser viewer](https://xero-viewer.pages.dev/xero/).

## Try a small example

**[Download XERO_demo.zip](examples/XERO_demo.zip?raw=true)** (about 0.46 MB).

1. Unzip the download.
2. In **Plugins → XERO → Open compressed image**, select the extracted `XERO_demo` folder.
3. Select frames **1–160** and open the stack. Use ImageJ's brightness/contrast controls and frame slider to explore it.

This is a **synthetic fluorescence-like movie**, 64 × 64 pixels × 160 frames, compressed with a 1% relative-error bound. It is an installation example, not biological data or a compression-performance benchmark. No training, Python installation or additional model download is required. [Example details and reproduction](examples/README.md).

## Platforms and memory

Use a matching 64-bit Java runtime (Java 8+; Fiji normally supplies one). CPU mode needs no GPU driver. GPU mode needs the GPU vendor's OpenCL driver. Native JOCL bindings for Windows, macOS and Linux are packaged, but packaging is not proof of hardware compatibility. Local release testing is on Linux/RTX 5090; run the bundled compatibility check on your own machine.

Virtual loading keeps a bounded output-frame cache and decodes from the preceding I-frame after a distant seek. It does not keep the full recording in RAM. Neural workspace and temporal history still scale with XY; a large image can exceed a small GPU's memory.

**Fast (full frame)** is the default. Optional **Low VRAM (512/256-pixel tiles)** uses the same files and output arithmetic, but moves intermediate state through system RAM and can be substantially slower. It is a fallback, not a speed optimization, and does not reduce total system RAM use. For 2048² images in this mode, an 8-GiB Java heap and sufficient free RAM are recommended, not guaranteed to fit all models.

## Data integrity

The built-in compatibility check compares decoded float32 values with independently generated CUDA reference hashes.

Only open trusted archives. Integrity hashes detect corruption, not authenticity; this is research software, not a security-audited parser for hostile inputs.

## Build from source

Building requires a JDK (or Java plus Eclipse ECJ) and Python; **end users do not need these build tools**. Download these Maven artifacts into `deps/`, using the filenames shown:

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
