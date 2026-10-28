# XERO for ImageJ / Fiji

**[Download XERO_ImageJ.jar](https://xero-viewer.pages.dev/downloads/XERO_ImageJ.jar)** · [Source](src) · [Licenses](THIRD_PARTY_NOTICES.md)

Open XERO-compressed images in Fiji/ImageJ with CPU or OpenCL GPU decoding.

## Install

1. [Download `XERO_ImageJ.jar`](https://xero-viewer.pages.dev/downloads/XERO_ImageJ.jar) and copy it into Fiji/ImageJ's `plugins` directory. Keep the `.jar` extension; do not unzip the plugin.
2. Restart ImageJ, then run **Plugins → Compression (XERO) → Check CPU or GPU compatibility**. Start with CPU, then check GPU if available.
3. Choose **Plugins → Compression (XERO) → Open compressed image** and select the local archive folder containing `manifest.json`, `model.npz` and `group_*.bin`.
4. Use **Auto** compute, choose a frame range, and open it. Selected frames are decoded into an ordinary ImageJ stack. The progress window supports cancellation.

Frame numbers are **one-based and inclusive**. Select a local or mounted archive folder. For online viewing, use the [browser viewer](https://xero-viewer.pages.dev/xero/).

## Try a small example

Choose one of these short biological recordings:

| Example | Frame size | Frames | Relative-error bound |
|---|---|---:|---:|
| [OpenScope dendritic calcium](https://xero-viewer.pages.dev/downloads/XERO_OpenScope.zip) | 496 × 496 | 128 | 1% |
| [Zebrafish spinal-cord voltage](https://xero-viewer.pages.dev/downloads/XERO_Spinal_Cord.zip) | 1024 × 148 | 128 | 0.5% |

1. Click an example link above to download its ZIP, then unzip it.
2. In **Plugins → Compression (XERO) → Open compressed image**, select the extracted `XERO_OpenScope` or `XERO_Spinal_Cord` folder.
3. Select frames **1–128** and open the stack. Use ImageJ's brightness/contrast controls and frame slider to explore it. GPU is recommended; CPU is supported but can take longer.

For other recordings or frame ranges, use the [Python archive downloader](../docs/download.md).

### Longer examples

| Example | Frames | ZIP size | Bound |
|---|---:|---:|---:|
| [Download OpenScope dendritic calcium (longer)](https://xero-viewer.pages.dev/downloads/XERO_OpenScope_Longer.zip) | 5,120 | 99.95 MB | 1% |
| [Download zebrafish spinal-cord voltage (longer)](https://xero-viewer.pages.dev/downloads/XERO_Spinal_Cord_Longer.zip) | 8,192 | 99.58 MB | 0.5% |

Open frames **1–5120** or **1–8192**, or a smaller range. Each full example needs approximately **5 GB for float32 pixels**, plus decoder workspace. To open all frames, allocate a **16 GiB ImageJ heap** on a computer with sufficient free RAM. [Memory setup →](examples/README.md#longer-examples)

## Platforms and memory

Requires 64-bit Java 8+ (included with Fiji). Windows, macOS and Linux bindings are bundled. GPU mode requires an OpenCL driver; run the compatibility check on your machine.

Selected frames are loaded into RAM as float32 pixels (4 bytes per pixel). The plugin limits output to one third of the ImageJ heap to leave room for decoder workspace. If a range exceeds this limit, reduce it or increase the heap.

**Fast (full frame)** is the default. **Low VRAM (512/256-pixel tiles)** reduces GPU memory use at the cost of speed and system RAM. For 2048² images in this mode, start with an 8-GiB Java heap; requirements depend on the model and selected range.

## Data integrity

The built-in compatibility check compares decoded float32 values with independently generated CUDA reference hashes.

Only open trusted archives; integrity hashes detect corruption, not authenticity.

## Build from source

Requires a JDK 9+ and Python (optionally Eclipse ECJ). Builds target Java 8 with `--release 8`. Download these Maven artifacts into `deps/`:

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
