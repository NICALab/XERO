# XERO for ImageJ / Fiji

[Download XERO_ImageJ.jar](https://xero-viewer.pages.dev/downloads/XERO_ImageJ.jar) · [Example recordings](examples/README.md) · [Source](src)

## Install and open

Requires 64-bit Java 8+ (included with Fiji). Windows, macOS and Linux bindings are bundled; GPU mode requires an OpenCL driver.

1. Copy `XERO_ImageJ.jar` into Fiji/ImageJ's `plugins` directory, then restart ImageJ.
2. Run **Plugins → Compression (XERO) → Check CPU or GPU compatibility**.
3. Select **Plugins → Compression (XERO) → Open compressed image** and choose a local `.ome.zarr` folder.
4. Use **Auto** compute, select a frame range and open it. Frame numbers are **one-based and inclusive**.

Selected frames are decoded into an ImageJ stack. Loading can be cancelled from the progress window.

## Memory

Output uses 4 bytes per pixel and is limited to one third of the ImageJ heap to reserve decoder workspace. If a range exceeds the limit, shorten it or increase **Edit → Options → Memory & Threads**, then restart ImageJ.

**Fast (full frame)** is the default. **Low VRAM (512/256-pixel tiles)** reduces GPU memory use at the cost of speed and system RAM. For 2048² images in this mode, start with an 8-GiB Java heap; requirements depend on the model and selected range.

Only open trusted archives. The compatibility check compares decoded pixels with CUDA reference hashes; it does not authenticate a dataset.

## Build from source

Requires a JDK 9+ and Python; builds target Java 8. Download these Maven artifacts into `deps/`:

| Filename | Maven coordinates |
|---|---|
| `gson.jar` | `com.google.code.gson:gson:2.11.0` |
| `jocl.jar` | `org.jocl:jocl:2.0.5` |
| `aircompressor.jar` | `io.airlift:aircompressor:0.27` |

Supply ImageJ's `ij.jar` separately:

```bash
python imagej/build.py --deps deps --ij /path/to/ij.jar --out imagej/build
```

[Third-party licenses](THIRD_PARTY_NOTICES.md) · [Download other recordings](../docs/download.md)
