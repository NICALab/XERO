# Example recordings

Download and unzip an example, then open its `.ome.zarr` folder in [ImageJ](../README.md), [napari](../../napari/README.md) or [Python](../../docs/python.md). These examples reconstruct previously denoised recordings.

| Recording | Width × height | Frames | ZIP size | Bound |
|---|---|---:|---:|---:|
| [OpenScope dendritic calcium](https://xero-viewer.pages.dev/downloads/XERO_OpenScope_OME_Zarr.zip) | 496 × 496 | 128 | 2.86 MB | 1% |
| [Zebrafish spinal-cord voltage](https://xero-viewer.pages.dev/downloads/XERO_Spinal_Cord_OME_Zarr.zip) | 1024 × 148 | 128 | 1.99 MB | 0.5% |
| [OpenScope dendritic calcium — longer](https://xero-viewer.pages.dev/downloads/XERO_OpenScope_Longer_OME_Zarr.zip) | 496 × 496 | 5,120 | 99.96 MB | 1% |
| [Zebrafish spinal-cord voltage — longer](https://xero-viewer.pages.dev/downloads/XERO_Spinal_Cord_Longer_OME_Zarr.zip) | 1024 × 148 | 8,192 | 99.57 MB | 0.5% |

## Longer examples

The full longer examples decode to **5.04 GB** (OpenScope) and **4.97 GB** (spinal cord) of float32 pixels. MB and GB are decimal.

In ImageJ, allocate a **16,384 MB heap** through **Edit → Options → Memory & Threads** and restart before opening a full longer example. Ensure sufficient free RAM, or select a shorter range. The first 128 frames need approximately 126 MB or 78 MB, respectively, plus decoder workspace.

## Sources

- OpenScope: [Allen Institute OpenScope](https://openscope.alleninstitute.org/), subject 412933.
- Spinal-cord voltage: data used with [SUPPORT](https://doi.org/10.1038/s41592-023-02005-8).

Checksums and source ranges are in [`metadata.json`](metadata.json). Each ZIP's `download.json` maps source indices to local frames. Data-provider usage and attribution conditions apply.

[Download other recordings or frame ranges](../../docs/download.md)
