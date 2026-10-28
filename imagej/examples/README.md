# Try biological imaging data

Short and longer excerpts from the compressed datasets served by the XERO viewer:

| File | Imaging modality | Dimensions (width × height × frames) | Bound |
|---|---|---|---:|
| [XERO_OpenScope.zip](https://xero-viewer.pages.dev/downloads/XERO_OpenScope.zip) | Mouse dendritic calcium, OpenScope sub-412933 | 496 × 496 × 128 | 1% |
| [XERO_Spinal_Cord.zip](https://xero-viewer.pages.dev/downloads/XERO_Spinal_Cord.zip) | Zebrafish spinal-cord voltage | 1024 × 148 × 128 | 0.5% |

### Longer examples

| Example | Frames | ZIP download | Decoded float32 pixels | Bound |
|---|---:|---:|---:|---:|
| [Download OpenScope dendritic calcium (longer)](https://xero-viewer.pages.dev/downloads/XERO_OpenScope_Longer.zip) | 5,120 | 99.95 MB | 5.04 GB | 1% |
| [Download zebrafish spinal-cord voltage (longer)](https://xero-viewer.pages.dev/downloads/XERO_Spinal_Cord_Longer.zip) | 8,192 | 99.58 MB | 4.97 GB | 0.5% |

MB and GB are decimal units. Longer examples use the same frame sizes as their short counterparts.

Unzip the example, then select its folder in **Plugins → Compression (XERO) → Open compressed image**. Choose frames **1–128** for a quick test, or **1–5120** / **1–8192** for the full longer examples. GPU decoding is recommended.

To open a full longer example, set the Java heap to **16,384 MB** in **Edit → Options → Memory & Threads** and restart ImageJ. Ensure sufficient free RAM, or open a smaller range. The first 128 frames need approximately 126 MB (OpenScope) or 78 MB (spinal cord), plus workspace.

These examples reconstruct previously denoised recordings. Checksums and source ranges are recorded in [`metadata.json`](metadata.json); `download.json` inside each ZIP maps source indices to local frames.

## Download another recording or frame range

```bash
python tutorials/05_download_compressed.py --list
python tutorials/05_download_compressed.py --dataset openscope \
  --start 0 --end 128 --output downloads/openscope
```

See the [downloader instructions](../../docs/download.md).

## Provenance and usage

- OpenScope: [Allen Institute OpenScope](https://openscope.alleninstitute.org/), subject 412933.
- Spinal-cord voltage: data used with [SUPPORT](https://doi.org/10.1038/s41592-023-02005-8).

Original data-provider usage and attribution conditions apply. XERO's MIT software license does not relicense these biological datasets.
