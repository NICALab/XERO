# Try biological imaging data

Short and longer excerpts from the compressed datasets served by the XERO viewer:

| File | Imaging modality | Dimensions (width × height × frames) | Bound |
|---|---|---|---:|
| [XERO_OpenScope.zip](XERO_OpenScope.zip) | Mouse dendritic calcium, OpenScope sub-412933 | 496 × 496 × 128 | 1% |
| [XERO_Spinal_Cord.zip](XERO_Spinal_Cord.zip) | Zebrafish spinal-cord voltage | 1024 × 148 × 128 | 0.5% |

### Longer examples

Each ZIP is below **100 MB** (100,000,000 bytes) and contains a continuous recording excerpt, not repeated frames.

| Example | Frames | ZIP download | Decoded float32 pixels | Bound |
|---|---:|---:|---:|---:|
| [Download OpenScope dendritic calcium (longer)](https://xero-viewer.pages.dev/downloads/XERO_OpenScope_Longer.zip) | 5,120 | 99.95 MB | 5.04 GB | 1% |
| [Download zebrafish spinal-cord voltage (longer)](https://xero-viewer.pages.dev/downloads/XERO_Spinal_Cord_Longer.zip) | 8,192 | 99.58 MB | 4.97 GB | 0.5% |

The longer excerpts use the same frame sizes as the short examples. MB and GB here are decimal units. Unzipping leaves the images compressed; the float32 memory is needed when opening them in ImageJ.

The **longer-example links above download the ZIP directly**, without a login or API key. Files are stored in Cloudflare R2 and streamed through our read-only download endpoint, with support for resumed downloads. Identical [OpenScope](XERO_OpenScope_Longer.zip) and [spinal-cord](XERO_Spinal_Cord_Longer.zip) files remain in this repository as mirrors.

For the short examples or GitHub mirrors, open the file page and click **Download raw file** (download icon). If the repository is private, GitHub requires an authorized login. Unzip the downloaded file before opening it in ImageJ.

In Fiji/ImageJ choose **Plugins → Compression (XERO) → Open compressed image**, select the extracted archive folder, and choose a frame range. For a quick check, open frames **1–128**. For the complete longer examples, choose **1–5120** (OpenScope) or **1–8192** (spinal cord). Use **Auto** or **GPU / OpenCL** when available. CPU also works, but is slower. The first 128 frames require approximately 126 MB and 78 MB of float32 pixels respectively, plus decoder workspace.

**Opening the entire longer example:** allow a **16 GiB Java heap** (16,384 MB in ImageJ's **Edit → Options → Memory & Threads**, then restart), provided your computer has sufficient free RAM. The plugin reserves headroom by limiting the selected output to one third of the heap. Alternatively, open a smaller frame range; there is no need to decode the whole example. Workspace and ImageJ overhead are additional to the pixel sizes above.

These are reconstructed, previously denoised recordings, not raw/noisy movies. Each ZIP includes `manifest.json`, `model.npz` and the required entropy packets. No training or additional model download is needed. Source compressed model and packet bytes are unchanged; only the manifest is restricted to the excerpt. A tiny excerpt's model overhead is not representative of full-recording compression ratios.

`metadata.json` records each ZIP's byte size, checksum, original dataset ID and frame range. `download.json` inside each archive records the source-to-local index mapping.

## Download another recording or frame range

```bash
python tutorials/05_download_compressed.py --list
python tutorials/05_download_compressed.py --dataset openscope \
  --start 0 --end 128 --output downloads/openscope
```

See the [downloader instructions](../../docs/download.md). Python is needed only for this optional downloader, not for opening the supplied ZIPs in ImageJ.

## Provenance and usage

- OpenScope: [Allen Institute OpenScope](https://openscope.alleninstitute.org/), subject 412933.
- Spinal-cord voltage: data used with [SUPPORT](https://doi.org/10.1038/s41592-023-02005-8).

Original data-provider usage and attribution conditions apply. XERO's MIT software license does not relicense these biological datasets. Small artificial fixtures used by automated codec tests are separate from these user-facing examples.
