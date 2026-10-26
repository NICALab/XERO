# Try biological imaging data

Two short excerpts from the compressed datasets served by the XERO viewer:

| File | Imaging modality | Dimensions (width × height × frames) | Bound |
|---|---|---|---:|
| [XERO_OpenScope.zip](XERO_OpenScope.zip) | Mouse dendritic calcium, OpenScope sub-412933 | 496 × 496 × 128 | 1% |
| [XERO_Spinal_Cord.zip](XERO_Spinal_Cord.zip) | Zebrafish spinal-cord voltage | 1024 × 148 × 128 | 0.5% |

Open the file page and click **Download raw file** (download icon), then unzip it. If the repository is private, GitHub requires an authorized login.

In Fiji/ImageJ choose **Plugins → Compression (XERO) → Open compressed image**, select the extracted `XERO_OpenScope` or `XERO_Spinal_Cord` folder, and open frames **1–128**. Use **Auto** or **GPU / OpenCL** when available. CPU also works, but is slower. Selected frames are loaded into RAM: approximately 126 MB and 78 MB of float32 pixels respectively, plus decoder workspace.

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
