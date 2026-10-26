# Download compressed data, without decompression

[`tutorials/05_download_compressed.py`](../tutorials/05_download_compressed.py) is a standalone Python standard-library script. It works without installing XERO, PyTorch, CUDA or an S3 client. No account/API key is required for the public viewer data. Downloading public data does not grant access to the private R2 bucket or permission to upload.

## Choose one dataset string

```bash
# Live list: one ID per recording, channel/plane and error bound.
python tutorials/05_download_compressed.py --list
python tutorials/05_download_compressed.py --list --filter glutamate
```

Use an exact ID from the first column. A collection name is accepted only if it has one recording/bound; ambiguous names fail instead of silently choosing a channel or bound. Two convenient aliases are provided:

| String | Recording | Bound |
|---|---|---:|
| `openscope` (or `openscope496`) | OpenScope dendritic calcium | 1% |
| `spinal-cord-voltage` | Zebrafish spinal-cord voltage | 0.5% |

```bash
python tutorials/05_download_compressed.py --dataset openscope \
  --start 0 --end 128 --output downloads/openscope
```

The catalog includes all currently connected recordings, including multiple planes/regions where relevant. Each selection is a single `[frames, height, width]` stream, not an automatically combined 3D or multichannel recording. The frame axis can be time or z, as indicated by the catalog's `axis` field.

## Index convention and I-frame context

`--start` is zero-based and included; `--end` is zero-based and excluded. For example, `--start 100 --end 200` requests source frames 100–199.

Compressed packets cannot be cut at arbitrary frame boundaries without decoding/re-encoding. The script downloads the preceding I-frame context and the complete last packet. For OpenScope, a request for `[100, 200)` stores `[0, 256)`. It writes a **locally rebased, decodable manifest** while keeping all model and packet bytes unchanged.

The resulting folder contains `manifest.json`, `model.npz`, `group_*.bin` and `download.json`. The last file records the requested source range, actually stored range, zero-based local range, and one-based inclusive ImageJ range. In that example, the requested images are local ImageJ frames **101–200**. When the source start is a later I-frame, local frame 1 corresponds to that I-frame rather than source frame 0.

Open the folder in **Plugins → Compression (XERO) → Open compressed image**. The Python script itself never decompresses images or writes TIFF/HDF5 output.

## Call from Python

```python
from runpy import run_path

api = run_path("tutorials/05_download_compressed.py")
datasets = list(api["list_datasets"]())
result = api["download"](
    dataset="openscope",
    start=0,
    end=128,
    output="downloads/openscope",
)
print(result["stored_source_range"])
```

## Integrity and interrupted downloads

Every model/packet is checked against its SHA-256. Re-running the same command verifies and reuses completed files. An interrupted file is discarded and that file is downloaded again; byte-level continuation within a file is not implemented. Different datasets/ranges or conflicting files are rejected rather than overwritten. The final `manifest.json` is published only after all required files verify.

Only the catalog, selected manifest, one model and necessary packets are requested. No decompressed image cache is created. Downloads still consume network/storage resources and the server's request allowance. Use a short range first; downloading the whole catalog is not automatic.
