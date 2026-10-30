# Download compressed data

Use [`tutorials/05_download_compressed.py`](../tutorials/05_download_compressed.py) to download OME-Zarr recordings and frame ranges. Requires only the Python standard library; no pixels are decompressed during download.

## Choose one dataset string

```bash
# Live list: one ID per recording, channel/plane and error bound.
python tutorials/05_download_compressed.py --list
python tutorials/05_download_compressed.py --list --filter glutamate
```

Use an ID from the first column. Collection names also work when they identify a single recording and bound. Two aliases are provided:

| String | Recording | Bound |
|---|---|---:|
| `openscope` (or `openscope496`) | OpenScope dendritic calcium | 1% |
| `spinal-cord-voltage` | Zebrafish spinal-cord voltage | 0.5% |

```bash
python tutorials/05_download_compressed.py --dataset openscope \
  --start 0 --end 128 --output downloads/openscope.ome.zarr
```

Each selection is a single `[frames, height, width]` stream. Channels and planes are listed separately. The catalog's `axis` field indicates whether frames correspond to time or z.

## Index convention and I-frame context

`--start` is zero-based and included; `--end` is zero-based and excluded. For example, `--start 100 --end 200` requests source frames 100–199.

The download includes complete restart chunks. For OpenScope, requesting `[100, 200)` stores `[0, 256)`, allowing the selected frames to decode independently of the server.

The `.ome.zarr` folder contains array metadata, compressed chunks and `download.json`. The last file records the requested source range, actually stored range, zero-based local range, and one-based inclusive ImageJ range. In that example, the requested images are local ImageJ frames **101–200**. When the source start is a later I-frame, local frame 1 corresponds to that I-frame rather than source frame 0.

Open the downloaded folder in [ImageJ](../imagej/README.md), [napari](../napari/README.md) or [Python](python.md).

## Call from Python

```python
from runpy import run_path

api = run_path("tutorials/05_download_compressed.py")
datasets = list(api["list_datasets"]())
result = api["download_ome_zarr"](
    dataset="openscope",
    start=0,
    end=128,
    output="downloads/openscope.ome.zarr",
)
print(result["stored_source_range"])
```

## Interrupted downloads

Files are checked against their SHA-256 hashes before final metadata is written.
Use a new empty output directory. Interrupted OME-Zarr subset downloads do not
yet support resuming; retry into another empty folder.
