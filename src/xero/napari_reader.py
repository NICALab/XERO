"""napari reader: lazy per-frame tasks, no whole-volume decompression cache."""
import json
from pathlib import Path


def get_reader(path):
    if isinstance(path, list):
        if len(path) != 1:
            return None
        path = path[0]
    if str(path).startswith(("http://", "https://")):
        return read_image if str(path).rstrip("/").endswith(".zarr") else None
    try:
        meta = json.loads((Path(path)/"zarr.json").read_text())
    except (OSError, ValueError):
        return None
    return read_image if meta.get("attributes", {}).get("xero", {}).get("format") == "xero.ome_zarr.v1" else None


def read_image(path):
    import dask.array as da
    import zarr
    from .omezarr import open_image
    if isinstance(path, list):
        path = path[0]
    image = open_image(path)
    root = zarr.open_group(str(path), mode="r")
    multiscale = root.attrs["ome"]["multiscales"][0]
    # A 2048² frame is 16 MiB; never request an entire large restart from Zarr.
    data = da.from_array(image, chunks=(1, image.shape[1], image.shape[2]),
                         asarray=False, fancy=False, name=False)
    scale = multiscale["datasets"][0]["coordinateTransformations"][0]["scale"]
    translate = next((v['translation'] for v in multiscale['datasets'][0]['coordinateTransformations'] if v['type']=='translation'), [0,0,0])
    return [(data, dict(name=multiscale.get("name", "XERO"), scale=scale, translate=translate,
                        metadata=dict(axes=multiscale["axes"], xero=root.attrs["xero"]["format"])), "image")]
