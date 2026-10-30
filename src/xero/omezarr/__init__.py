"""XERO OME-Zarr (optional ``zarr`` extra)."""
from .codec import XeroRestartCodec
from .repack import repack_archive


def open_image(path):
    """Open the single-scale image lazily. Missing chunks raise on access."""
    import zarr
    root = zarr.open_group(str(path), mode="r")
    x = root.attrs.get("xero", {})
    if x.get("format") != "xero.ome_zarr.v1" or x.get("complete") is not True:
        raise ValueError("Not a complete XERO OME-Zarr image")
    return root["0"]


def close_decoders():
    """Release persistent Java/OpenCL resources explicitly (also done at exit)."""
    from .bridge import close
    close()
