"""Atomic, byte-preserving OME-Zarr export. Never trains or encodes pixels."""
import argparse
import base64
import copy
import json
import math
import os
from pathlib import Path
import tempfile
import time
import zlib

import zarr
from .codec import XeroRestartCodec, canonical, pack_chunk, sha, unpack_header


def repack_manifest(manifest, model, read_packet, destination, *, axis, scale=None, units=None, name=None, resume=False, progress=None):
    """read_packet(group) returns the original packet, including container slices.

    A complete destination is published by one local rename. Failed staging
    directories remain for inspection and never look like a complete image.
    """
    out = Path(destination)
    if out.exists():
        raise FileExistsError(out)
    m = copy.deepcopy(manifest)
    n, h, w = m["shape"]
    k = m["restart"]
    if not all(type(v) is int and v > 0 for v in (n, h, w)):
        raise ValueError("Invalid shape")
    if m["history"] != "repeat_first_decoded" or sha(model) != m["model_sha256"]:
        raise ValueError("Source model/history mismatch")
    codec = XeroRestartCodec(model_zlib_base64=base64.b64encode(zlib.compress(model, 9)).decode(),
        model_sha256=sha(model), profile=m["profile"], restart=k, height=h, width=w, tolerance=m["tolerance"])
    end = 0
    names = set()
    for g in m["groups"]:
        if g["file"] in names or g["start"] != end or not 1 <= g["frames"] <= 64 or end//k != (end+g["frames"]-1)//k:
            raise ValueError("Incomplete/invalid source index")
        if g["bytes"] <= 0:
            raise ValueError("Empty source packet")
        names.add(g["file"]); end += g["frames"]
    if end != n or axis not in ("t", "z"):
        raise ValueError("Incomplete source or unspecified frame axis")
    axes = [dict(name=axis, type="time" if axis=="t" else "space"),
            dict(name="y", type="space"), dict(name="x", type="space")]
    scale = [1., 1., 1.] if scale is None else list(scale)
    if len(scale) != 3 or not all(math.isfinite(v) and v > 0 for v in scale):
        raise ValueError("Provide three positive scales")
    if units is not None:
        if len(units) != 3:
            raise ValueError("Provide three units")
        for a, u in zip(axes, units):
            if u:
                a["unit"] = u
    attributes = dict(ome=dict(version="0.5", multiscales=[dict(name=name or out.stem,
        axes=axes, datasets=[dict(path="0", coordinateTransformations=[dict(type="scale", scale=scale)])])]),
        xero=dict(format="xero.ome_zarr.v1", complete=False, source_payloads_unchanged=True,
        bound_reference="compressor input", physical_calibration="provided" if units else "index coordinates"))
    out.parent.mkdir(parents=True, exist_ok=True)
    candidates = sorted(out.parent.glob(out.name+".incomplete-*")) if resume else []
    stage = candidates[0] if candidates else Path(tempfile.mkdtemp(prefix=out.name+".incomplete-", dir=out.parent))
    started = time.perf_counter()
    if candidates:
        root = zarr.open_group(str(stage), mode="a")
        metadata = json.loads((stage/'0/zarr.json').read_text())
        if metadata['codecs'] != [codec.to_dict()] or metadata['shape'] != [n,h,w]:
            raise ValueError("Resume source/configuration mismatch")
    else:
        root = zarr.create_group(str(stage), zarr_format=3, attributes=attributes)
        root.create_array("0", shape=(n,h,w), chunks=(min(n,k),h,w), dtype="float32",
            serializer=codec, compressors=None, filters=None, fill_value="NaN", dimension_names=(axis,"y","x"))
    chunks = []
    source_bytes = len(model) + len(canonical(manifest))
    by_origin = {}
    for g in m["groups"]:
        by_origin.setdefault(g["start"]//k*k, []).append(g)
    for origin, groups in by_origin.items():
        key = f"0/c/{origin//k}/0/0"
        path = stage/key
        if path.exists():
            raw = path.read_bytes(); header = unpack_header(raw)
            expected = [dict(start=g['start']-origin, frames=g['frames'], bytes=g['bytes'], sha256=g['sha256']) for g in groups]
            if header != dict(origin=origin, frames=sum(g['frames'] for g in groups), groups=expected):
                raise ValueError("Resume chunk/source index mismatch")
            offset = 44+len(canonical(header))
            at = offset
            for g in groups:
                if sha(raw[at:at+g['bytes']]) != g['sha256']:
                    raise ValueError("Resume packet integrity failure")
                at += g['bytes']
            if at != len(raw):
                raise ValueError("Resume chunk length mismatch")
        else:
            raw, offset = pack_chunk([(g, read_packet(g)) for g in groups], origin)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name('0.partial'); tmp.write_bytes(raw); tmp.replace(path)
        for g in groups:
            g["xero_chunk"] = key; g["xero_offset"] = offset
            offset += g["bytes"]; source_bytes += g["bytes"]
        chunks.append(dict(key=key, bytes=len(raw), sha256=sha(raw)))
        if progress:
            progress(len(chunks), len(by_origin), len(raw))
    m["model"] = "embedded in 0/zarr.json"
    attributes["xero"].update(complete=True, archive=m, chunks=chunks)
    root.attrs.update(attributes)
    total = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file())
    receipt = dict(format="xero.ome_zarr.v1", shape=[n,h,w], restart=k,
        source_required_bytes=source_bytes, zarr_bytes=total, chunks=len(chunks),
        repack_seconds=time.perf_counter()-started, retrained=False, reencoded=False)
    (stage/"repack.json").write_bytes(canonical(receipt))
    # No replacement of an existing destination, including concurrent exports.
    if out.exists():
        raise FileExistsError(out)
    os.rename(stage, out)
    return receipt


def repack_archive(source, destination, *, axis, scale=None, units=None):
    source = Path(source).resolve()
    if (source/"INCOMPLETE").exists():
        raise ValueError("Incomplete source archive")
    m = json.loads((source/"manifest.json").read_text())
    def safe(name):
        p = (source/name).resolve()
        if not p.is_relative_to(source) or Path(name).name != name:
            raise ValueError("Unsafe source path")
        return p
    return repack_manifest(m, safe(m["model"]).read_bytes(), lambda g: safe(g["file"]).read_bytes(),
                           destination, axis=axis, scale=scale, units=units)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source");p.add_argument("destination");p.add_argument("--axis",choices=["t","z"],required=True)
    a=p.parse_args();print(json.dumps(repack_archive(a.source,a.destination,axis=a.axis),indent=2))


if __name__ == "__main__":
    main()
