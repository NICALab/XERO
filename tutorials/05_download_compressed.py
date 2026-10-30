#!/usr/bin/env python3
"""Download a decodable XERO frame range. Python standard library only.

No decompression, training, GPU, Cloudflare account or API key is required.
Indices are zero-based, end-exclusive. The stored range includes the preceding
I-frame and complete entropy packets. See download.json for the exact mapping.
"""
import argparse
import copy
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import tempfile
import time
import zlib
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_SERVER = "https://xero-viewer.pages.dev"
USER_AGENT = "XERO-downloader/1.0"
ALIASES = {"openscope": "openscope496", "spinal-cord-voltage": "full-0163f6d27114c850e101"}
LABELS = {"Ahrens neuron": "Neuronal calcium", "cohen_voltron2": "Dendritic voltage",
          "zarchon": "Spinal-cord voltage", "Zeguan private": "Light-sheet voltage"}


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or ".." in value:
        raise ValueError("Unsafe archive identifier or filename")
    return value


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))
    return value


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("Invalid SHA-256 in manifest")
    return value


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _open(url):
    # An explicit User-Agent also avoids generic urllib clients being blocked
    # by hosting-layer bot rules. No cookies or credentials are sent.
    for attempt in range(3):
        try:
            return urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=60)
        except HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError("HTTP %s fetching %s" % (e.code, url)) from e
        except (URLError, TimeoutError) as e:
            if attempt == 2:
                raise RuntimeError("Cannot fetch " + url) from e
        time.sleep(2 ** attempt)


def _json(url):
    with _open(url) as response:
        data = response.read(16 * 1024 * 1024 + 1)
    if len(data) > 16 * 1024 * 1024:
        raise ValueError("Oversized catalog or manifest")
    return json.loads(data)


def list_datasets(server=DEFAULT_SERVER):
    """Return one entry per downloadable recording/channel/plane/bound."""
    rows = _json(server.rstrip("/") + "/catalog.json")
    if not isinstance(rows, list):
        raise ValueError("Invalid dataset catalog")
    for row in rows:
        _name(row["id"])
        if row.get("collection") in LABELS:
            row = row.copy()
            row["collectionTitle"] = LABELS[row["collection"]]
        yield row


def select_dataset(rows, dataset):
    key = ALIASES.get(dataset, dataset)
    exact = [x for x in rows if x["id"] == key]
    choices = exact or [x for x in rows if x.get("collection") == key]
    if len(choices) != 1:
        raise ValueError("Dataset is unknown or has multiple recordings/bounds. Use --list and select one exact ID.")
    return choices[0]


def plan_range(manifest, start, end):
    """Rebase metadata only. Packet and model bytes remain unchanged."""
    _integer(start, "start")
    _integer(end, "end", 1)
    if manifest.get("profile") not in ("fiarc_integer_experimental_v1", "fiarc_integer_boundscale_v2"):
        raise ValueError("Unsupported archive profile")
    shape = manifest["shape"]
    if len(shape) != 3:
        raise ValueError("Expected a [frames, height, width] stream")
    for value in shape:
        _integer(value, "shape", 1)
    restart = _integer(manifest["restart"], "restart", 128)
    if restart & (restart - 1) or not 0 <= start < end <= shape[0]:
        raise ValueError("Require 0 <= start < end <= %d and a valid restart interval" % shape[0])
    if manifest.get("history") != "repeat_first_decoded":
        raise ValueError("Unsupported initial history")
    model = _name(manifest["model"])
    _sha(manifest["model_sha256"])
    seen = {"manifest.json", "download.json", model}
    if len(seen) != 3:
        raise ValueError("Reserved model filename")
    position = 0
    groups = manifest["groups"]
    for g in groups:
        _name(g["file"]); _sha(g["sha256"])
        n = _integer(g["frames"], "packet frames", 1)
        _integer(g["bytes"], "packet bytes", 1)
        if g["file"] in seen or g["start"] != position or n > 64 or position % restart + n > restart:
            raise ValueError("Invalid packet index")
        seen.add(g["file"])
        position += n
    if position != shape[0]:
        raise ValueError("Incomplete source index")
    first = start // restart * restart
    selected = [g for g in groups if first <= g["start"] < end]
    if not selected or selected[0]["start"] != first:
        raise ValueError("Requested restart is not available")
    stop = selected[-1]["start"] + selected[-1]["frames"]
    result = {k: manifest[k] for k in ("profile", "restart", "model", "model_sha256", "history", "tolerance")}
    result["shape"] = [stop - first, *shape[1:]]
    result["groups"] = [dict(g, start=g["start"] - first) for g in selected]
    return result, first, stop


def _write_json(path, value):
    # Only call for the downloader's known receipt/manifest paths.
    fd, temp = tempfile.mkstemp(prefix=".xero-", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf8") as f:
            json.dump(value, f, indent=2); f.write("\n")
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _download(url, path, sha, size=None):
    if path.is_symlink():
        raise ValueError("Refusing symlink: " + str(path))
    if path.exists():
        if (size is None or path.stat().st_size == size) and _hash(path) == sha:
            return 0
        raise FileExistsError("Existing file has a different checksum; use a new output folder: " + str(path))
    fd, temp = tempfile.mkstemp(prefix=".xero-", suffix=".partial", dir=path.parent)
    try:
        h = hashlib.sha256(); count = 0
        with os.fdopen(fd, "wb") as f, _open(url) as response:
            while True:
                b = response.read(1024 * 1024)
                if not b:
                    break
                count += len(b)
                if count > (size if size is not None else 64 * 1024 * 1024):
                    raise ValueError("Response exceeds expected file size")
                h.update(b); f.write(b)
        if (size is not None and count != size) or h.hexdigest() != sha:
            raise ValueError("Downloaded file checksum/length mismatch: " + path.name)
        os.replace(temp, path)
        return count
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def download(dataset, start, end, output, server=DEFAULT_SERVER, progress=print):
    """Download compressed bytes and metadata; return requested/local ranges.

    dataset: exact ID from list_datasets(), or an unambiguous collection/alias.
    start/end: original stream indices, zero-based and end-exclusive.
    output: new directory, or the same directory for a verified resume.
    """
    row = select_dataset(list(list_datasets(server)), dataset)
    base = server.rstrip("/") + "/archive/" + _name(row["id"]) + "/"
    original = _json(base + "manifest.json")
    subset, first, stop = plan_range(original, start, end)
    plan = {"dataset": row["id"], "server": server.rstrip("/"),
            "source_shape_tyx": original["shape"], "requested_source_range": [start, end],
            "stored_source_range": [first, stop], "requested_local_range": [start-first, end-first],
            "imagej_requested_first_last": [start-first+1, end-first],
            "indices": "zero-based, end-exclusive; ImageJ indices are one-based, inclusive",
            "source_manifest_sha256": hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()}
    output = Path(output)
    if output.is_symlink():
        raise ValueError("Refusing a symlink output folder")
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / "download.json"
    if receipt.is_symlink() or (output / "manifest.json").is_symlink():
        raise ValueError("Refusing symlink metadata")
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if any(previous.get(k) != v for k, v in plan.items()):
            raise FileExistsError("This folder belongs to another download; choose a new output folder")
    elif any(output.iterdir()):
        raise FileExistsError("Output folder is not empty; choose a new output folder")
    _write_json(receipt, dict(plan, complete=False))
    if progress:
        progress("Requested [%d, %d); storing [%d, %d) including restart context and packet tails." % (start, end, first, stop))
    transferred = 0
    files = [(subset["model"], subset["model_sha256"], None)]
    files += [(g["file"], g["sha256"], g["bytes"]) for g in subset["groups"]]
    for i, (name, sha, size) in enumerate(files, 1):
        n = _download(base+name, output/name, sha, size)
        transferred += n
        if progress:
            progress("%d/%d %s: %s" % (i, len(files), name, ("%.2f MB" % (n/1e6)) if n else "verified; already downloaded"))
    _write_json(output / "manifest.json", subset)
    result = dict(plan, complete=True, transferred_bytes=transferred,
                  archive_bytes=sum((output/n).stat().st_size for n, _, _ in files)+(output/'manifest.json').stat().st_size)
    _write_json(receipt, result)
    if progress:
        progress("Ready: %s. No decompression performed. ImageJ requested frames: %d-%d." % (output, *plan["imagej_requested_first_last"]))
    return result


def download_ome_zarr(dataset, start, end, output, server=DEFAULT_SERVER, progress=print):
    """Download complete restart chunks and rebase their metadata, not pixels."""
    row=select_dataset(list(list_datasets(server)),dataset)
    root=row.get('omeZarr','')
    if not re.fullmatch(r'/omezarr/[\w-]+\.ome\.zarr/',root):
        raise ValueError('OME-Zarr is not yet published for this recording')
    base=server.rstrip('/')+root
    group=_json(base+'zarr.json');array=_json(base+'0/zarr.json')
    x=group.get('attributes',{}).get('xero',{})
    if x.get('format')!='xero.ome_zarr.v1' or x.get('complete') is not True:
        raise ValueError('Incomplete/unsupported XERO OME-Zarr')
    m=x['archive'];n,h,w=m['shape'];k=_integer(m['restart'],'restart',128)
    _integer(start,'start');_integer(end,'end',1)
    if k&(k-1) or not 0<=start<end<=n:raise ValueError('Invalid requested range')
    if array.get('shape')!=m['shape'] or array.get('data_type')!='float32' or len(array['codecs'])!=1 or array['codecs'][0]['name']!='xero.restart_v1':
        raise ValueError('Unsupported array/codec')
    cfg=array['codecs'][0]['configuration']
    for key in ['profile','restart','tolerance','model_sha256']:
        if cfg[key]!=m[key]:raise ValueError('Model/index mismatch')
    encoded=cfg['model_zlib_base64']
    if len(encoded)>32*1024**2:raise ValueError('Oversized model')
    dec=zlib.decompressobj();model=dec.decompress(base64.b64decode(encoded,validate=True),16*1024**2+1)
    if len(model)>16*1024**2 or not dec.eof or dec.unused_data or hashlib.sha256(model).hexdigest()!=_sha(m['model_sha256']):
        raise ValueError('Model integrity failure')
    first=start//k*k;stop=min(n,((end+k-1)//k)*k)
    plan=dict(dataset=row['id'],server=server,format='xero.ome_zarr.v1',requested_source_range=[start,end],stored_source_range=[first,stop],
        requested_local_range=[start-first,end-first],imagej_requested_first_last=[start-first+1,end-first],source_model_sha256=m['model_sha256'])
    output=Path(output)
    if output.is_symlink():raise ValueError('Symlink output is not allowed')
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise FileExistsError('Choose a new empty OME-Zarr output folder')
    _write_json(output/'download.json',dict(plan,complete=False))
    chunks={c['key']:c for c in x['chunks']};index=[];receipts=[];transferred=0
    canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':')).encode()
    for origin in range(first,stop,k):
        key=f'0/c/{origin//k}/0/0';c=chunks[key]
        size=_integer(c['bytes'],'chunk bytes',44)
        if size>512*1024**2:raise ValueError('Oversized restart')
        incoming=output/'incoming.partial'
        transferred+=_download(base+key,incoming,_sha(c['sha256']),size)
        raw=incoming.read_bytes()
        if raw[:8]!=b'XEROZ001' or hashlib.sha256(raw[40:]).digest()!=raw[8:40]:raise ValueError('Chunk integrity failure')
        hs,=struct.unpack_from('<I',raw,40)
        if not 0<hs<=1024**2 or 44+hs>len(raw):raise ValueError('Invalid chunk header')
        header=json.loads(raw[44:44+hs]);groups=[g for g in m['groups'] if origin<=g['start']<min(n,origin+k)]
        expected=dict(origin=origin,frames=sum(g['frames'] for g in groups),groups=[{a:(g[a]-origin if a=='start' else g[a]) for a in ['start','frames','bytes','sha256']} for g in groups])
        if header!=expected:raise ValueError('Chunk/index mismatch')
        at=44+hs
        for g in groups:
            if hashlib.sha256(raw[at:at+g['bytes']]).hexdigest()!=_sha(g['sha256']):raise ValueError('Packet integrity failure')
            at+=g['bytes']
        if at!=len(raw):raise ValueError('Trailing chunk bytes')
        header['origin']=origin-first;head=canonical(header);body=struct.pack('<I',len(head))+head+raw[44+hs:]
        rebased=b'XEROZ001'+hashlib.sha256(body).digest()+body
        newkey=f'0/c/{(origin-first)//k}/0/0';p=output/newkey;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(rebased)
        at=44+len(head)
        for g in groups:
            index.append(dict(g,start=g['start']-first,xero_chunk=newkey,xero_offset=at));at+=g['bytes']
        receipts.append(dict(key=newkey,bytes=len(rebased),sha256=hashlib.sha256(rebased).hexdigest()))
        incoming.unlink()
        if progress:progress(f'Restart {origin//k}: verified; compressed payload unchanged')
    group=copy.deepcopy(group);array=copy.deepcopy(array);x=group['attributes']['xero']
    x['archive']=dict(m,shape=[stop-first,h,w],groups=index);x['chunks']=receipts
    array['shape']=[stop-first,h,w];array['chunk_grid']['configuration']['chunk_shape']=[min(k,stop-first),h,w]
    transforms=group['attributes']['ome']['multiscales'][0]['datasets'][0]['coordinateTransformations']
    scale=next(v['scale'] for v in transforms if v['type']=='scale')
    if first:
        translation=next((v for v in transforms if v['type']=='translation'),None)
        if translation is None:transforms.append(dict(type='translation',translation=[first*scale[0],0,0]))
        else:translation['translation'][0]+=first*scale[0]
    _write_json(output/'0/zarr.json',array);_write_json(output/'zarr.json',group)
    result=dict(plan,complete=True,transferred_bytes=transferred)
    _write_json(output/'download.json',result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list", action="store_true", help="List all recording/channel/plane/bound IDs")
    p.add_argument("--filter", default="", help="Filter the --list output")
    p.add_argument("--dataset", help="Dataset ID, unambiguous collection, openscope or spinal-cord-voltage")
    p.add_argument("--start", type=int, help="Original first frame, zero-based (included)")
    p.add_argument("--end", type=int, help="Original last boundary, zero-based (excluded)")
    p.add_argument("--output", type=Path)
    p.add_argument("--server", default=DEFAULT_SERVER)
    p.add_argument("--format", choices=['native','ome-zarr'], default='native')
    a = p.parse_args()
    try:
        if a.list:
            print("ID\tDataset\tRegion\tBound (%)\tFrames x height x width")
            for x in list_datasets(a.server):
                fields=[x['id'], x.get('collectionTitle', x.get('collection','')), x.get('region',''),
                        "%g" % (100*x['errorBound']), " x ".join(map(str,x['shape']))]
                if a.filter.lower() in " ".join(fields).lower():
                    print("\t".join(fields))
            return
        if a.dataset is None or a.start is None or a.end is None or a.output is None:
            p.error("Use --list, or supply --dataset, --start, --end and --output")
        (download_ome_zarr if a.format=='ome-zarr' else download)(a.dataset, a.start, a.end, a.output, a.server)
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        p.exit(1, "Download failed: %s\n" % e)


if __name__ == "__main__":
    main()
