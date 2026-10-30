"""Package only decoder inputs in a small, directly openable ImageJ example."""
import argparse
import json
from pathlib import Path
import zipfile
from xero.archive import manifest, digest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();m=manifest(a.archive)
    if (a.archive/'zarr.json').exists():
        from xero.omezarr.source import Source
        src=Source(a.archive)
        for g in m['groups']:src.packet(g)
        chunks=json.loads((a.archive/'zarr.json').read_text())['attributes']['xero']['chunks']
        names=['zarr.json','0/zarr.json']
        for c in chunks:
            path=(a.archive/c['key']).resolve()
            if not path.is_relative_to(a.archive.resolve()) or digest(path)!=c['sha256']:
                raise ValueError('Chunk integrity failure')
            names.append(c['key'])
        if (a.archive/'download.json').exists():names.append('download.json')
        folder=a.archive.name if a.archive.name.endswith('.ome.zarr') else a.archive.name+'.ome.zarr'
    else:
        names=['manifest.json',m['model']]+[g['file'] for g in m['groups']]
        for g in m['groups']:
            if digest(a.archive/g['file'])!=g['sha256']:
                raise ValueError('Packet integrity failure')
        folder='XERO_demo'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.output,'x',zipfile.ZIP_DEFLATED) as z:
        for name in names:
            # Fixed timestamps and permissions for reproducible packaging.
            entry=zipfile.ZipInfo(folder+'/'+name,(2026,1,1,0,0,0))
            entry.compress_type=zipfile.ZIP_DEFLATED
            entry.external_attr=0o100644<<16
            z.writestr(entry,(a.archive/name).read_bytes())
    print(f'{a.output}: {a.output.stat().st_size} bytes, SHA-256 {digest(a.output)}')


if __name__=='__main__':main()
