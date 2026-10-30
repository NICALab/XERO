"""Native CUDA container access, without decoding or materializing pixels."""
import json
from pathlib import Path
import re
from .codec import unpack_model, sha


class Source:
    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        root = json.loads((self.folder/'zarr.json').read_text())
        x = root.get('attributes', {}).get('xero', {})
        if root.get('zarr_format') != 3 or x.get('format') != 'xero.ome_zarr.v1' or x.get('complete') is not True:
            raise ValueError('Incomplete/unsupported XERO OME-Zarr')
        self.manifest = x['archive']
        a = json.loads((self.folder/'0/zarr.json').read_text())
        if a.get('shape') != self.manifest['shape'] or a.get('data_type') != 'float32' or len(a['codecs']) != 1 or a['codecs'][0]['name'] != 'xero.restart_v1':
            raise ValueError('Unsupported OME-Zarr array/codec')
        self.config = a['codecs'][0]['configuration']
        for key in ['profile','restart','tolerance','model_sha256']:
            if self.config[key] != self.manifest[key]:
                raise ValueError('OME-Zarr model/index mismatch: '+key)
        self.model = unpack_model(self.config['model_zlib_base64'], self.config['model_sha256'])

    def packet(self, group):
        key = group['xero_chunk']
        if not re.fullmatch(r'0/c/[0-9]+/0/0', key):
            raise ValueError('Unsafe chunk key')
        p = (self.folder/key).resolve()
        if not p.is_relative_to(self.folder):
            raise ValueError('Chunk outside archive')
        start, size = group['xero_offset'], group['bytes']
        if type(start) is not int or type(size) is not int or start < 44 or not 0 < size <= 512*1024**2 or start+size > p.stat().st_size:
            raise ValueError('Invalid/truncated packet extent')
        with p.open('rb') as f:
            f.seek(start);data=f.read(size)
        if len(data) != size or sha(data) != group['sha256']:
            raise ValueError('Packet integrity failure')
        return data
