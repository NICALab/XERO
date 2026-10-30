"""Write independently verified restart packets directly to OME-Zarr chunks."""
import base64
import copy
import hashlib
import os
from pathlib import Path
import struct
import tempfile
import zlib

import zarr
from .codec import XeroRestartCodec, MAGIC, MAX_CHUNK, canonical, sha


class RestartWriter:
    """One restart at a time; retain only the packet index between writes.

    The encoder must audit reconstructed pixels before calling append().
    Interrupted outputs remain in a distinct, incomplete staging directory.
    """
    def __init__(self, destination, manifest, model, *, axis):
        self.output = Path(destination)
        if self.output.exists() or self.output.is_symlink():
            raise FileExistsError(self.output)
        if axis not in ('t', 'z'):
            raise ValueError('Specify axis t (movie) or z (structural stack)')
        self.manifest = copy.deepcopy(manifest)
        m = self.manifest
        n, h, w = m['shape']
        if any(type(v) is not int or v < 1 for v in (n, h, w)):
            raise ValueError('Invalid shape')
        if m['groups'] or m['history'] != 'repeat_first_decoded' or sha(model) != m['model_sha256']:
            raise ValueError('Invalid initial model/index')
        codec = XeroRestartCodec(model_zlib_base64=base64.b64encode(zlib.compress(model, 9)).decode(),
            model_sha256=sha(model), profile=m['profile'], restart=m['restart'],
            height=h, width=w, tolerance=m['tolerance'])
        m['model'] = 'embedded in 0/zarr.json'
        self.axis = axis
        self.position = 0
        self.chunks = []
        self.attributes = dict(ome=dict(version='0.5', multiscales=[dict(
            name=self.output.name.removesuffix('.ome.zarr'),
            axes=[dict(name=axis, type='time' if axis == 't' else 'space'),
                  dict(name='y', type='space'), dict(name='x', type='space')],
            datasets=[dict(path='0', coordinateTransformations=[dict(type='scale', scale=[1.,1.,1.])])])]),
            xero=dict(format='xero.ome_zarr.v1', complete=False,
                      bound_reference='compressor input', physical_calibration='index coordinates'))
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.stage = Path(tempfile.mkdtemp(prefix=self.output.name+'.incomplete-', dir=self.output.parent))
        self.root = zarr.create_group(str(self.stage), zarr_format=3, attributes=self.attributes)
        self.root.create_array('0', shape=(n,h,w), chunks=(min(n,m['restart']),h,w), dtype='float32',
            serializer=codec, compressors=None, filters=None, fill_value='NaN', dimension_names=(axis,'y','x'))

    def append(self, origin, packets):
        """Accept (index, bytes-like payload) pairs after independent auditing."""
        m = self.manifest
        n, h, w = m['shape']; k = m['restart']
        if origin != self.position or origin % k or origin >= n:
            raise ValueError('Restart is out of order')
        end = origin
        headers = []
        names = {g['file'] for g in m['groups']}
        for g, payload in packets:
            if (g['start'] != end or type(g['frames']) is not int or not 1 <= g['frames'] <= 64
                    or g['file'] in names or Path(g['file']).name != g['file']
                    or len(payload) != g['bytes'] or sha(payload) != g['sha256']):
                raise ValueError('Invalid verified packet')
            if len(payload) < 20 or struct.unpack_from('<8sIII', payload) != (b'FSEEDSG1', g['frames'], h, w):
                raise ValueError('Packet header mismatch')
            names.add(g['file'])
            headers.append(dict(start=end-origin, frames=g['frames'], bytes=g['bytes'], sha256=g['sha256']))
            end += g['frames']
        if end != min(n, origin+k):
            raise ValueError('Incomplete restart')
        header = canonical(dict(origin=origin, frames=end-origin, groups=headers))
        prefix = struct.pack('<I', len(header))
        size = 44 + len(header) + sum(len(payload) for _, payload in packets)
        if len(header) > 1024**2 or size > MAX_CHUNK:
            raise ValueError('Restart exceeds the reader limit; choose a shorter restart interval')
        # Hash/write the existing buffers in pieces, with no concatenated copy
        # of the compressed restart and no intermediate group_*.bin files.
        body = hashlib.sha256(prefix + header)
        for _, payload in packets:
            body.update(payload)
        key = f'0/c/{origin//k}/0/0'
        path = self.stage/key; path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with path.with_suffix('.partial').open('xb') as f:
            for part in (MAGIC, body.digest(), prefix, header):
                f.write(part); digest.update(part)
            for _, payload in packets:
                f.write(payload); digest.update(payload)
        path.with_suffix('.partial').replace(path)
        offset = 44 + len(header)
        for g, payload in packets:
            m['groups'].append(dict(g, xero_chunk=key, xero_offset=offset))
            offset += len(payload)
        self.chunks.append(dict(key=key, bytes=size, sha256=digest.hexdigest()))
        self.position = end

    def finish(self, report):
        if self.position != self.manifest['shape'][0]:
            raise ValueError('Cannot publish an incomplete image')
        self.attributes['xero'].update(complete=True, archive=self.manifest, chunks=self.chunks)
        self.root.attrs.update(self.attributes)
        total = sum(p.stat().st_size for p in self.stage.rglob('*') if p.is_file())
        result = dict(report, format='xero.ome_zarr.v1', axis=self.axis,
            compressed_bytes=total, compression_ratio=report['source_bytes']/total,
            uint16_equivalent_ratio=report['float32_bytes']/2/total,
            byte_accounting='OME-Zarr metadata and chunks; excludes encode_report.json and filesystem overhead')
        (self.stage/'encode_report.json').write_bytes(canonical(result))
        if self.output.exists() or self.output.is_symlink():
            raise FileExistsError(self.output)
        os.rename(self.stage, self.output)
        return result
