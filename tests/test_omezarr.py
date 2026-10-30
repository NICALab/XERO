import json
from pathlib import Path
import zipfile
import numpy as np
import pytest

zarr = pytest.importorskip('zarr')
from xero.omezarr import repack_archive, open_image
from xero.omezarr.codec import unpack_header
from xero.omezarr.source import Source


@pytest.fixture
def pair(tmp_path):
    fixture=Path(__file__).parent/'fixtures/XERO_synthetic.zip'
    with zipfile.ZipFile(fixture) as z:
        z.extractall(tmp_path)
    native=tmp_path/'XERO_demo';ome=tmp_path/'image.ome.zarr'
    repack_archive(native,ome,axis='t')
    return native,ome


def test_packets_and_model_unchanged(pair):
    native,ome=pair;s=Source(ome);m=json.loads((native/'manifest.json').read_text())
    assert s.model==(native/'model.npz').read_bytes()
    assert s.manifest['shape']==m['shape']
    for g in s.manifest['groups']:
        assert s.packet(g)==(native/g['file']).read_bytes()
    a=open_image(ome)
    assert a.shape==(160,64,64) and a.chunks==(128,64,64)
    assert a.metadata.dimension_names==('t','y','x')
    assert np.isnan(a.fill_value)
    meta=json.loads((ome/'zarr.json').read_text())
    assert meta['attributes']['ome']['version']=='0.5'
    assert meta['attributes']['xero']['complete']


def test_missing_is_not_fill(pair):
    _,ome=pair
    (ome/'0/c/0/0/0').rename(ome/'missing')
    with pytest.raises(FileNotFoundError,match='Missing XERO'):
        zarr.open(str(ome/'0'),mode='r')[0:1,0:1,0:1]


def test_corruption_rejected_before_decoder(pair):
    _,ome=pair;p=ome/'0/c/0/0/0';data=bytearray(p.read_bytes());data[-1]^=1;p.write_bytes(data)
    with pytest.raises(ValueError,match='integrity'):
        open_image(ome)[0:1,0:1,0:1]


def test_no_accidental_reencoding_or_overwrite(pair):
    native,ome=pair
    with pytest.raises(FileExistsError):repack_archive(native,ome,axis='t')
    with pytest.raises(NotImplementedError):open_image(ome).metadata.codecs[0].compute_encoded_size(1,None)


def test_chunk_framing_and_tail(pair):
    _,ome=pair
    h=unpack_header((ome/'0/c/1/0/0').read_bytes())
    assert h['origin']==128 and h['frames']==32
    assert h['groups'][0]['start']==0


def test_napari_declines_unrelated_directories(tmp_path):
    from xero.napari_reader import get_reader
    assert get_reader(str(tmp_path)) is None


def test_download_rebases_restart_without_changing_packets(pair,tmp_path,monkeypatch):
    import importlib.util
    import io
    module=Path(__file__).resolve().parents[1]/'tutorials/05_download_compressed.py'
    spec=importlib.util.spec_from_file_location('xero_download_ome_test',module)
    d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
    native,ome=pair
    def remote(url):
        key=url.removeprefix('https://example.test/')
        if key=='catalog.json':return io.BytesIO(json.dumps([dict(id='test',omeZarr='/omezarr/test.ome.zarr/')]).encode())
        return io.BytesIO((ome/key.removeprefix('omezarr/test.ome.zarr/')).read_bytes())
    monkeypatch.setattr(d,'_open',remote)
    out=tmp_path/'download.ome.zarr'
    receipt=d.download_ome_zarr('test',129,134,out,'https://example.test',progress=None)
    assert receipt['stored_source_range']==[128,160]
    assert receipt['requested_local_range']==[1,6]
    s=Source(out);assert s.manifest['shape']==[32,64,64]
    assert s.packet(s.manifest['groups'][0])==(native/'group_000002.bin').read_bytes()
    assert unpack_header((out/'0/c/0/0/0').read_bytes())['origin']==0
    assert open_image(out).shape==(32,64,64)
