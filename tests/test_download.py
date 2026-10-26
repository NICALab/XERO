"""Offline downloader integrity/range tests. No credentials or GPU required."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('downloader',Path(__file__).resolve().parents[1]/'tutorials/05_download_compressed.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)


def fixture():
    files={'model.npz':b'model'}
    m={'profile':'fiarc_integer_experimental_v1','shape':[512,8,9],'restart':256,
       'model':'model.npz','model_sha256':hashlib.sha256(files['model.npz']).hexdigest(),
       'history':'repeat_first_decoded','tolerance':.01,'groups':[]}
    for i in range(8):
        name='group_%06d.bin'%i;files[name]=bytes([i])*20
        m['groups'].append(dict(file=name,start=i*64,frames=64,bytes=20,sha256=hashlib.sha256(files[name]).hexdigest()))
    return m,files


@pytest.mark.parametrize('start,end,stored',[(0,1,(0,64)),(100,200,(0,256)),(256,257,(256,320)),(300,380,(256,384)),(500,512,(256,512))])
def test_restart_context_and_packet_tail(start,end,stored):
    m,_=fixture();subset,a,b=d.plan_range(m,start,end)
    assert (a,b)==stored and subset['shape'][0]==b-a
    assert subset['groups'][0]['start']==0
    assert subset['groups'][0]['sha256']==m['groups'][a//64]['sha256']


@pytest.mark.parametrize('start,end',[(-1,2),(1,1),(0,513),(True,2),(1.5,3)])
def test_invalid_ranges(start,end):
    with pytest.raises(ValueError):d.plan_range(fixture()[0],start,end)


def mock_remote(monkeypatch,bad=False):
    m,files=fixture();calls=[]
    documents={'catalog.json':json.dumps([dict(id='sample',collection='sample',shape=m['shape'],errorBound=.01)]).encode(),
               'archive/sample/manifest.json':json.dumps(m).encode()}
    documents.update({'archive/sample/'+k:v for k,v in files.items()})
    if bad:documents['archive/sample/group_000004.bin']=b'x'*20
    def fetch(url):calls.append(url);return io.BytesIO(documents[url.removeprefix('https://example.test/')])
    monkeypatch.setattr(d,'_open',fetch)
    return calls


def test_download_resume_and_rebased_manifest(tmp_path,monkeypatch):
    calls=mock_remote(monkeypatch)
    receipt=d.download('sample',300,380,tmp_path/'out','https://example.test',progress=None)
    assert receipt['stored_source_range']==[256,384]
    assert receipt['imagej_requested_first_last']==[45,124]
    m=json.loads((tmp_path/'out/manifest.json').read_text())
    assert [g['start'] for g in m['groups']]==[0,64] and m['shape']==[128,8,9]
    assert len(calls)==5
    calls.clear()
    again=d.download('sample',300,380,tmp_path/'out','https://example.test',progress=None)
    assert again['transferred_bytes']==0 and len(calls)==2
    with pytest.raises(FileExistsError):d.download('sample',0,20,tmp_path/'out','https://example.test',progress=None)


def test_corruption_does_not_publish_manifest(tmp_path,monkeypatch):
    mock_remote(monkeypatch,bad=True)
    with pytest.raises(ValueError,match='checksum'):
        d.download('sample',300,380,tmp_path/'out','https://example.test',progress=None)
    assert not (tmp_path/'out/manifest.json').exists()
    assert not list((tmp_path/'out').glob('*.partial'))


def test_unrelated_files_untouched(tmp_path,monkeypatch):
    mock_remote(monkeypatch);(tmp_path/'important').write_bytes(b'keep')
    with pytest.raises(FileExistsError):d.download('sample',0,20,tmp_path,'https://example.test',progress=None)
    assert (tmp_path/'important').read_bytes()==b'keep'


def test_reject_unsafe_paths_and_incomplete_index():
    m,_=fixture();m['groups'][0]['file']='../evil'
    with pytest.raises(ValueError):d.plan_range(m,0,1)
    m,_=fixture();m['groups'].pop()
    with pytest.raises(ValueError):d.plan_range(m,0,1)


def test_ambiguous_collection_rejected():
    with pytest.raises(ValueError):d.select_dataset([dict(id='one',collection='fish'),dict(id='two',collection='fish')],'fish')
