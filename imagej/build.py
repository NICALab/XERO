"""Build XERO_ImageJ.jar. Dependencies are explicit; no machine-specific paths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--deps',type=Path,required=True,help='Directory with gson.jar, jocl.jar, aircompressor.jar')
    p.add_argument('--ij',type=Path,required=True,help='ImageJ ij.jar (build-time only)')
    p.add_argument('--ecj',type=Path,help='Eclipse compiler JAR; otherwise use javac')
    p.add_argument('--out',type=Path,default=Path('imagej/build'))
    a=p.parse_args();here=Path(__file__).resolve().parent
    a.out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='xero-classes-',dir=a.out) as temp:
        build(a,here,Path(temp))


def build(a,here,classes):
    jars=[a.deps/(name+'.jar') for name in ('gson','jocl','aircompressor')]
    cp=os.pathsep.join(map(str,[a.ij,*jars]));sources=sorted((here/'src').glob('*.java'))
    compiler=['java','-jar',str(a.ecj),'-8'] if a.ecj else ['javac','--release','8']
    subprocess.run([*compiler,'-cp',cp,'-d',str(classes),*map(str,sources)],check=True)
    out=a.out/'XERO_ImageJ.jar';seen=set()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(classes.rglob('*.class')):
            name=str(path.relative_to(classes));z.write(path,name);seen.add(name)
        for jar in jars:
            with zipfile.ZipFile(jar) as dep:
                for name in dep.namelist():
                    if name.endswith('/') or name in seen or name=='module-info.class' or name.startswith('META-INF/versions/') or name=='META-INF/MANIFEST.MF' or name.upper().endswith(('.SF','.RSA','.DSA')):continue
                    z.writestr(name,dep.read(name));seen.add(name)
        z.writestr('plugins.config','Plugins>XERO, "Open compressed image...", org.fiarc.strictjava.FiARC_Standalone_Open\nPlugins>XERO, "Check CPU or GPU compatibility...", org.fiarc.strictjava.FiARC_Java_Check\n')
        for name in ('resident.cl','portable.cl','entropy.cl'):z.write(here/'src'/name,'fiarc/'+name)
        scalar=(here/'src/float32_int.c').read_text().replace('unsigned long long','ulong').replace('long long','long').replace('__builtin_clzll','clz').replace('1ull','1ul').replace('half','halfway')
        z.writestr('fiarc/scalar.cl',scalar)
        for path in sorted((here/'fixtures/current').iterdir()):
            if path.is_file():z.write(path,'fiarc/check_cli/'+path.name)
        for path in (here/'licenses').glob('*.txt'):z.write(path,'licenses/'+path.name)
        z.write(here.parent/'LICENSE','licenses/XERO-MIT.txt')
        z.write(here/'README.md','README.md')
    receipt=dict(jar=out.name,bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        dependencies={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in jars})
    (a.out/'build_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
