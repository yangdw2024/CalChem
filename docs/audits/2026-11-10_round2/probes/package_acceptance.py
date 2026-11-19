import json,os,subprocess,sys,tarfile,tempfile,zipfile
from pathlib import Path
tempfile.tempdir=os.environ['TEMP']
BASE=Path(os.environ['TEMP']).parent
OUT=BASE/os.environ.get('CALCHEM_PACKAGE_ACCEPTANCE_DIR','package_acceptance');OUT.mkdir(exist_ok=True)
def inspect(wheel):
    with zipfile.ZipFile(wheel) as z:
        names=z.namelist();meta=z.read(next(n for n in names if n.endswith('.dist-info/WHEEL'))).decode()
        binaries=[n for n in names if n.endswith(('.pyd','.so'))]
        assert 'Root-Is-Purelib: false' in meta and 'Tag: cp310-abi3-win_amd64' in meta,meta
        assert len(binaries)==1,binaries
        assert 'calchem/data/d3/COPYING.LESSER' in names
        dest=OUT/('install_'+wheel.parent.parent.name);dest.mkdir(exist_ok=True);z.extractall(dest)
    smoke='''import calchem,json;from calchem import _core as c
w=c.run_single_point(['He'],[[0.,0.,0.]],method='rhf',basis='sto-3g')
print(json.dumps({'package':calchem.__file__,'extension':c.__file__,'energy':w.total_energy_hartree,'converged':w.converged}))
assert w.converged and abs(w.total_energy_hartree+2.8077839575)<1e-8
assert hasattr(c.IrcBranch,'converged_to_minimum')
assert hasattr(c.IrcResult,'barrier_forward_kcal_mol')
d=c.run_d3_dispersion(['O','H','H'],[[0.,0.,.1173/.529177210903],[0.,.757/.529177210903,-.4692/.529177210903],[0.,-.757/.529177210903,-.4692/.529177210903]],'b3lyp','bj')
assert abs(d.energy_hartree+.000573876613451887)<1e-12
try:c.run_single_point(['He'],[[0.,0.,0.]],method='cam-b3lyp')
except ValueError:pass
else:raise AssertionError('wheel incorrectly advertises native LRC')
'''
    env=os.environ.copy();env['PYTHONPATH']=str(dest)
    r=subprocess.run([sys.executable,'-B','-c',smoke],cwd=OUT,env=env,capture_output=True,text=True,check=True)
    print(json.dumps({'probe':'wheel', 'file':str(wheel),'metadata':meta,'binary':binaries,'smoke':json.loads(r.stdout)}),flush=True)
source=BASE/os.environ.get('CALCHEM_PACKAGE_STAGE','package_stage')/'dist'
inspect(next(source.glob('*.whl')))
sdist=next(source.glob('*.tar.gz'))
with tarfile.open(sdist) as t:
    names=t.getnames()
    assert any(n.endswith('/crates/calchem-dft/data/d3/c6.bin') for n in names)
    assert not any(n.endswith('.pyd') for n in names)
    unpack=OUT/'sdist';unpack.mkdir(exist_ok=True);t.extractall(unpack,filter='data')
root=next(unpack.iterdir())
env=os.environ.copy()
job=subprocess.run([sys.executable,'-B','-c','import os,tempfile;tempfile.tempdir=os.environ["TEMP"];from setuptools import build_meta;print(build_meta.build_wheel("dist"))'],cwd=root,env=env,capture_output=True,text=True)
(OUT/'sdist-wheel-build.log').write_text(job.stdout+'\n'+job.stderr,encoding='utf-8')
assert job.returncode==0,job.stderr[-1500:]
inspect(next((root/'dist').glob('*.whl')))
print(json.dumps({'probe':'sdist_rebuild','members':len(names),'contains_native_binary':False,'success':True}),flush=True)
