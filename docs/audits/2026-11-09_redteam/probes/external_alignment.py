"""Fresh Gaussian/ORCA runs with explicit geometry, methods, AO convention, and retained raw files."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import numpy as np

ROOT=Path(r'E:\YDW\Trae_Gaussian')
sys.path.insert(0,str(ROOT))
tempfile.tempdir=os.environ['TEMP']
RUNTIME=Path(os.environ['TEMP']).parent
OUT=RUNTIME/'external_alignment'
OUT.mkdir(exist_ok=True)
from calchem import _core as c

symbols=['O','H','H']
xyz=np.array([[0.,0.,.1173],[0.,.757,-.4692],[0.,-.757,-.4692]])
atoms='\n'.join(f'{s} {r[0]:.12f} {r[1]:.12f} {r[2]:.12f}' for s,r in zip(symbols,xyz))

def emit(d):
    print(json.dumps(d),flush=True)

for method,gmethod in [('rhf','HF'),('svwn5','SVWN5'),('blyp','BLYP'),('pbe','PBEPBE'),('b3lyp','B3LYP'),('cam-b3lyp','CAM-B3LYP'),('wb97x-d','wB97XD')]:
    work=OUT/('g16_'+method);work.mkdir(exist_ok=True)
    inp=f'%chk=reference.chk\n%mem=512MB\n%nprocshared=1\n#p {gmethod}/6-31G(d,p) 6D 10F NoSymm SCF=(Tight,MaxCycle=150) Integral=UltraFine Pop=Full\n\nIndependent CalChem audit\n\n0 1\n{atoms}\n\n'
    (work/'input.gjf').write_text(inp,encoding='ascii')
    env=os.environ.copy();env.update({'g16root':r'D:\G16W','GAUSS_EXEDIR':r'D:\G16W','GAUSS_SCRDIR':str(work)})
    env['PATH']=r'D:\G16W'+os.pathsep+env['PATH']
    t=time.perf_counter()
    try:
        r=subprocess.run([r'D:\G16W\g16.exe','input.gjf'],cwd=work,env=env,capture_output=True,text=True,errors='replace',timeout=120)
        (work/'launcher.txt').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
        log=next((p for p in [work/'input.out',work/'input.log'] if p.exists()),None)
        text=log.read_text(errors='replace') if log else r.stdout+r.stderr
        en=re.findall(r'SCF Done:\s+E\([^)]*\)\s*=\s*([-\d.]+)',text)
        dip=re.findall(r'X=\s*([-\d.]+)\s+Y=\s*([-\d.]+)\s+Z=\s*([-\d.]+)\s+Tot=\s*([-\d.]+)',text)
        occ=re.findall(r'Alpha\s+occ\. eigenvalues --([^\n]+)',text)
        virt=re.findall(r'Alpha virt\. eigenvalues --([^\n]+)',text)
        w=c.run_single_point(symbols,(xyz/.529177210903).tolist(),method=method,basis='6-31g(d,p)',grid_level='fine',max_iter=150,conv_energy=1e-12,conv_density=1e-10)
        emit({'engine':'Gaussian16','method':method,'basis':'6-31g(d,p)','ao_convention':'6D10F','rc':r.returncode,'normal': 'Normal termination of Gaussian' in text,'raw':str(log),'sha256':hashlib.sha256(text.encode()).hexdigest(),'g16_energy':float(en[-1]) if en else None,'native_energy':w.total_energy_hartree,'delta_Eh':w.total_energy_hartree-float(en[-1]) if en else None,'g16_dipole_debye':list(map(float,dip[-1])) if dip else None,'native_dipole_debye':list(w.dipole_debye),'g16_homo_Eh':float(occ[-1].split()[-1]) if occ else None,'g16_lumo_Eh':float(virt[0].split()[0]) if virt else None,'native_homo_Eh':w.homo_energy_hartree,'native_lumo_Eh':w.lumo_energy_hartree,'elapsed_both_seconds':time.perf_counter()-t})
    except Exception as exc:
        emit({'engine':'Gaussian16','method':method,'error':repr(exc)})

for method,omethod in [('rhf','RHF'),('pbe','PBE'),('b3lyp','B3LYP D3BJ NoRI')]:
    work=OUT/('orca_'+method);work.mkdir(exist_ok=True)
    inp=f'! {omethod} STO-3G VeryTightSCF DEFGRID3 NoAutoStart\n%maxcore 512\n* xyz 0 1\n{atoms}\n*\n'
    (work/'input.inp').write_text(inp,encoding='ascii')
    try:
        t=time.perf_counter()
        r=subprocess.run([r'E:\ORCA\software\ORCA6.1.1\orca.exe','input.inp'],cwd=work,env=os.environ.copy(),capture_output=True,text=True,errors='replace',timeout=120)
        text=r.stdout+'\n'+r.stderr;(work/'output.out').write_text(text,encoding='utf-8')
        en=re.findall(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)',text)
        disp=re.findall(r'Dispersion correction\s+([-\d.]+)',text)
        w=c.run_single_point(symbols,(xyz/.529177210903).tolist(),method=method,basis='sto-3g',grid_level='fine',max_iter=150,conv_energy=1e-12,conv_density=1e-10)
        emit({'engine':'ORCA','method':method,'basis':'sto-3g','rc':r.returncode,'normal':'ORCA TERMINATED NORMALLY' in text,'raw':str(work/'output.out'),'orca_energy':float(en[-1]) if en else None,'native_energy':w.total_energy_hartree,'delta_Eh':w.total_energy_hartree-float(en[-1]) if en else None,'orca_d3_Eh':float(disp[-1]) if disp else None,'native_d3_Eh':c.run_d3_dispersion(symbols,(xyz/.529177210903).tolist(),'b3lyp','bj').energy_hartree if method=='b3lyp' else None,'elapsed_both_seconds':time.perf_counter()-t})
    except Exception as exc:
        emit({'engine':'ORCA','method':method,'error':repr(exc)})
