import json, os, re, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
ROOT=Path(r'E:\YDW\Trae_Gaussian');sys.path.insert(0,str(ROOT));tempfile.tempdir=os.environ['TEMP']
from calchem import _core as c
OUT=Path(os.environ['TEMP']).parent/'external_followup';OUT.mkdir(exist_ok=True)
syms=['O','H','H'];xyz=np.array([[0.,0.,.1173],[0.,.757,-.4692],[0.,-.757,-.4692]])

def g16(name, route, coords):
    d=OUT/name;d.mkdir(exist_ok=True)
    atoms='\n'.join(f'{s} {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}' for s,p in zip(syms,coords))
    (d/'input.gjf').write_text(f'%mem=512MB\n%nprocshared=1\n#p {route} NoSymm SCF=(Tight,MaxCycle=150) Integral=UltraFine\n\nIndependent followup\n\n0 1\n{atoms}\n\n',encoding='ascii')
    env=os.environ.copy();env.update({'g16root':r'D:\G16W','GAUSS_EXEDIR':r'D:\G16W','GAUSS_SCRDIR':str(d)});env['PATH']=r'D:\G16W'+os.pathsep+env['PATH']
    r=subprocess.run([r'D:\G16W\g16.exe','input.gjf'],cwd=d,env=env,capture_output=True,text=True,errors='replace',timeout=120)
    (d/'launcher.txt').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
    p=d/'input.out';t=p.read_text(errors='replace') if p.exists() else r.stdout
    if r.returncode!=0 or 'Normal termination of Gaussian' not in t:
        raise RuntimeError(f'Gaussian failed: {name}, rc={r.returncode}, {p}')
    return t

def emit(label,**data):print(json.dumps({'test':label,**data}),flush=True)

t=g16('pbe_sto','PBEPBE/STO-3G',xyz)
w=[]
for gl in ['coarse','medium','fine']:
    f=c.run_single_point(syms,(xyz/.529177210903).tolist(),method='pbe',basis='sto-3g',grid_level=gl)
    w.append({'grid':gl,'energy':f.total_energy_hartree})
emit('pbe_sto',gaussian=float(re.findall(r'SCF Done:.*?=\s*([-\d.]+)',t)[-1]),native=w)

t=g16('b3lyp_d3','B3LYP/STO-3G EmpiricalDispersion=GD3BJ',xyz)
emit('d3_bj',gaussian_dispersion=re.findall(r'(?:Grimme-D3|R6Disp|Dispersion|Edisp)[^\n]*',t),native_dispersion=c.run_d3_dispersion(syms,(xyz/.529177210903).tolist(),'b3lyp','bj').energy_hartree)

for label,route,method,tda in [('cis','CIS(NStates=3,Singlets)/STO-3G','cis',True),('tddft_b3lyp','B3LYP/STO-3G TD(NStates=3,Singlets)','b3lyp',False)]:
    t=g16(label,route,xyz)
    states=re.findall(r'Excited State\s+(\d+):[^\n]*?([\d.]+) eV[^\n]*?f=([\d.]+)',t)
    r=c.run_tddft(syms,(xyz/.529177210903).tolist(),method=method,basis='sto-3g',n_states=3,tda=tda,grid_level='fine',conv_tol=1e-10)
    emit(label,gaussian_states=states,native_states=[{'energy_ev':s.excitation_energy_ev,'oscillator_strength':s.oscillator_strength} for s in r.states])

o=c.run_optimize(syms,(xyz/.529177210903).tolist(),method='rhf',basis='sto-3g',max_steps=80)
x=np.asarray(o.coords).reshape(3,3)
t=g16('frequency','HF/STO-3G Freq',x*.529177210903)
r=c.run_frequency(syms,x.tolist(),method='rhf',basis='sto-3g',step_bohr=.0003,symmetry_number=2,conv_energy=1e-12,conv_density=1e-10)
emit('frequency',gaussian_frequencies=re.findall(r'Frequencies --\s+([^\n]+)',t),gaussian_ir=re.findall(r'IR Inten\s+--\s+([^\n]+)',t),native_frequencies=r.frequencies_cm,native_ir=r.ir_intensities_km_mol,gaussian_zpe=re.findall(r'Zero-point correction=\s+([^\s]+)',t),native_zpe=r.zpve_hartree)
