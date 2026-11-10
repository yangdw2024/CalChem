import json,os,re,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(r'E:\YDW\Trae_Gaussian');sys.path.insert(0,str(ROOT));tempfile.tempdir=os.environ['TEMP']
from calchem import _core as c
base=Path(os.environ['TEMP']).parent/'external_extra';base.mkdir(exist_ok=True)
for name,route,syms,coords,method,basis in [
 ('cis_full','CIS(Full,NStates=3,Singlets,Conver=9)/STO-3G',['O','H','H'],[[0.,0.,.1173],[0.,.757,-.4692],[0.,-.757,-.4692]],'cis','sto-3g'),
 ('hbr_hf','HF/LANL2DZ',['H','Br'],[[0.,0.,0.],[0.,0.,1.414]],'rhf','lanl2dz'),
 ('hbr_b3lyp','B3LYP/LANL2DZ',['H','Br'],[[0.,0.,0.],[0.,0.,1.414]],'b3lyp','lanl2dz')]:
    d=base/name;d.mkdir(exist_ok=True)
    atoms='\n'.join(f'{s} {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}' for s,p in zip(syms,coords))
    (d/'input.gjf').write_text(f'%mem=512MB\n%nprocshared=1\n#p {route} NoSymm SCF=(Tight,MaxCycle=150) Integral=UltraFine\n\nAudit active space and ECP\n\n0 1\n{atoms}\n\n',encoding='ascii')
    env=os.environ.copy();env.update({'g16root':r'D:\G16W','GAUSS_EXEDIR':r'D:\G16W','GAUSS_SCRDIR':str(d)});env['PATH']=r'D:\G16W'+os.pathsep+env['PATH']
    r=subprocess.run([r'D:\G16W\g16.exe','input.gjf'],cwd=d,env=env,capture_output=True,text=True,errors='replace',timeout=120)
    (d/'launcher.txt').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
    p=d/'input.out';t=p.read_text(errors='replace') if p.exists() else r.stdout
    data={'case':name,'rc':r.returncode,'normal':'Normal termination of Gaussian' in t,'raw':str(p)}
    if data['normal']:
        x=(np.asarray(coords)/.529177210903).tolist()
        if method=='cis':
            q=c.run_tddft(syms,x,method='cis',basis=basis,tda=True,n_states=3,conv_tol=1e-10)
            data.update(gaussian_states=re.findall(r'Excited State\s+(\d+):[^\n]*?([\d.]+) eV[^\n]*?f=([\d.]+)',t),native_states=[{'energy_ev':s.excitation_energy_ev,'oscillator_strength':s.oscillator_strength} for s in q.states])
        else:
            q=c.run_single_point(syms,x,method=method,basis=basis,grid_level='fine',conv_energy=1e-12,conv_density=1e-10)
            en=float(re.findall(r'SCF Done:.*?=\s*([-\d.]+)',t)[-1]);data.update(gaussian_energy=en,native_energy=q.total_energy_hartree,delta_Eh=q.total_energy_hartree-en)
    print(json.dumps(data),flush=True)
