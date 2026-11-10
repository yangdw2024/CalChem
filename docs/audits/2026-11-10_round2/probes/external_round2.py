"""Fresh, bounded Gaussian jobs. Raw inputs and outputs remain in E: runtime."""
import json,os,re,subprocess,sys,tempfile,time
from pathlib import Path
import numpy as np
R=Path(r'E:\YDW\Trae_Gaussian');sys.path.insert(0,str(R));tempfile.tempdir=os.environ['TEMP']
from calchem import _core as c
OUT=Path(os.environ['TEMP']).parent/'external_round2';OUT.mkdir(exist_ok=True)
def emit(**kw): print(json.dumps(kw),flush=True)
def g16(name,route,sym,xyz,charge=0,mult=1):
    d=OUT/name;d.mkdir(exist_ok=True)
    atoms='\n'.join(f'{s} {p[0]:.14f} {p[1]:.14f} {p[2]:.14f}' for s,p in zip(sym,xyz))
    (d/'input.gjf').write_text(f'%mem=512MB\n%nprocshared=1\n#p {route} NoSymm SCF=(Conver=10,MaxCycle=150) Integral=UltraFine\n\nRound2 independent\n\n{charge} {mult}\n{atoms}\n\n',encoding='ascii')
    env=os.environ.copy();env.update(g16root=r'D:\G16W',GAUSS_EXEDIR=r'D:\G16W',GAUSS_SCRDIR=str(d));env['PATH']=r'D:\G16W'+os.pathsep+env['PATH']
    t=time.perf_counter();p=subprocess.run([r'D:\G16W\g16.exe','input.gjf'],cwd=d,env=env,capture_output=True,text=True,errors='replace',timeout=240)
    (d/'launcher.txt').write_text(p.stdout+'\n'+p.stderr,encoding='utf8')
    log=(d/'input.out').read_text(errors='replace')
    assert p.returncode==0 and 'Normal termination of Gaussian' in log,name
    return log,time.perf_counter()-t
water=(['O','H','H'],np.array([[0,0,.1173],[0,.757,-.4692],[0,-.757,-.4692]]))
ethylene=(['C','C','H','H','H','H'],np.array([[-.67,0,0],[.67,0,0],[-1.23,.93,0],[-1.23,-.93,0],[1.23,.93,0],[1.23,-.93,0]]))
for name,(sym,xyz) in [('water',water),('ethylene',ethylene)]:
    o=c.run_optimize(sym,(xyz/.529177210903).tolist(),method='rhf',basis='sto-3g',max_steps=100)
    assert o.converged
    x=np.asarray(o.coords).reshape(-1,3)
    log,t=g16(name+'_ir','HF/STO-3G Freq',sym,x*.529177210903)
    gf=[float(v) for line in re.findall(r'Frequencies --\s*([^\n]+)',log) for v in line.split()]
    gi=[float(v) for line in re.findall(r'IR Inten\s+--\s*([^\n]+)',log) for v in line.split()]
    f=c.run_frequency(sym,x.tolist(),method='rhf',basis='sto-3g',step_bohr=.0003,conv_energy=1e-12,conv_density=1e-10)
    emit(test='fresh_IR',molecule=name,gaussian_freq=gf,native_freq=list(f.frequencies_cm),gaussian_ir=gi,native_ir=list(f.ir_intensities_km_mol),max_ir_error=float(max(abs(np.array(gi)-f.ir_intensities_km_mol))),seconds=t,geometry_bohr=x.tolist())
for name,sym,xyz in [('water',*water),('ethylene',*ethylene),('hbr',['H','Br'],np.array([[0,0,0],[0,0,1.414]])),('water_stretched',water[0],water[1]+np.array([[0,0,0],[0,.1,0],[0,0,0]]))]:
    for damping,keyword in [('bj','GD3BJ'),('zero','GD3')]:
        basis='LANL2DZ' if name=='hbr' else 'STO-3G'
        log,t=g16(name+'_d3_'+damping,f'B3LYP/{basis} EmpiricalDispersion={keyword}',sym,xyz)
        lines=re.findall(r'(?:Grimme-D3|R6Disp|Edisp)[^\n]*',log)
        d=c.run_d3_dispersion(sym,(xyz/.529177210903).tolist(),'b3lyp',damping)
        emit(test='fresh_D3',molecule=name,damping=damping,gaussian_lines=lines,native=d.energy_hartree,seconds=t)
