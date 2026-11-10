"""Orthogonal numerical and adversarial acceptance of an explicitly selected binary."""
import hashlib,importlib.util,json,os,re,sys,tempfile,threading,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
R=Path(r'E:\YDW\Trae_Gaussian');sys.path.insert(0,str(R));tempfile.tempdir=os.environ['TEMP']
EXT=Path(os.environ['CALCHEM_AUDIT_EXTENSION'])
spec=importlib.util.spec_from_file_location('calchem._core',EXT)
c=importlib.util.module_from_spec(spec);sys.modules['calchem._core']=c;spec.loader.exec_module(c)
def emit(name,**data): print(json.dumps({'probe':name,**data},allow_nan=False),flush=True)
def rejected(name,fn):
    try: fn()
    except Exception as e: emit(name,rejected=True,type=type(e).__name__,message=str(e));return
    raise AssertionError(name+' accepted invalid/unsupported input')
def sp(s,x,method='rhf',basis='sto-3g',charge=0,mult=1,max_iter=100):
    return c.run_single_point(s,np.asarray(x).tolist(),charge,mult,method,basis,max_iter=max_iter,conv_energy=1e-12,conv_density=1e-10,grid_level='fine')
emit('provenance',extension=str(EXT),sha256=hashlib.sha256(EXT.read_bytes()).hexdigest(),version=c.__version__)
water=(['O','H','H'],np.array([[0,0,.1173],[0,.757,-.4692],[0,-.757,-.4692]])/.529177210903)
eth=(['C','C','H','H','H','H'],np.array([[-.67,0,0],[.67,0,0],[-1.23,.93,0],[-1.23,-.93,0],[1.23,.93,0],[1.23,-.93,0]])/.529177210903)
geoms={'water':water,'ethylene':eth,'hbr':(['H','Br'],np.array([[0,0,0],[0,0,1.414]])/.529177210903),'water_stretched':(water[0],water[1]+np.array([[0,0,0],[0,.1,0],[0,0,0]])/.529177210903)}
audit=R/'docs/audits/2026-11-10_round2'
for row in map(json.loads,(audit/'evidence/external-round2.jsonl').read_text(encoding='utf-8-sig').splitlines()):
    if row['test']=='fresh_D3':
        sy,x=geoms[row['molecule']];d=c.run_d3_dispersion(sy,x.tolist(),'b3lyp',row['damping'])
        ref=float(re.search(r'energy=\s*([-\d.]+)',row['gaussian_lines'][0])[1])
        error=d.energy_hartree-ref;assert abs(error)<1e-9,(row,error)
        h=2e-4;fd=np.zeros_like(x)
        for a in range(len(sy)):
            for q in range(3):
                xp=x.copy();xm=x.copy();xp[a,q]+=h;xm[a,q]-=h
                fd[a,q]=(c.run_d3_dispersion(sy,xp.tolist(),'b3lyp',row['damping']).energy_hartree-c.run_d3_dispersion(sy,xm.tolist(),'b3lyp',row['damping']).energy_hartree)/(2*h)
        graderr=float(abs(fd-np.asarray(d.gradient)).max());assert graderr<2e-9
        emit('D3_external_and_FD',molecule=row['molecule'],damping=row['damping'],reference=ref,energy=d.energy_hartree,error_Eh=error,max_gradient_error=graderr)
    elif row['test']=='fresh_IR':
        sy=water[0] if row['molecule']=='water' else eth[0]
        f=c.run_frequency(sy,row['geometry_bohr'],method='rhf',basis='sto-3g',step_bohr=.0003,conv_energy=1e-12,conv_density=1e-10)
        err=float(abs(np.array(row['gaussian_ir'])-f.ir_intensities_km_mol).max());assert err<5e-4
        emit('IR_external',molecule=row['molecule'],native_ir=list(f.ir_intensities_km_mol),gaussian_ir=row['gaussian_ir'],max_error_km_mol=err,max_frequency_error=float(abs(np.array(row['gaussian_freq'])-f.frequencies_cm).max()))

from calchem.dft.d3 import compute_d3_energy,compute_d3_gradient
for fun in (compute_d3_energy,compute_d3_gradient):
    rejected('unknown_D3_wrapper',lambda:fun(*water,functional='typo',damping='bj'))
    rejected('custom_D3_wrapper',lambda:fun(*water,s6=1.0))
rejected('unknown_D3_damping',lambda:c.run_d3_dispersion(water[0],water[1].tolist(),'b3lyp','typo'))
for name in ['cam-b3lyp','wb97x-d','wB97X']:
    rejected('unsupported_LRC_'+name,lambda:sp(*water,method=name))

for sym,x,method,basis,charge,mult in [
    (['N'],[[0,0,0]],'uhf','sto-3g',0,4),(['O'],[[0,0,0]],'uhf','sto-3g',0,3),
    (['H','H'],[[0,0,0],[0,0,8]],'uhf','sto-3g',0,1),
    (['O','H'],[[0,0,0],[.1,0,1.8]],'uhf','sto-3g',0,2),
    (['O','H'],[[0,0,0],[.1,0,1.8]],'b3lyp','sto-3g',0,2),
    (['Br'],[[0,0,0]],'uhf','lanl2dz',0,2),
    (['Fe'],[[0,0,0]],'uhf','lanl2dz',2,5)]:
    t=time.perf_counter()
    try:w=sp(sym,x,method,basis,charge,mult)
    except Exception as e:
        emit('spin_stress',symbols=sym,method=method,basis=basis,multiplicity=mult,outcome='rejected_or_nonconverged',message=str(e));continue
    assert np.isfinite(w.total_energy_hartree) and w.converged
    emit('spin_stress',symbols=sym,method=method,basis=basis,multiplicity=mult,energy=w.total_energy_hartree,s_squared=w.s_squared,n_alpha=w.n_alpha,n_beta=w.n_beta,iterations=w.iterations,seconds=time.perf_counter()-t)
    if len(sym)==2 and sym[0]=='O':
        x=np.array(x,dtype=float);g=np.asarray(c.run_gradient(sym,x,charge,mult,basis,method,grid_level='fine'))
        errors=[]
        for h in [3e-4,1e-4]:
            xp=x.copy();xm=x.copy();xp[1,2]+=h;xm[1,2]-=h
            fd=(sp(sym,xp,method,basis,charge,mult).total_energy_hartree-sp(sym,xm,method,basis,charge,mult).total_energy_hartree)/(2*h)
            errors.append(abs(fd-g[1,2]))
        assert max(errors)<1e-6
        emit('radical_gradient',method=method,errors=errors,total_force_norm=float(np.linalg.norm(g.sum(axis=0))))

sy,x=water
gas=sp(sy,x)
sol=c.run_solvation(sy,x.tolist(),'sto-3g',method='rhf',conv_energy=1e-12,conv_density=1e-10)
assert abs(sol.gas_energy-gas.total_energy_hartree)<1e-10
emit('independent_gas_reference',gas=gas.total_energy_hartree,labeled_gas=sol.gas_energy,solvated=sol.solvated_energy,delta_g=sol.delta_g_solv_kcal_mol,identity_error_kcal=sol.delta_g_solv_kcal_mol-(sol.solvated_energy-gas.total_energy_hartree)*627.5094740631)
rejected('SCF_iteration_budget',lambda:sp(sy,x,max_iter=1))
rejected('invalid_spin',lambda:sp(['H'],[[0,0,0]],mult=4))
rejected('coincident_atoms',lambda:sp(['H','H'],[[0,0,0],[0,0,0]]))
for model in ['cpcm','cosmo']:
    d=np.asarray(gas.density).reshape(gas.n_ao,gas.n_ao)
    value=c.run_solvation_gradient(sy,x.tolist(),'sto-3g',d,_solvent_model=model)
    emit('PCM_partial_gradient',model=model,gradient=np.asarray(value).tolist())
rejected('noncontiguous_density',lambda:c.run_solvation_gradient(sy,x.tolist(),'sto-3g',np.asfortranarray(d)))

stop=threading.Event();ticks=[]
def heartbeat():
    while not stop.wait(.002):ticks.append(time.perf_counter())
th=threading.Thread(target=heartbeat);th.start();t=time.perf_counter();sp(sy,x,'b3lyp');elapsed=time.perf_counter()-t;stop.set();th.join()
assert len(ticks)>5,(elapsed,len(ticks))
emit('GIL_heartbeat',seconds=elapsed,ticks=len(ticks))
with ThreadPoolExecutor(2) as pool:
    fs=[pool.submit(sp,sy,x+v) for v in [0.,2.0]]
    energies=[f.result().total_energy_hartree for f in fs]
assert abs(energies[0]-energies[1])<1e-10
emit('concurrent_translation',energies=energies,error=energies[1]-energies[0])
emit('acceptance_finished',passed=True)
