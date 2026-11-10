"""Independent audit probes. Never changes business source or expected values."""
import importlib.util
import json
import math
import os
import pathlib
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(r'E:\YDW\Trae_Gaussian')
sys.path.insert(0, str(ROOT))
tempfile.tempdir = os.environ['TEMP']
extension = pathlib.Path(os.environ['TEMP']).parents[0] / '_core.pyd'
spec = importlib.util.spec_from_file_location('calchem._core', extension)
c = importlib.util.module_from_spec(spec)
sys.modules['calchem._core'] = c
spec.loader.exec_module(c)

def emit(label, data):
    print(json.dumps({'probe': label, **data}, allow_nan=False), flush=True)

def sp(symbols, xyz, method='rhf', basis='sto-3g', charge=0, multiplicity=1, grid='fine'):
    return c.run_single_point(symbols, np.asarray(xyz).tolist(), charge, multiplicity, method, basis,
                              max_iter=150, conv_energy=1e-12, conv_density=1e-10, grid_level=grid)

def grad(symbols, xyz, method='rhf', basis='sto-3g', charge=0, multiplicity=1):
    return np.asarray(c.run_gradient(symbols, np.asarray(xyz,dtype=float), charge, multiplicity, basis, method, grid_level='fine'))

emit('provenance', {'extension': str(extension), 'python':sys.version})
symbols=['O','H','H']
xyz=np.array([[.04,-.03,.05],[.07,1.51,-.04],[1.21,-.66,.09]])
for method,basis,charge,mult in [('rhf','sto-3g',0,1),('rhf','6-31g(d,p)',0,1),('b3lyp','sto-3g',0,1),('pbe','sto-3g',0,1),('uhf','sto-3g',1,2)]:
    t=time.perf_counter()
    w=sp(symbols,xyz,method,basis,charge,mult)
    g=grad(symbols,xyz,method,basis,charge,mult)
    steps=[]
    for h in [1e-3,3e-4,1e-4]:
        fd=np.zeros_like(g)
        for a in range(3):
            for q in range(3):
                xp,xm=xyz.copy(),xyz.copy(); xp[a,q]+=h;xm[a,q]-=h
                fd[a,q]=(sp(symbols,xp,method,basis,charge,mult).total_energy_hartree-sp(symbols,xm,method,basis,charge,mult).total_energy_hartree)/(2*h)
        err=g-fd
        steps.append({'h_bohr':h,'max_abs_Eh_bohr':float(abs(err).max()),'relative_L2':float(np.linalg.norm(err)/max(np.linalg.norm(fd),1e-15))})
    angle=.714
    rot=np.array([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1.]])
    wr=sp(symbols,xyz@rot.T,method,basis,charge,mult)
    gr=grad(symbols,xyz@rot.T,method,basis,charge,mult)
    wt=sp(symbols,xyz+[2.1,-1.7,.9],method,basis,charge,mult)
    order=[2,0,1];wp=sp([symbols[a] for a in order],xyz[order],method,basis,charge,mult)
    hist=np.asarray(w.energy_history)
    emit('gradient_and_symmetry',{'method':method,'basis':basis,'charge':charge,'energy':w.total_energy_hartree,'gradient':g.tolist(), 'dipole_debye':list(w.dipole_debye), 'fd':steps,'gradient_sum_norm':float(np.linalg.norm(g.sum(axis=0))), 'translation_energy_error':wt.total_energy_hartree-w.total_energy_hartree,'rotation_energy_error':wr.total_energy_hartree-w.total_energy_hartree,'rotation_gradient_max_error':float(abs(gr-g@rot.T).max()), 'permutation_energy_error':wp.total_energy_hartree-w.total_energy_hartree,'scf_uphill_steps':int(np.sum(np.diff(hist)>1e-10)),'scf_max_uphill':float(max(np.diff(hist),default=0.0)),'seconds':time.perf_counter()-t})

for delta in [-.2,-.1,-.01,0.,.01,.1,.2]:
    x=xyz.copy(); x[1,1]+=delta/0.529177210903
    w=sp(symbols,x)
    g=grad(symbols,x)
    emit('coordinate_scan',{'delta_angstrom':delta,'energy':w.total_energy_hartree,'gradient_H_y':float(g[1,1]),'dipole_debye':list(w.dipole_debye)})

# An optimized ordinary minimum must not be certified as a first-order transition state.
o=c.run_optimize(['H','H'], [[0.,0.,0.],[0.,0.,1.4]], basis='sto-3g', max_steps=60)
x=np.asarray(o.coords).reshape(2,3).tolist()
ts=c.run_ts_search(['H','H'], x, method='rhf',basis='sto-3g',max_steps=1)
emit('minimum_as_ts',{'converged':ts.converged,'reported_imaginary_frequency_cm':ts.imaginary_frequency_cm,'geometry_bohr':ts.geometry_bohr,'gradient_max':float(abs(np.asarray(ts.gradient)).max())})

# D3 wrapper must fail transparently on unsupported functional, rather than change models.
from calchem.dft.d3 import compute_d3_energy
try:
    c.run_d3_dispersion(symbols,xyz.tolist(),'not-a-functional','bj')
except Exception as exc:
    emit('d3_invalid_core',{'raises':type(exc).__name__,'message':str(exc)})
try:
    emit('d3_invalid_wrapper',{'returned_energy':float(compute_d3_energy(symbols,xyz,'not-a-functional',damping='bj'))})
except Exception as exc:
    emit('d3_invalid_wrapper',{'raises':type(exc).__name__,'message':str(exc)})

for model in ['cpcm','cosmo']:
    w=sp(symbols,xyz)
    sg=c.run_solvation_gradient(symbols,xyz.tolist(),'sto-3g',np.asarray(w.density).reshape(w.n_ao,w.n_ao),_solvent_model=model)
    emit('solvent_gradient_model',{'model':model,'gradient':np.asarray(sg).tolist()})

# Deliberately force nonconvergence to test that the returned object is not silently successful.
try:
    r=c.run_solvation(symbols,xyz.tolist(),'sto-3g',method='rhf',max_iter=1)
    emit('scrf_one_iteration',{'returned':True,'converged':r.converged,'fields':[n for n in dir(r) if not n.startswith('_')]})
except Exception as exc:
    emit('scrf_one_iteration',{'returned':False,'raises':type(exc).__name__,'message':str(exc)})
