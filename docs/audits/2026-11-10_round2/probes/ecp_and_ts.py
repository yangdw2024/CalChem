import hashlib,json,os,sys,tempfile,time
from pathlib import Path
import numpy as np
R=Path(r'E:\YDW\Trae_Gaussian');sys.path.insert(0,str(R));tempfile.tempdir=os.environ['TEMP']
from calchem import _core as c
def emit(name,**kw):print(json.dumps({'probe':name,**kw}),flush=True)
emit('provenance',extension=c.__file__,sha256=hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest())
for symbols,x,basis,mult in [(['H','Br'],np.array([[.03,0.,0.],[0.,.04,2.7]]),'lanl2dz',1),(['H','I'],np.array([[.03,0.,0.],[0.,.04,3.0]]),'lanl2dzdp',1)]:
    g=np.asarray(c.run_gradient(symbols,x,multiplicity=mult,basis=basis,method='rhf'))
    errors=[]
    for h in [3e-4,1e-4]:
        xp=x.copy();xm=x.copy();xp[1,2]+=h;xm[1,2]-=h
        def energy(y):return c.run_single_point(symbols,y.tolist(),method='rhf',basis=basis,multiplicity=mult,conv_energy=1e-12,conv_density=1e-10).total_energy_hartree
        fd=(energy(xp)-energy(xm))/(2*h);errors.append(abs(fd-g[1,2]))
    assert max(errors)<1e-6
    emit('ECP_gradient',symbols=symbols,basis=basis,gradient_errors=errors,total_force_norm=float(np.linalg.norm(g.sum(axis=0))))
o=c.run_optimize(['H','H'],[[0.,0.,0.],[0.,0.,1.4]],basis='sto-3g',method='rhf')
ts=c.run_ts_search(['H','H'],np.asarray(o.coords).reshape(2,3).tolist(),method='rhf',basis='sto-3g',max_steps=1)
assert not ts.converged and ts.imaginary_frequency_cm>0.
emit('minimum_TS_rejection',converged=ts.converged,frequency_cm=ts.imaginary_frequency_cm)
from calchem.backends.native import NativeCalchem
from calchem.core.request import MoleculeSpec,CalculationRequest
spec=MoleculeSpec(symbols=['C','N','H'],coordinates_angstrom=(np.array([[0.,0.,0.],[2.308,0.,0.],[.672,2.171,0.]])*.529177210903).tolist())
req=CalculationRequest(task='irc',molecule=spec,backend='native',method='rhf',basis='sto-3g',options={'max_points':3,'max_gs_iter':5})
r=NativeCalchem().run(req)
assert r.status=='not_converged' and r.irc_forward_barrier_kcal_mol is None and r.irc_reverse_barrier_kcal_mol is None
emit('IRC_partial_status',status=r.status,points=len(r.irc_path),forward_barrier=r.irc_forward_barrier_kcal_mol,reverse_barrier=r.irc_reverse_barrier_kcal_mol)
