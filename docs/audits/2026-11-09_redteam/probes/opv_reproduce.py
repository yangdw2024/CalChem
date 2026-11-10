"""Reproduce the existing OPV suite without overwriting its historical JSON artifact."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT=Path(r'E:\YDW\Trae_Gaussian');sys.path.insert(0,str(ROOT))
tempfile.tempdir=os.environ['TEMP']
from calchem.workflows.opv import OPVWorkflow,get_real_opv_system,get_pm6_fragment,get_y6_fragment,CHEMDATAS_REFERENCE_DATA
wf=OPVWorkflow(backend='native')

def emit(label,**data):
    print(json.dumps({'step':label,**data}),flush=True)

inventory={}
for name in ['y6_ic_ff','y6_btp_core','y6_full','pm6_monomer','pm6_l8bo_cluster']:
    m=get_real_opv_system(name);inventory[name]=m
    p=Path(CHEMDATAS_REFERENCE_DATA[name]['source_file'])
    emit('inventory',name=name,n_atoms=len(m.symbols),formula=dict(Counter(m.symbols)),source=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None)
donor=get_pm6_fragment();ic=get_y6_fragment();nc=get_y6_fragment(variant='nc1');cluster=inventory['pm6_l8bo_cluster']
for method in ['b3lyp','camb3lyp','wb97xd']:
    for name,m in [('Donor-BDD1',donor),('IC-2F',ic)]:
        t=time.perf_counter();r=wf.single_point(m.to_spec(),method=method,basis='sto-3g',name=name)
        emit('single_point',name=name,method=method,basis='sto-3g',converged=r.converged,energy=r.scf_energy_hartree,homo_ev=r.homo_ev,lumo_ev=r.lumo_ev,gap_ev=r.gap_ev,seconds=time.perf_counter()-t,errors=r.errors)

parts={}
for name,m in [('cluster60',cluster),('donor32',get_real_opv_system('donor_bdd1')),('acceptor28',get_real_opv_system('acceptor_nc1'))]:
    t=time.perf_counter();r=wf.single_point(m.to_spec(),method='camb3lyp',basis='sto-3g',dispersion='d3bj',name=name)
    parts[name]=r
    emit('cluster_binding_part',name=name,converged=r.converged,total_energy=r.total_energy_hartree,dispersion=r.dispersion_energy_hartree,seconds=time.perf_counter()-t,errors=r.errors)
if all(r.total_energy_hartree is not None for r in parts.values()):
    emit('cluster_binding',energy_kcal=(parts['cluster60'].total_energy_hartree-parts['donor32'].total_energy_hartree-parts['acceptor28'].total_energy_hartree)*627.509474)
for solvent in ['chlorobenzene','chloroform']:
    for name,m in [('donor32',donor),('ic2f',ic)]:
        t=time.perf_counter();r=wf.solvation_free_energy(m.to_spec(),solvent=solvent,method='b3lyp',basis='sto-3g',name=name)
        emit('solvation',name=name,solvent=solvent,delta_g_kcal=r.delta_g_solv_kcal_mol,area_A2=r.cavity_surface_area_angstrom2,seconds=time.perf_counter()-t)
t=time.perf_counter()
r=wf.optical_absorption(ic.to_spec(),method='camb3lyp',basis='sto-3g',n_states=3,tda=True,compute_exciton_binding=False,name='IC2F_audit')
emit('optical',optical_gap_ev=r.optical_gap_ev,wavelength_max_nm=r.wavelength_max_nm,oscillator_strength_max=r.oscillator_strength_max,ks_gap_ev=r.ks_gap_ev,states=r.states,seconds=time.perf_counter()-t)
