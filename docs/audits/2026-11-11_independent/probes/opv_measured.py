import ctypes
from ctypes import wintypes
from collections import Counter
import json
import time
from calchem import _core
from calchem.workflows.opv import get_real_opv_system

class MemoryCounters(ctypes.Structure):
    _fields_ = [('cb',wintypes.DWORD), ('PageFaultCount',wintypes.DWORD)] + [
        (n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize',
        'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
        'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]

system = get_real_opv_system('y6_ic_ff')
started = time.perf_counter()
wf = _core.run_single_point(system.symbols,
    [[c/0.529177210903 for c in r] for r in system.coordinates_angstrom],
    method='b3lyp', basis='sto-3g', grid_level='coarse',
    conv_energy=1e-6, conv_density=1e-5, direct=True)
elapsed = time.perf_counter()-started
memory = MemoryCounters()
memory.cb = ctypes.sizeof(memory)
ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
handle = ctypes.windll.kernel32.GetCurrentProcess()
ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,ctypes.POINTER(MemoryCounters),wintypes.DWORD]
ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle,ctypes.byref(memory),ctypes.sizeof(memory))
print(json.dumps({'system':'IC-2F','atoms':len(system.symbols),'formula_counts':dict(Counter(system.symbols)),
    'n_ao':wf.n_ao,'method':'B3LYP','basis':'STO-3G','grid':'coarse','direct':True,'density_fitting':False,
    'converged':wf.converged,'iterations':wf.iterations,'energy_hartree':wf.total_energy_hartree,
    'homo_hartree':wf.homo_energy_hartree,'lumo_hartree':wf.lumo_energy_hartree,
    'wall_seconds':elapsed,'process_peak_working_set_bytes':memory.PeakWorkingSetSize if ok else None,
    'reference_metadata':system.reference_data,
    'comparison_note':'Different functional and basis from Gaussian source: no energy-agreement certification.'},
    ensure_ascii=False,default=str),flush=True)
