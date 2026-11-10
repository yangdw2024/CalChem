"""Direct high-level donor/acceptor alignment run, using fresh native SCFs."""
from dataclasses import asdict
import json,os,sys,tempfile,time
from pathlib import Path
ROOT=Path(r'E:\YDW\Trae_Gaussian');sys.path.insert(0,str(ROOT));tempfile.tempdir=os.environ['TEMP']
from calchem.workflows.opv import OPVWorkflow,get_pm6_fragment,get_y6_fragment
t=time.perf_counter()
wf=OPVWorkflow(backend='native')
r=wf.align_donor_acceptor(get_pm6_fragment().to_spec(),get_y6_fragment().to_spec(),method='b3lyp',basis='sto-3g',optimize_geometries=False)
print(json.dumps({'calculation':'direct_OPVWorkflow.align_donor_acceptor','seconds':time.perf_counter()-t,'result':asdict(r)}),flush=True)
