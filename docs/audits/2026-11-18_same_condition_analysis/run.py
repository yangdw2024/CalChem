import hashlib, json, os, subprocess, time
from pathlib import Path
import psutil
root=Path('E:/YDW/Trae_Gaussian')
out=root/'docs/audits/2026-11-18_same_condition_analysis'
exe=root/'build_target/release/calchem_cli.exe'
for molecule,xc in [('ydw-PTQ10','b3lyp'),('ydw-BTP-4F','pbe')]:
    path=Path('E:/YDW/ChemDatas/Gjf')/(molecule+'.gjf')
    args=[str(exe),str(path),'--rij','--gpu','--grid','coarse','--xc',xc,'--basis','sto-3g','--conv-e','1e-6','--conv-d','1e-8','--max-iter','128']
    env=dict(os.environ,RAYON_NUM_THREADS='4',CALCHEM_PROFILE='1')
    start=time.monotonic()
    peak=0
    with (out/(molecule+'.log')).open('w',encoding='utf-8') as f:
        proc=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,env=env)
        p=psutil.Process(proc.pid)
        while proc.poll() is None:
            try: peak=max(peak,p.memory_info().peak_wset)
            except psutil.NoSuchProcess: pass
            time.sleep(.25)
    data={'args':args,'threads':4,'exit':proc.returncode,'wall_s':time.monotonic()-start,'peak_working_set_bytes':peak,'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest()}
    (out/(molecule+'.json')).write_text(json.dumps(data,indent=2),encoding='utf-8')
    if proc.returncode: raise SystemExit(proc.returncode)
