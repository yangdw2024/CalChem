import csv
import json
from pathlib import Path
import mpmath as mp

mp.mp.dps=70
root=Path(r'E:\YDW\Trae_Gaussian\docs\audits\2026-11-09_redteam')
rows=[]
for r in csv.DictReader((root/'evidence/boys-values.csv').open(encoding='utf-8-sig')):
    n=int(r['n']); x=mp.mpf(r['x']);v=mp.mpf(r['value'])
    if r['kind']=='erf':
        ref=mp.erf(x)
    elif x==0:
        ref=mp.mpf(1)/(2*n+1)
    else:
        ref=mp.gammainc(mp.mpf(n)+mp.mpf('.5'),0,x)/(2*x**(mp.mpf(n)+mp.mpf('.5')))
    rows.append({'kind':r['kind'],'n':n,'x':float(x),'actual':float(v),'reference':float(ref),'absolute_error':float(abs(v-ref)),'relative_error':float(abs(v-ref)/max(abs(ref),mp.mpf('1e-300')))})
for label,subset in [('boys_n_le_16',[r for r in rows if r['kind']=='boys' and r['n']<=16]),('boys_n_ge_24',[r for r in rows if r['kind']=='boys' and r['n']>=24]),('erf',[r for r in rows if r['kind']=='erf'])]:
    print(json.dumps({'probe':label,'samples':len(subset),'worst_relative':max(subset,key=lambda r:r['relative_error']),'worst_absolute':max(subset,key=lambda r:r['absolute_error'])}))
