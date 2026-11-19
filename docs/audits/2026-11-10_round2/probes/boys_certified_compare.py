import csv,json
from pathlib import Path
import mpmath as mp
mp.mp.dps=80
AUDIT=Path(__file__).resolve().parent.parent
rows=[]
for r in csv.DictReader((AUDIT/'evidence/boys-certified.csv').open(encoding='utf-8-sig')):
    n=int(r['n']);t=mp.mpf(r['x']);v=mp.mpf(r['value'])
    ref=mp.mpf(1)/(2*n+1) if not t else mp.gammainc(mp.mpf(n)+mp.mpf('.5'),0,t)/(2*t**(mp.mpf(n)+mp.mpf('.5')))
    rows.append(dict(n=n,t=float(t),v=float(v),ref=float(ref),abs=float(abs(v-ref)),rel=float(abs(v-ref)/ref),refmp=ref))
def summarize(name,rs):
    worst=max(rs,key=lambda r:r['rel']);print(json.dumps({'group':name,'samples':len(rs),'worst':{k:v for k,v in worst.items() if k!='refmp'}}),flush=True)
    return worst
normal=[r for r in rows if r['refmp']>mp.mpf('2.2250738585072014e-308')]
ordinary=[r for r in normal if r['n']<=128]
focus=[r for r in normal if 24<=r['n']<=128 and 34.99<=r['t']<=35.01]
summarize('normal_all',normal);summarize('ordinary_n_le_128',ordinary)
worst=summarize('focus_24_128_T35',focus)
assert worst['rel']<1e-15
# IEEE 754 cannot promise a relative error bound when the exact result underflows.
tiny=[r for r in rows if r not in normal]
print(json.dumps({'group':'subnormal_or_underflow','samples':len(tiny),'max_absolute_error':max(r['abs'] for r in tiny)}))
assert all(r['v']>=0 for r in rows)
assert max(r['rel'] for r in ordinary)<1e-13
