"""Convert upstream D3 tables, never fitted to molecular test energies.
Sources and their exact SHA256 are retained beside generated data.
"""
from pathlib import Path
import re, struct, hashlib, json, shutil
BASE=Path(r'E:\YDW\Trae_Gaussian\build_target\runtime\round2_20261110')
SRC=BASE/'references'; DEST=BASE/'repair_stage/crates/calchem-dft/data/d3'
DEST.mkdir(parents=True,exist_ok=True)
def nums(s): return [float(x) for x in re.findall(r'([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)_wp',s)]
def arr(file, marker):
    s=(SRC/file).read_text(); s=s[s.index(marker):]; return nums(s[s.index('[')+1:s.index(']')])
s=(SRC/'reference.f90').read_text()
cn=arr('reference.f90','reference_cn(max_ref, max_elem)')
assert len(cn)==103*7
c6=[]
for a,b,values in re.findall(r'c6ab_view\((\d+):(\d+)\)\s*=\s*\[(.*?)\]',s,re.S):
    v=nums(values); assert len(v)==int(b)-int(a)+1 and len(c6)==int(a)-1
    c6.extend(v)
assert len(c6)==7*7*103*104//2
r4=arr('data_r4r2.f90','r4_over_r2(max_elem)')
cov=arr('covrad.f90','covalent_rad_2009(max_elem)')
vdw=arr('data_vdwrad.f90','vdwrad(max_elem*(1+max_elem)/2)')
assert len(r4)==len(cov)==118 and len(vdw)==5356
(DEST/'c6.bin').write_bytes(struct.pack('<'+'d'*len(c6),*c6))
out=['// Generated from upstream tables; see ../data/d3/NOTICE.md.\n']
for name,values in [('REFERENCE_CN',cn),('R4_OVER_R2',r4),('COVALENT_ANGSTROM',cov),('VDW_PAIR_ANGSTROM',vdw)]:
    out.append(f'pub const {name}: [f64; {len(values)}] = [\n')
    out.extend(', '.join(repr(v) for v in values[i:i+7])+',\n' for i in range(0,len(values),7))
    out.append('];\n')
(BASE/'repair_stage/crates/calchem-dft/src/d3_data.rs').write_text(''.join(out))
manifest={}
for f in ['reference.f90','data_r4r2.f90','data_vdwrad.f90','covrad.f90','COPYING','COPYING.LESSER']:
    shutil.copyfile(SRC/f,DEST/f)
    manifest[f]=hashlib.sha256((SRC/f).read_bytes()).hexdigest()
(DEST/'source_sha256.json').write_text(json.dumps(manifest,indent=2))
(DEST/'NOTICE.md').write_text('''# Standard D3 reference data

Retrieved 2026-11-10 from https://github.com/dftd3/simple-dftd3 (main).
reference.f90, data_r4r2.f90, data_vdwrad.f90 and their converted tables are
LGPL-3.0-or-later (s-dftd3 authors); COPYING and COPYING.LESSER are included.
covrad.f90 is from https://github.com/grimme-lab/mctc-lib, Apache-2.0.
Exact downloaded revisions are identified by source_sha256.json; GitHub's
commit API was rate-limited, so no unverified commit id is asserted.

c6.bin is a lossless little-endian f64 conversion of reference_c6 in Fortran
column-major order (7,7,5356). d3_data.rs holds reference CN and atomic radii.
Original corresponding source tables are included here to make conversion
and redistribution auditable. No upstream numerical engine is called.
The Rust evaluator is a CalChem implementation of two-body D3; ATM is absent.
''',encoding='utf-8')
print(json.dumps({'c6_values':len(c6),'cn_values':len(cn),'source_sha256':manifest}))
