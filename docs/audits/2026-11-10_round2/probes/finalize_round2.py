"""Preserve final artifacts without changing business files or overwriting baselines."""
from pathlib import Path
import difflib,hashlib,json,os,re,zipfile
R=Path(r'E:\YDW\Trae_Gaussian');A=R/'docs/audits/2026-11-10_round2';B=R/'build_target/runtime/round2_20261110';OUT=B/'final_documents'
OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
baseline=json.loads((A/'evidence/source_before.json').read_text(encoding='utf-8-sig'))
files=set(baseline)
for part in ('crates','calchem','tests','tools','config','.github'):
    files.update(str(p.relative_to(R)).replace('\\','/') for p in (R/part).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
files.update(['setup.py','pyproject.toml','Cargo.toml','Cargo.lock','MANIFEST.in','docs/NATIVE_METHOD_BOUNDARIES.md'])
now={rel:{'sha256':sha(R/rel),'bytes':(R/rel).stat().st_size} for rel in sorted(files) if (R/rel).is_file()}
changed=[rel for rel,m in now.items() if rel not in baseline or m['sha256']!=baseline[rel]['sha256']]
missing=[rel for rel in baseline if not (R/rel).is_file()]
assert not missing,missing
before=zipfile.ZipFile(A/'rollback/source_snapshot.zip');assert before.testzip() is None
restore=B/'restore_round2_candidate';restore.mkdir(exist_ok=True)
if not (restore/'Cargo.toml').exists():before.extractall(restore)
restored={rel:sha(restore/rel)==v['sha256'] for rel,v in baseline.items()}
assert all(restored.values())
patch=[]
with zipfile.ZipFile(OUT/'replacements.zip','w',zipfile.ZIP_DEFLATED) as z:
    for rel in changed:
        z.write(R/rel,rel)
        new=(R/rel).read_bytes();old=before.read(rel) if rel in baseline else b''
        if b'\0' not in old+new:
            try:
                patch.extend(difflib.unified_diff(old.decode('utf-8-sig').replace('\r\n','\n').splitlines(True),new.decode('utf-8-sig').replace('\r\n','\n').splitlines(True),fromfile='a/'+rel if rel in baseline else '/dev/null',tofile='b/'+rel))
            except UnicodeDecodeError:pass
(OUT/'changes.patch').write_text(''.join(patch),encoding='utf-8')

code_files=['Cargo.toml','setup.py','pyproject.toml','MANIFEST.in','calchem/_core.pyi','calchem/dft/d3.py','calchem/backends/native.py','calchem/workflows/opv.py',
 'crates/calchem-int/src/boys.rs','crates/calchem-int/src/symmetry.rs','crates/calchem-dft/src/d3.rs','crates/calchem-dft/src/d3_data.rs','crates/calchem-dft/src/grid.rs','crates/calchem-dft/src/xc.rs',
 'crates/calchem-scf/src/lib.rs','crates/calchem-scf/src/diis.rs','crates/calchem-scf/src/rhf.rs','crates/calchem-scf/src/uhf.rs','crates/calchem-scf/src/rks.rs','crates/calchem-scf/src/uks.rs',
 'crates/calchem-grad/src/lib.rs','crates/calchem-opt/src/lib.rs','crates/calchem-opt/src/prfo.rs','crates/calchem-opt/src/irc.rs','crates/calchem-solv/src/scrf.rs','crates/calchem-solv/src/gradient.rs','crates/calchem-solv/src/solvent.rs',
 'crates/calchem-vib/src/masses.rs','crates/calchem-vib/src/thermo.rs','crates/calchem-vib/src/modes.rs','crates/calchem-vib/src/hessian.rs','crates/calchem-py/src/lib.rs']
chunks=['# 完整 Drop-in 文件 — CalChem 第二轮\n\n每个代码块都是对应最终生产文件的完整内容。替换前先做 SHA256 备份并在独立候选目录复核，不能覆盖用户的新修改。二进制标准 D3 数据及原始表/许可证在 replacements.zip 的 data/d3 路径内；不可只替换 d3.rs 而遗漏其数据依赖。此文件为最终源码的自动导出，没有伪代码、省略号替代实现或模拟返回值。\n\n']
for rel in code_files:
    p=R/rel;lang={'rs':'rust','py':'python','pyi':'python','toml':'toml'}.get(p.suffix[1:],'text')
    chunks.append(f'## `{rel}`\n\nSHA256: `{sha(p)}`\n\n```{lang}\n'+p.read_text(encoding='utf-8-sig')+'\n```\n\n')
(OUT/'06_DROPIN_CODE.md').write_text(''.join(chunks),encoding='utf-8')

raw=B/'external_round2'
with zipfile.ZipFile(OUT/'external_raw_round2.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(raw.rglob('*')):
        if p.is_file() and p.suffix in ('.gjf','.out','.log','.txt','.inp'):
            z.write(p,str(p.relative_to(raw)).replace('\\','/'))
with zipfile.ZipFile(OUT/'external_raw_round2.zip') as z:assert z.testzip() is None

manifest={'final_files':now,'changed_since_original_round2_baseline':changed,'missing_original_files':missing,
 'restored_to_new_candidate':str(restore),'restored_files':len(restored),'all_restored_hashes_match':all(restored.values()),
 'original_snapshot_sha256':sha(A/'rollback/source_snapshot.zip'),'resume_snapshot_sha256':sha(B/'resume_1429/resume-source.zip'),
 'deployed_extension_sha256':sha(R/'calchem/_core.pyd'),
 'replacements_sha256':sha(OUT/'replacements.zip'),'external_raw_zip_sha256':sha(OUT/'external_raw_round2.zip')}
(OUT/'final_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'changed':len(changed),'baseline_files':len(baseline),'final_files':len(now),'missing':missing,'restored':len(restored),'extension_sha256':manifest['deployed_extension_sha256']}))
