from pathlib import Path
import difflib
import hashlib
import json

audit = Path(__file__).resolve().parents[1]
root = audit.parents[2]
backup = audit/'backup'
changes, patches, manifest = [], [], {}
new_files = [
    'crates/calchem-int/tests/independent_audit.rs',
    'crates/calchem-cuda/tests/ffi_contract.rs',
    'crates/calchem-tddft/tests/independent_solver.rs',
    'crates/calchem-grad/tests/rij_provenance.rs',
]
for old in sorted(p for p in backup.rglob('*') if p.is_file()):
    relative = old.relative_to(backup)
    current = root/relative
    if not current.is_file():
        changes.append({'path':relative.as_posix(),'state':'missing'})
        continue
    before, after = old.read_bytes(), current.read_bytes()
    if before == after:
        continue
    record = {'path':relative.as_posix(), 'sha256_before':hashlib.sha256(before).hexdigest(),
        'sha256_after':hashlib.sha256(after).hexdigest(), 'state':'modified'}
    try:
        left, right = before.decode('utf-8-sig').splitlines(True), after.decode('utf-8-sig').splitlines(True)
        record['changed_line_starts'] = [j+1 for op,i,ie,j,je in difflib.SequenceMatcher(None,left,right,autojunk=False).get_opcodes() if op!='equal']
        patches.extend(difflib.unified_diff(left,right,fromfile='before/'+relative.as_posix(),tofile='after/'+relative.as_posix()))
    except UnicodeDecodeError:
        record['binary'] = True
    changes.append(record)
for name in new_files:
    current = root/name
    raw = current.read_bytes()
    changes.append({'path':name,'state':'new','sha256_after':hashlib.sha256(raw).hexdigest(),'changed_line_starts':[1]})
    patches.extend(difflib.unified_diff([],raw.decode().splitlines(True),fromfile='/dev/null',tofile='after/'+name))
for folder in ['crates','calchem','scripts','tools/dev']:
    for p in sorted((root/folder).rglob('*')):
        if p.is_file() and p.suffix in {'.rs','.cu','.toml','.py','.pyi','.ps1','.pyd'} and '__pycache__' not in p.parts:
            manifest[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
outputs = {
    'evidence/changes.json':json.dumps(changes,indent=2,ensure_ascii=False),
    'evidence/source-binary-sha256.json':json.dumps(manifest,indent=2),
    'evidence/independent-changes.patch':''.join(patches),
}
lines=['# 本次修改文件与行号\n','只对比本次 backup，不将起点已有改动算成本次贡献。格式化变化也在差异中保留。\n',
    '| 文件 | 状态 | 修改后的行号起点 |','|---|---|---|']
for c in changes:
    starts = c.get('changed_line_starts',[])
    lines.append('| `'+c['path']+'` | '+c['state']+' | '+', '.join(map(str,starts))+' |')
lines += ['\n详细差异：`evidence/independent-changes.patch`。原/新哈希：`evidence/changes.json`。源码与实际 pyd 哈希：`evidence/source-binary-sha256.json`。',
    '\nRust 新增 density_fitting 字段属于公共结构字面量 API 变化；Python 暴露只读来源字段，未实现的科学路径现明确拒绝。']
outputs['04_CHANGE_MAP.md'] = '\n'.join(lines)+'\n'
print(json.dumps(outputs,ensure_ascii=False))
