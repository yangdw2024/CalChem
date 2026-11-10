import hashlib
import json
import pathlib
import subprocess
import zipfile

ROOT = pathlib.Path(r'E:\YDW\Trae_Gaussian')
OUT = ROOT / 'docs/audits/2026-11-09_redteam'
for part in ('evidence', 'rollback', 'runtime'):
    (OUT / part).mkdir(parents=True, exist_ok=True)
paths = []
for name in ('crates', 'calchem', 'tests', 'tools', 'config', '.github'):
    paths.extend(p for p in (ROOT / name).rglob('*') if p.is_file()
                 and '__pycache__' not in p.parts and p.suffix != '.pyc')
paths.extend(p for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.toml', '.lock', '.md', '.yaml', '.yml'))
files = {str(p.relative_to(ROOT)).replace('\\', '/'): {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(set(paths))}
(OUT / 'evidence/source_before.json').write_text(json.dumps(files, indent=2), encoding='utf-8')
with zipfile.ZipFile(OUT / 'rollback/source_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as z:
    for path in files:
        z.write(ROOT / path, path)
for label, args in (
    ('git-status-before.txt', ['git', 'status', '--porcelain=v1', '-uall']),
    ('git-log.txt', ['git', 'log', '-20', '--date=iso-strict', '--format=%H %ad %s']),
    ('toolchain.txt', ['rustc', '-Vv']),
):
    r = subprocess.run(args, cwd=ROOT, capture_output=True)
    (OUT / 'evidence' / label).write_bytes(r.stdout + r.stderr)
inventory = {str(p.relative_to(ROOT)): len(p.read_text(encoding='utf-8').splitlines()) for p in paths if p.suffix in ('.py', '.rs')}
(OUT / 'evidence/source_inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
print(json.dumps({'snapshot_files': len(files), 'source_lines': sum(inventory.values()), 'snapshot_bytes': (OUT / 'rollback/source_snapshot.zip').stat().st_size}, indent=2))
