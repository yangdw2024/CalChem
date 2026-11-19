"""Read-only summary; the caller saves stdout using a native shell file writer."""
from pathlib import Path
import hashlib
import json
import re

audit = Path(__file__).resolve().parents[1]
root = audit.parents[2]
evidence = audit/'evidence'
results = {}
for name in ['build-complete','cargo-complete','pytest-complete','opv-special','cuda-runtime','opv-measured']:
    code_path = evidence/f'{name}.exit.txt'
    if not code_path.exists():
        raise SystemExit(f'Missing completion marker: {code_path}')
    code = int(code_path.read_text(encoding='utf-8-sig').strip())
    if code:
        raise SystemExit(f'Nonzero completion marker: {name}={code}')
    results[name] = {'exit_code':code}

cargo = (evidence/'cargo-complete.log').read_text(encoding='utf-8-sig',errors='replace')
counts = re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored',cargo)
results['cargo-complete'].update(passed=sum(int(x[0]) for x in counts),
    failed=sum(int(x[1]) for x in counts),ignored=sum(int(x[2]) for x in counts),suite_summaries=len(counts))
for name in ['pytest-complete','opv-special']:
    log = (evidence/f'{name}.log').read_text(encoding='utf-8-sig',errors='replace')
    count = re.findall(r'(\d+) passed',log)
    if not count:
        raise SystemExit(f'No passed count in {name}')
    results[name].update(passed=int(count[-1]),failed=0)
cuda = (evidence/'cuda-runtime.log').read_text(encoding='utf-8-sig',errors='replace')
results['cuda-runtime'].update(device_available='CUDA available=true' in cuda,
    native_kernel_executed='kernel executed=true' in cuda,
    note='Unavailable-device accuracy test returns early: framework ok does not certify GPU precision.')
measured = json.loads((evidence/'opv-measured.json').read_text(encoding='utf-8-sig'))
results['opv-measured'].update(measured)

pyd = root/'calchem/_core.pyd'
dll = root/'build_target/release/_core.dll'
hashes = {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [pyd,dll]}
if len(set(hashes.values())) != 1:
    raise SystemExit('Installed pyd != just-built DLL')
source_manifest = json.loads((evidence/'source-binary-sha256.json').read_text(encoding='utf-8-sig'))
drift = [name for name, h in source_manifest.items() if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h]
if drift:
    raise SystemExit(f'Source/binary manifest drift: {drift}')
workspace = (root/'Cargo.toml').read_text()
crates = len(re.findall(r'^\s*"crates/',workspace,re.M))
print(json.dumps({'date':'2026-11-11','workspace_crates':crates,'results':results,
    'artifact_hashes':hashes,'manifest_drift':drift,
    'limitations':['No 726/2000-atom full SCF executed','No native GPU kernel execution certified',
        'wB97X-D/SMD/RI analytic gradients/RI response/range-separated SCRF rejected, not implemented',
        'Passing tests are not full independent scientific or industrial certification']},
    ensure_ascii=False,indent=2))
