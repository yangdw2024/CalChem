import json
import pathlib
import re

ROOT=pathlib.Path(r'E:\YDW\Trae_Gaussian')
patterns={
 'unsafe_or_leak':r'\bunsafe\b|static mut|Box::leak|mem::forget',
 'native_external_or_simd':r'Command::|std::arch|simd|use faer|faer::',
 'synchronization':r'\bMutex\b|\bRwLock\b|\bArc\b|OnceLock',
 'molecular_identity':r'(?i)water|benzene|ic.?2f|bdd1|pm6|\by6\b',
 'literal_drives':r'[CDE]:[/\\]',
 'placeholder':r'(?i)fake|hardcod|dummy|placeholder|unimplemented!|todo!',
 'absolute_energy_literal':r'-(?:74|75|76|881|886|2378|2409|3451|3469)\.[0-9]{3}',
}
result={'files':{},'hits':{k:[] for k in patterns}}
for folder in ['crates','calchem','tests','tools','.github']:
    for p in (ROOT/folder).rglob('*'):
        if p.is_file() and p.suffix in ['.rs','.py','.toml','.yml']:
            rel=p.relative_to(ROOT).as_posix();t=p.read_text(encoding='utf-8-sig');lines=t.splitlines()
            result['files'][rel]=len(lines)
            rust_test_start=t.find('#[cfg(test)]')
            for n,line in enumerate(lines,1):
                for label,pat in patterns.items():
                    if re.search(pat,line):
                        result['hits'][label].append({'file':rel,'line':n,'text':line.strip(),'rust_test_block': p.suffix=='.rs' and rust_test_start>=0 and n>len(t[:rust_test_start].splitlines())})
result['total_lines']=sum(result['files'].values())
result['rust_crates']=sorted(p.parent.name for p in (ROOT/'crates').glob('*/Cargo.toml'))
print(json.dumps(result,ensure_ascii=False,indent=2))
