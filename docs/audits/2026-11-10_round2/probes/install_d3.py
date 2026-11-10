from pathlib import Path
import tomllib,re,shutil
BASE=Path(r'E:\YDW\Trae_Gaussian\build_target\runtime\round2_20261110')
ROOT=BASE/'repair_stage'; AUDIT=Path(__file__).parent
params=tomllib.loads((BASE/'references/parameters.toml').read_text())['parameter']
aliases={'b3lyp':'"b3lyp" | "b3lyp5"','pbe':'"pbe"','pbe0':'"pbe0"','blyp':'"blyp"','bp':'"bp86"','tpss':'"tpss"','b97d':'"b97" | "b97-d"','camb3lyp':'"cam-b3lyp" | "camb3lyp"','hf':'"hf"'}
construct=[]
for scheme,name in [('bj','d3bj'),('zero','d3zero')]:
    construct.append(f'pub fn {name}(functional:&str)->Result<Self,CalcError> {{\nlet f=functional.trim().to_ascii_lowercase();\nlet damping=match f.as_str() {{')
    for key,alias in aliases.items():
        p=params[key]['d3'][scheme]
        if scheme=='bj': fields={k:p.get(k,1.0) for k in ['s6','s8','a1','a2']}; variant='BeckeJohnson'
        else: fields={'s6':p.get('s6',1.),'s8':p['s8'],'sr6':p['rs6'],'sr8':1.,'alpha6':14.,'alpha8':16.}; variant='Zero'
        construct.append(alias+'=>DampingScheme::'+variant+'{'+','.join(f'{k}:{float(v)!r}' for k,v in fields.items())+'},')
    construct.append('_=>return Err(CalcError::UnsupportedMethod(format!("Unsupported D3 functional: {functional}"))),};\nOk(Self{damping,functional_name:f})\n}')
src=(AUDIT/'d3_evaluator.rs').read_text(encoding='utf-8-sig').replace('// PARAMETER_CONSTRUCTORS','\n'.join(construct))
(ROOT/'crates/calchem-dft/src/d3.rs').write_text(src)
shutil.copyfile(BASE/'references/parameters.toml',ROOT/'crates/calchem-dft/data/d3/parameters.toml')
p=ROOT/'crates/calchem-dft/tests/d3_dispersion.rs';s=p.read_text();s=re.sub(r'(d3\.energy_and_gradient\([^;]+\));',r'\1.unwrap();',s);p.write_text(s)
print('Generated standard two-body D3 and updated tests for Result contract')
