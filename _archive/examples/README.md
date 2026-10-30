# CalChem 使用示例

## 1. 单点能计算

### 命令行方式

```bash
calchem run water.xyz --method b3lyp --basis def2-svp
```

### Python API 方式

```python
from calchem import Molecule
from calchem.api import single_point

mol = Molecule.from_xyz_file("water.xyz")
result = single_point(mol, method="B3LYP", basis="def2-SVP")
print(f"Energy: {result.energy:.6f} Hartree")
```

## 2. 几何优化

```python
from calchem.api import optimize

result = optimize(mol, method="B3LYP", basis="def2-SVP", max_cycles=50)
print(f"Optimized energy: {result.energy:.6f} Hartree")
result.molecule.save_xyz("water_opt.xyz")
```

## 3. 频率分析

```python
from calchem.api import frequency

result = frequency(mol, method="B3LYP", basis="def2-SVP")
print(f"ZPE: {result.zpe:.6f} Hartree")
for i, freq in enumerate(result.frequencies):
    print(f"  Mode {i+1}: {freq:.2f} cm^-1")
```

## 4. RESP 电荷拟合 + GROMACS 拓扑

```python
from calchem.api import resp_charges
from calchem.md.gromacs_io import write_gromacs_topology

charges = resp_charges(mol, method="B3LYP", basis="def2-TZVP")
write_gromacs_topology(mol, charges, output_dir="gromacs_files")
```

## 5. TDDFT 激发态

```python
from calchem.excited.tddft import TDDFT

calc = TDDFT(mol, xc="b3lyp", basis="def2-SVP", nstates=10)
result = calc.run()
for i, exc in enumerate(result.excitations):
    print(f"S{i+1}: {exc.energy:.3f} eV, f={exc.oscillator_strength:.4f}")
```

## 6. 批量计算 (YAML 配置)

```yaml
# batch_calc.yaml
jobs:
  - name: water_b3lyp_svp
    molecule: water.xyz
    method: B3LYP
    basis: def2-SVP
    type: optimize
  - name: water_b3lyp_tzvp_freq
    molecule: water_opt.xyz
    method: B3LYP
    basis: def2-TZVP
    type: frequency
```

```bash
calchem batch batch_calc.yaml
```
