# CalChem — Professional Keywords

> 给接手的 AI 在 system prompt / 用户 prompt 里锚定领域。**直接复制粘贴即可**。
> 分 5 组：领域、算法、软件、应用、项目专属。

---

## 1. 领域锚定（Domain）

```
quantum chemistry, ab initio computational chemistry,
electronic structure theory, Hartree-Fock, post-Hartree-Fock,
density functional theory (DFT),
organic photovoltaics (OPV), organic solar cells (OSC),
conjugated molecules, donor-acceptor systems,
Gaussian 16 alternative / reimplementation,
reference energy regression, numerical correctness
```

## 2. 算法关键词（Methods / Algorithms）

### 波函数方法
```
Restricted Hartree-Fock (RHF), Unrestricted Hartree-Fock (UHF),
Restricted Open-Shell Hartree-Fock (ROHF),
Self-Consistent Field (SCF), DIIS (Direct Inversion in Iterative Subspace),
initial guess from core Hamiltonian,
MP2 (Møller–Plesset perturbation theory, 2nd order),
MP4, CCSD (Coupled-Cluster Singles and Doubles),
CCSD(T) ("gold standard"), CASSCF (Complete Active Space SCF),
CCD, CIS (Configuration Interaction Singles)
```

### 基组与积分
```
Gaussian Type Orbitals (GTO), basis sets: STO-3G, 3-21G, 6-31G(d),
6-311G(d,p), def2-SVP, def2-TZVP, cc-pVDZ, cc-pVTZ, d95,
one-electron integrals (overlap S, kinetic T, nuclear attraction V_N),
two-electron integrals / electron repulsion integrals (ERI),
Obara-Saika recursion, VRR (Vertical Recurrence Relations),
spherical harmonics transformation, density fitting / resolution of identity (RI),
Schwarz screening, direct SCF
```

### DFT
```
exchange-correlation functionals: LDA (VWN5), GGA (PBE, BLYP),
meta-GGA (TPSS), hybrid (B3LYP, PBE0),
D3 dispersion correction (Grimme),
Lebedev quadrature grid, Becke partitioning,
Vxc numerical quadrature, Fock matrix assembly
```

### 优化与性质
```
analytical gradients, numerical Hessian, geometry optimization,
BFGS optimizer, Berny optimizer, IRC (Intrinsic Reaction Coordinate),
transition state search, normal mode / vibrational frequency analysis,
Mulliken population analysis, RESP charges, CM5 charges,
multipole moments, cube file output (HOMO/LUMO visualization)
```

### 其他
```
CIS, TDDFT (linear-response time-dependent DFT), oscillator strength,
PCM, SMD solvation models,
ONIOM (hybrid QM/QM),
GROMACS topology generation, GAFF force field input,
excited states, UV-Vis spectra
```

## 3. 软件栈关键词（Software Stack）

```
Python 3.10+ (developed on 3.13), numpy, scipy, h5py,
PyYAML, click (CLI framework), rich (terminal output), pydantic,
pytest, pytest-cov, pytest-xdist,
ruff (lint), mypy (type checker), pre-commit,
PyQt6 (GUI), matplotlib (plots), pyqtgraph, PyOpenGL,
PyInstaller (Windows EXE packaging),
CuPy / CUDA (GPU acceleration, planned but stub),
GitHub Actions (CI, not actually running)
```

## 4. 应用场景关键词（Use Cases）

```
organic solar cell (OSC) donor/acceptor molecules,
PM6 polymer, ClFFCl small molecule acceptor,
HOMO/LUMO energy level alignment,
molecular dynamics force field preparation,
RESP charge fitting for GROMACS,
transition dipole / oscillator strength for UV-Vis,
end-group halogenation effects,
conjugated segment geometry optimization
```

## 5. 项目专属关键词（Project-Specific）

```
CalChem, Trae_Gaussian,
from-scratch implementation (NOT a PySCF wrapper),
single-developer project,
Trae AI IDE as pair-programming tool,
Windows-native (D:\YDW\Trae_Gaussian),
Gaussian 16 (D:\G16W) as numerical reference oracle,
numerical regression threshold: |ΔE| < 1e-8 Hartree,
reference molecules: H2, He, Ne, H2O, CH3 (all STO-3G),
PyQt6 GUI lives at D:\YDW\CalChem\gui (separate directory),
embedded AI workflow framework: trae_framework.py
```

---

## 给接手 AI 的 Prompt 片段（可直接复制）

```
You are taking over the CalChem project, a from-scratch Python
quantum chemistry engine targeting organic photovoltaics (OPV).
It implements RHF/UHF, DFT (LDA/PBE/TPSS/B3LYP), MP2/MP4/CCSD/
CCSD(T)/CASSCF, geometry optimization, and RESP/GROMACS integration.
Numerical correctness against Gaussian 16 is the hard constraint:
all validated energies (H2/He/Ne/H2O on STO-3G) must remain within
1e-8 Hartree after any refactor. The project currently has no git
history, no working CI, a stub GPU module, and hardcoded Windows
paths. Refactor without breaking the numerical baseline.
```
