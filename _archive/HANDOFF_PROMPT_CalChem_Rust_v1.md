# CalChem v1.0 — Rust Quantum Chemistry Engine: Development Handoff Prompt

## Role
You are a senior computational chemistry software architect. I need you to design the complete development roadmap and module-by-module task specification for CalChem, a Rust-based high-performance quantum chemistry engine intended to outperform Gaussian, ORCA, PySCF, and REST.

## Project Identity
CalChem is NOT a Gaussian clone. It is a unified quantum chemistry platform with a Rust-native computational core, Python (PyO3) bindings, and MolForge as the front-end molecular workbench. Gaussian/ORCA/PySCF are optional external backends, not the core. The goal is a native engine that beats existing software on speed, numerical accuracy, and developer ergonomics.

## Current State (Stage 0 + Stage 1 complete)

### What works NOW (verified numerically)
- **Rust workspace**: 9 crates (`calchem-model`, `calchem-basis`, `calchem-tensor`, `calchem-int`, `calchem-scf`, `calchem-grad`, `calchem-prop`, `calchem-opt`, `calchem-py`)
- **Basis loading**: STO-3G loaded from embedded JSON, builds contracted GTO shells for arbitrary molecules
- **s-type integrals**: Overlap, kinetic, nuclear attraction, and electron repulsion integrals for s-shells, matching Python reference to 1e-11 Eh
- **RHF SCF**: Basic SCF iteration (core guess, no DIIS) converges for closed-shell s-type molecules
- **PyO3 bridge**: `calchem._core.run_single_point(atomic_numbers, coords_bohr, charge, multiplicity, method, basis)` returns real energies

### Numerical baseline (golden values)
| Molecule | Method | Basis | Energy (Eh) | Current error |
|---|---|---|---|---|
| H2 (1.4 Bohr) | RHF | STO-3G | -1.116714325063 | 4.5e-13 |
| He | RHF | STO-3G | -2.8077839575 | 4.0e-11 |
| H2O (Szabo geom) | RHF | STO-3G | -74.6953191771 | ~2.8 (p-type broken) |

### What is BROKEN / missing
1. **p-type integrals**: Nuclear attraction for p-shells uses a rough center-offset approximation (not proper Obara-Saika VRR/HRR). ERI for p-type is not implemented (only ssss).
2. **No DIIS**: SCF uses plain core guess + direct iteration. Convergence is slow for larger systems.
3. **No analytical gradients**: Only finite-difference available (in Python reference).
4. **Basis set library**: Only STO-3G embedded. 6-31G JSON exists but not loaded.
5. **No DFT**: PBE/B3LYP grid and XC functional not implemented in Rust.
6. **No parallelism**: No Rayon parallelization in integral or SCF loops.
7. **No RI/DF**: Density fitting not implemented.
8. **Tensor layer**: Jacobi eigendecomposition is O(n³) but not optimized. No BLAS/LAPACK linkage.

### Hardware target
- AMD Ryzen 7 9700X (8c/16t, Zen 5, AVX-512)
- 15.1 GB RAM
- Windows 11, MSVC toolchain
- Rust stable with `-C target-cpu=znver5 -C target-feature=+avx512f,+fma`

### Existing Python reference engine (v0.6.0)
The Python reference at `calchem/` (scf/hf.py, integrals/) serves as the numerical ground truth. It has full RHF/UHF, finite-difference gradients, geometry optimization (BFGS), numerical Hessian/frequencies, Gaussian/ORCA/PySCF/xTB/REST backends, and a PyQt6 GUI. This is the fallback path and the validation target.

## What I need from you

### 1. Architecture review
Review the current crate structure and recommend:
- Should we link to BLAS/LAPACK (via `faer` or `ndarray-linalg` + OpenBLAS) for eigendecomposition and matrix multiply?
- Should we implement our own Obara-Saika integral recursion or wrap a C library (libcint failed to compile on Windows MSVC)?
- What is the optimal module boundary for: integrals → SCF → gradients → properties?

### 2. Stage 2: Complete RHF (p-type + DIIS)
Give me file-by-file, function-by-function implementation tasks:
- Proper Obara-Saika VRR for overlap, kinetic, nuclear (s, p, d shells)
- ERI for ssss, spss, pppp (all 8 permutational symmetry classes)
- DIIS (direct inversion in iterative subspace) with error metric FDS-SDFD
- Convergence criteria and damping strategy
- Validation: H2O/STO-3G must match -74.6953191771 Eh to 1e-8

### 3. Stage 3: Analytical gradients
- RHF nuclear gradients: derivative of overlap, kinetic, nuclear, ERI
- Hellmann-Feynman vs derivative integral approach
- Geometry optimization (BFGS) in Rust
- Validation: H2 optimized bond length must match reference

### 4. Stage 4: Performance
- Rayon parallelization for integral shell loops
- Schwarz screening for ERI
- SIMD (AVX-512) for matrix operations
- Memory pool for temporary arrays
- Target: 10x faster than Python reference for H2O/STO-3G

### 5. Stage 5+: DFT, TDDFT, post-HF
- Recommended order and dependencies
- Minimum viable DFT (LDA → PBE → B3LYP)
- Grid construction (Lebedev-Laikov)
- What post-HF methods are feasible first (MP2 vs CCSD)

### 6. Build system
- How to properly link to Python (currently manual .pyd copy)
- How to bundle basis data files (currently include_str!)
- Cross-compilation considerations

## Constraints
- Do NOT use libcint (C compilation failed on Windows MSVC)
- Do NOT use heavy Python dependencies in the Rust core
- Every integral must be validated against the Python reference before merging
- Numerical tolerance: 1e-8 Eh for energies, 1e-6 Eh/Bohr for gradients
- The Python reference engine must remain working as fallback (dual-track)

## Deliverable format
For each stage, give me:
1. List of files to create/modify with exact paths
2. Function signatures with types
3. Algorithm pseudocode for each complex function
4. Validation test cases (molecule → expected value)
5. Estimated complexity (Big-O)
6. Dependencies between tasks (what must be done first)
