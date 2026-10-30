# CalChem v1.0 — Rust Quantum Chemistry Engine: Development Handoff Prompt v2

## Role
You are a senior computational chemistry software architect. Design the complete development roadmap and module-by-module task specification for CalChem, a Rust-native high-performance quantum chemistry engine intended to outperform Gaussian, ORCA, PySCF, and REST.

## Project Identity
CalChem is a unified quantum chemistry platform with a Rust-native computational core, Python (PyO3) bindings, and MolForge as the front-end molecular workbench. Gaussian/ORCA/PySCF are optional external backends. The goal is a native engine that beats existing software on speed, numerical accuracy, and developer ergonomics. OPV (organic solar cells) is dogfooding domain, not a boundary.

## Current State (Stage 1 + 1.5 complete)

### What works NOW (verified numerically)
- **Rust workspace**: 9 crates (`calchem-model`, `calchem-basis`, `calchem-tensor`, `calchem-int`, `calchem-scf`, `calchem-grad`, `calchem-prop`, `calchem-opt`, `calchem-py`)
- **Basis loading**: STO-3G loaded from embedded JSON, builds contracted GTO shells for arbitrary molecules
- **s-type integrals**: Overlap, kinetic, nuclear attraction, and ERI for s-shells, matching Python reference to 1e-11 Eh
- **RHF SCF**: Core guess + DIIS convergence acceleration (FDS-SDF error metric, up to 8 vectors)
- **PyO3 bridge**: `calchem._core.run_single_point()` returns real energies
- **Dual-track backend**: `calchem.backends.native.NativeCalchem` automatically uses Rust for H/He RHF/STO-3G, falls back to Python reference for everything else

### Numerical baseline (golden values)
| Molecule | Method | Basis | Energy (Eh) | Current error |
|---|---|---|---|---|
| H2 (1.4 Bohr) | RHF | STO-3G | -1.116714325063 | **4.5e-13** |
| He | RHF | STO-3G | -2.8077839575 | **4.0e-11** |
| H2O (Szabo geom) | RHF | STO-3G | -74.6953191771 | ~2.8 (p-type broken) |

### What is BROKEN / missing (prioritized)
1. **p-type integrals (CRITICAL)**: Nuclear attraction for p-shells uses rough center-offset approximation. ERI for p-type is zero. This blocks any molecule with p-block atoms (O, C, N, etc.). Need proper Obara-Saika VRR/HRR recurrence.
2. **No analytical gradients**: Only finite-difference available.
3. **Basis set library**: Only STO-3G embedded. 6-31G JSON exists but not loaded.
4. **No DFT**: PBE/B3LYP grid and XC functional not implemented.
5. **No parallelism**: No Rayon parallelization in integral or SCF loops.
6. **No RI/DF**: Density fitting not implemented.
7. **Tensor layer**: Jacobi eigendecomposition is O(n³) but not optimized. No BLAS/LAPACK linkage.
8. **No UHF**: Only closed-shell RHF.
9. **No geometry optimization in Rust**: Python reference has BFGS.

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
- Should we implement our own Obara-Saika integral recurrence or wrap a C library (libcint failed to compile on Windows MSVC)?
- What is the optimal module boundary for: integrals → SCF → gradients → properties?

### 2. Stage 2: Complete RHF (p-type + DIIS + more basis)
Give me file-by-file, function-by-function implementation tasks:
- Proper Obara-Saika VRR for overlap, kinetic, nuclear (s, p, d shells)
- ERI for ssss, spss, pppp (all 8 permutational symmetry classes)
- Load 6-31G basis from embedded JSON
- Validation: H2O/STO-3G must match -74.6953191771 Eh to 1e-8
- Validation: CH4/STO-3G must match reference to 1e-6

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
- p-type VRR must be correct — the current forward VRR attempt had a sign bug causing ~30 Eh error

## Deliverable format
For each stage, give me:
1. List of files to create/modify with exact paths
2. Function signatures with types
3. Algorithm pseudocode for each complex function
4. Validation test cases (molecule → expected value)
5. Estimated complexity (Big-O)
6. Dependencies between tasks (what must be done first)
