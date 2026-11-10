# CalChem v1.0 — Rust Quantum Chemistry Engine: Handoff Prompt v3 (Stage 2 Complete)

## Role
You are a senior computational chemistry software architect. CalChem Stage 2 is now numerically verified. Design Stage 3: analytical gradients + geometry optimization in Rust.

## Current State (Stage 2 VERIFIED)

### What works NOW (verified against golden values)
| Molecule | Method | Basis | Energy (Eh) | Error | Status |
|---|---|---|---|---|---|
| H2 (1.4 Bohr) | RHF | STO-3G | -1.1167143251 | 4.3e-13 | PASS |
| He | RHF | STO-3G | -2.8077839575 | 4.0e-11 | PASS |
| H2O (Szabo) | RHF | STO-3G | -74.6953191142 | 6.3e-08 | PASS |

### Architecture (9 crates, modular)
- `calchem-model`: Molecule/Atom/CalcError/ScfWavefunction/ScfOptions (thiserror)
- `calchem-basis`: STO-3G + 6-31G embedded JSON, ShellType S/P/D, proper normalization
- `calchem-tensor`: Matrix2D/Tensor4D + Jacobi eigh_sym (O(n³))
- `calchem-int`: boys.rs, primitive.rs (GaussianPair, overlap_1d, kinetic_primitive, nuclear_vrr), one_electron.rs (shell blocks), eri.rs (shell quartet + 8-fold symmetry), symmetry.rs
- `calchem-scf`: rhf.rs (full RHF with DIIS), diis.rs (Pulay commutator error)
- `calchem-py`: PyO3 bridge `run_single_point(symbols, geometry_bohr, charge, mult, method, basis)`
- `calchem-grad`, `calchem-prop`, `calchem-opt`: stubs

### Key implementation details
- **nuclear_vrr**: recursive Obara-Saika VRR, a-side recursion first, b-side second, m=0..max_m Boys values precomputed per atom
- **overlap_1d**: standard Obara-Saika 1D recurrence with `(pi/gamma)^1/2` base factor
- **kinetic_primitive**: second-derivative identity on ket: `T = -1/2[b(b-1)S(a,b-2) - 2β(2b+1)S(a,b) + 4β²S(a,b+2)]`
- **ERI**: shell quartet computation with 8 permutational symmetry classes
- **DIIS**: Pulay FDS-SDF error metric, up to 8 vectors, Gaussian elimination

### Python dual-track backend
- `calchem.backends.native.NativeCalchem` automatically routes RHF/STO-3G to Rust
- All other methods/bases fall back to Python reference engine
- Rust version string: `rust-0.2.0`

### Known gaps
1. **No analytical gradients** — only finite-difference available
2. **No geometry optimization in Rust** — Python reference has BFGS
3. **Jacobi eigendecomposition** — not BLAS-accelerated, O(n³) but slow
4. **No parallelism** — no Rayon usage yet (even though rayon is in workspace deps)
5. **No Schwarz screening** — all ERI computed densely
6. **6-31G basis** — loaded but not numerically validated
7. **No UHF** — closed-shell only
8. **No DFT** — no XC functional or grid

## What I need from you (Stage 3)

### 1. Analytical RHF gradients
Derive and implement:
- Nuclear gradient of overlap, kinetic, nuclear attraction, and ERI integrals
- Hellmann-Feynman contribution (electron-nuclear attraction derivative)
- Pulay force contribution (derivative of AO overlap w.r.t. nuclear coordinates)
- API: `calchem-grad/src/lib.rs` with `rhf_gradient(mol, basis, wf) -> Vec<[f64;3]>`
- Validation: H2 optimized bond length must match reference (0.741 Å ≈ 1.401 Bohr)
- Validation: H2O bond angle/length gradients must match finite-difference to 1e-4 Eh/Bohr

### 2. Geometry optimization in Rust
- BFGS optimizer (or quasi-Newton)
- Convergence criteria: max force < 1e-4 Eh/Bohr, RMS force < 5e-5, max displacement < 1e-3 Bohr
- API: `calchem-opt/src/lib.rs` with `optimize_geometry(mol, basis, method) -> Molecule`
- Validation: H2O optimized geometry must match reference (O-H ~0.958 Å, H-O-H ~104.5°)

### 3. Performance Stage 3
- Rayon parallelization for shell integral loops
- Schwarz screening threshold for ERI (1e-10)
- Cache-friendly matrix multiply (tiled or row-major)
- Target: H2O/STO-3G single point < 10 ms in release mode

### 4. 6-31G validation
- H2O/6-31G reference energy (should be ~-76.0 Eh)
- Verify basis loading produces correct AO count and normalization

## Constraints
- Do NOT link to BLAS/LAPACK yet (keep pure Rust Jacobi)
- Do NOT implement DFT yet
- Every gradient must be validated against finite-difference from Python reference
- Numerical tolerance: 1e-6 Eh for energies, 1e-4 Eh/Bohr for gradients
- Python dual-track must remain working
- Keep the modular crate structure; don't put gradients in calchem-int

## Deliverable format
For each component, give:
1. Exact file paths to create/modify
2. Function signatures with types
3. Algorithm pseudocode
4. Validation test cases
5. Dependencies between tasks
