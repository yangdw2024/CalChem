# Native Scientific Method Boundaries — 2026-11-24

This document records the exact verified scope, approximations, defaults, and known limitations of the native pure Rust (+ optional CUDA) CalChem engine in `v0.2.3`.

---

## 1. Architecture & Packaging

- CalChem is a pure Rust workspace (`calchem-cli` native executable) with optional FP64 CUDA acceleration (`calchem-cuda`) and an optional zero-Python Node.js CLI wrapper (`npm/`).
- CalChem ships strictly as a native binary (`calchem_cli`) with no Python extension bindings; `scripts/verify_calchem_job.py` and external benchmark/diagnostic drivers invoke `calchem_cli` directly via subprocess.

---

## 2. Electronic Structure Methods & Basis Sets

- **Supported SCF references**: Gas-phase and SCRF `RHF`, `UHF`, `RKS`, `UKS`.
- **Supported DFT functionals (`XcFunctional`)**: `SVWN5`, `BLYP`, `PBE`, `B3LYP`, `B3LYP5`, `PBE0`, `CAM-B3LYP`, `wB97X`, `wB97X-D` (using native exact and erf/erfc range-separated ERIs in `calchem-int`).
- **UHF / UKS caveats**:
  - Unrestricted calculations report `<S^2>` without spin projection or general SCF stability analysis.
  - Convergence establishes a stationary SCF solution, not necessarily the global electronic ground state.
  - **Open-shell `UB3LYP` local correlation formula**: CalChem implements `B3LYP` with Vosko-Wilk-Nusair Formula V (`VWN_RPA`, Stephens canonical form), whereas Gaussian 16 defaults to VWN Formula III (`VWN3`). Closed-shell `RB3LYP` and relative differences largely cancel, while open-shell `UB3LYP` absolute gas-phase energies differ from Gaussian 16 by **~0.14 kcal/mol** (`-0.1365 kcal/mol` measured on `CH3· / 6-31G(d)`).
- **Embedded basis sets**: `sto-3g`, `6-31g`, `6-31g(d)`, `6-31g(d,p)`, `lanl2dz`, and `lanl2dzdp`. Shell functions use **Cartesian `6D/10F`** conventions (`ShellType::D = 6`, `ShellType::F = 10`), which affects total AO counts relative to spherical `5D/7F` engines whenever explicit `F` functions appear.

---

## 3. DFT Numerical Quadrature & Dispersion

- **DFT grids**: Full radial mapping without fixed radial cutoffs (`Coarse`, `Medium`, `Fine`, `UltraFine`), with automatic `Fine` expansion for diffuse/small-exponent shells. Closed-shell CLI/SCRF uses the explicitly selected `--grid` (`Coarse` default if unspecified in CLI; `Fine` in `run_scrf_rks`); open-shell `run_scrf_unrestricted` defaults to `Medium` unless `--grid` is passed via `run_scrf_unrestricted_with_grid`. Note that Gaussian 16 defaults to `UltraFine` when comparing DFT energies unless explicitly matched.
- **Dispersion**:
  - `--disp d3bj` and `D3Model` implement **two-body Grimme D3(BJ)/D3(0)** with analytical nuclear gradients across `energy`, `grad`, `opt`, and `freq` (ATM three-body dispersion is excluded).
  - `wB97X-D` uses built-in Chai-Head-Gordon D2 dispersion (`wb97xd_chg_d2_energy_and_gradient`) consistent across energies and gradients; combining `wB97X-D` with `--disp` is explicitly rejected.

---

## 4. RI-J Density Fitting (`--rij`)

- **Auxiliary basis choices**:
  - `--aux-basis auto-jfit` (default): per-angular-momentum AutoAux construction with 2-center Coulomb Cholesky/SPD factorization (`trunc == 0` full rank verified on CHNOSF polarized sets).
  - `--aux-basis def2-jfit-f`: embedded MolSSI BSE `def2-universal-jfit` up to Cartesian F-shells (`L <= 3`).
- **Verified accuracy vs known minimal-basis boundary**:
  - On polarized `6-31G(d)` and `6-31G(d,p)` sets, `auto-jfit` matches Exact 4c energies within **`< 0.1 kcal/mol`** (`CH3CN`: `-0.030`, `C2H3F`: `-0.028`, `C4H4S`: `-0.087`, Caffeine: `-0.094 kcal/mol`) and analytical gradients within `9.4e-6 – 4.5e-5 Eh/Bohr`.
  - **Minimal basis `STO-3G` boundary**: On `IC-2F / STO-3G` (22 atoms, 94 AO, `B3LYP/Coarse`), RI-J introduces a **`-4.08 kcal/mol`** error (`auto-jfit`) or **`-4.58 kcal/mol`** error (`def2-jfit-f`) relative to Exact 4c. Exact 4c (`--direct`, without `--rij`) should be used for minimal-basis accuracy checks or benchmark ground truth.

---

## 5. CUDA GPU Acceleration Scope (`--gpu`)

- `calchem-cuda` accelerates:
  1. Packed RI-J 3-center integral evaluation **only when all primary and auxiliary shells satisfy `L <= 2` (`S/P/D`)** and tensor size fits the packed GPU cache criteria.
  2. RI-J tensor contractions (`contract_v` and `contract_coulomb`) on resident or host-mirrored tensors.
  3. DFT numerical XC grid quadrature (`V_xc`).
- **Explicit CPU fallbacks**:
  - Polarized basis sets generate F-type auxiliary shells (`L_aux = 3`) in both `auto-jfit` and `def2-jfit-f`; **3-center integral tensor construction falls back to CPU** and increments `gpu_fallback_invocations`.
  - Hybrid exact Hartree-Fock exchange $K$, all range-separated exchange, TD-DFT response sweeps, and all analytical nuclear gradients run **on CPU** even when `--gpu` is set.

---

## 6. TD-DFT Excited States

- Native `calchem-tddft` supports restricted closed-shell references (`RHF` / `RKS`, including `B3LYP`, `PBE0`, `CAM-B3LYP`, `wB97X-D`) in both TDA (`--td-mode tda`) and Full-LR/RPA (`--td-mode lr`) for `singlet`, `triplet`, or `both`, via stored ERIs or **AO-Direct** contractions (`--direct`).
- RI-J references, unrestricted (`UHF`/`UKS`) references, and excited-state nuclear gradients are rejected cleanly before SCF/TD-DFT.
- **Performance boundary**: On 24-atom Caffeine (`B3LYP/6-31G(d,p)`, `Fine`, 260 AO, 8 CPU threads), AO-Direct Full-LR (5 singlet roots) matches Gaussian 16 within `0.00006–0.00025 eV` (`max |Δω| = 0.000246 eV`), but executes 3 unbatched 4c ERI sweeps per trial vector on CPU and required **`5119.84 s` (~85.3 min)** (`5119.86 s` manifest) vs Gaussian 16's `192.2 s` (**~26.6× slower**, verified on `C00AB91F`).

---

## 7. Continuum Solvation (SCRF) & Gradients

- **Models**: `C-PCM`, `COSMO`, `IEF-PCM` (York-Karplus spherical-Gaussian regularized boundary operator `S_{ij} = erf(zeta_{ij} r_{ij})/r_{ij}`), `SMD` (`IEF-PCM` on `SmdIntrinsic` radii + parameterized Minnesota SMD-CDS), and `Onsager`.
- **Defaults vs Gaussian 16**:
  - `C-PCM`, `IEF-PCM`, and `COSMO` default to **`Bondi (alpha = 1.20)`** atomic radii, pure electrostatics (`include_cds = false`), and unrenormalized surface charges (`renormalize_charge = false`) — unlike Gaussian 16, which defaults to **`UFF (alpha = 1.10)`** for standard PCM. When matching CalChem PCM against Gaussian 16, specify `Radii=Bondi Alpha=1.20`.
  - Known non-aqueous CDS edge case: `ethanol` in `DMSO` under `SMD` produces `dG_CDS = -0.030 kcal/mol` vs Gaussian 16 `+0.140 kcal/mol` (total $\Delta G_{\text{solv}}$ diff `-0.150 kcal/mol`).
- **Gradients**:
  - CLI SCRF (`--solv ...`) is restricted to `--task energy`; solvent `grad`, `opt`, `freq`, and `tddft` are rejected at CLI parsing rather than silently dropping solvent terms.
  - Library function `calchem_solv::compute_pcm_gradient_fd` evaluates **fixed-AO-density central finite differences** on the exact same `ReactionFieldSolver` + `compute_cds_energy` surface as `scrf.rs` (it excludes orbital density response, and discrete unsmoothed Lebedev cavity areas are piecewise constant away from sphere-intersection switches). Analytic PCM gradients remain explicitly unavailable (`UnsupportedMethod`).

---

## 8. Gradients, Geometry Optimization, & Harmonic Frequencies

- **Gradients**: Analytical gas-phase nuclear gradients (`calchem-grad`) support RHF/UHF/RKS/UKS, ECPs, RI-J (`auto-jfit` / `def2-jfit-f`), range-separated exchange, and two-body D3/wB97X-D dispersion.
- **Optimization**: Native `calchem-opt` implements Cartesian BFGS / P-RFO with wavefunction guess reuse (`GasPoint`) and atomic checkpoint/resume (`--checkpoint`, `--resume`). Redundant internal coordinates (DLC) are not yet implemented.
- **Frequencies**: `calchem-vib` computes harmonic Hessians numerically from central differences of analytical gradients, projects rigid translations/rotations, supports `--freq-jobs N` concurrency control and per-displacement atomic caches (`--checkpoint <dir>`), and checks strict positive-definiteness / imaginary modes cleanly.

---

## 9. Memory & Resource Admission Bounds

- Default admission estimates (overridable via environment variables, not OS-enforced cgroup/allocator locks):
  - `CALCHEM_MAX_ERI_BYTES = 4 GiB` (automatic switch from Stored to Direct SCF when `opts.direct` is unset; explicit `--stored` above budget returns `ResourceLimit`).
  - `CALCHEM_MAX_MEMORY_BYTES = 8 GiB` (Direct SCF and RI-J metric working-memory estimate).
  - `CALCHEM_MAX_DF_BYTES = 4 GiB` (cached/packed RI-J 3-center tensor limit; falls back to screened on-the-fly path above threshold).
- **Large-molecule status**: Polarized end-to-end benchmarks above 48 atoms (72–726 atoms) remain unverified and are queued for Phase 5.
