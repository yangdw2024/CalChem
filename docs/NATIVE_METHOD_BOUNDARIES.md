# Native Scientific Method Boundaries — 2026-11-26 (`v0.3.0`)

This document records the exact verified scope, approximations, defaults, and known limitations of the native pure Rust (+ optional CUDA `sm_120`) CalChem engine in `v0.3.0` (`calchem_cli.exe` SHA256 `C9F553BB0F96D183F34BD1E4A79563D0FAF750C33EFC3B6B9426DD908F1FD653`, `calchem_kernels_sm120.cubin` SHA256 `5A7DBBFC0908E9A40DCD2BFE5DAA34CD9C98F2EEFCA919B7A1E379A4BC30F533`).

---

## 1. Architecture & Packaging

- CalChem is a 14-crate pure Rust workspace (`calchem-cli` native executable) with optional FP64 CUDA acceleration (`calchem-cuda`) and a strict fail-fast zero-Python Node.js CLI wrapper (`@yangdw2024/calchem@0.3.0` in `npm/`).
- CalChem ships strictly as a native binary (`calchem_cli`) with no Python extension bindings or PyPI distribution; `scripts/verify_calchem_job.py` and external benchmark/diagnostic drivers invoke `calchem_cli` directly via subprocess.
- On tag `v0.3.0`, `crates/*/Cargo.toml` remain at `0.2.2` to preserve byte-for-byte source parity with the audited `C9F553BB...` binary.

---

## 2. Electronic Structure Methods & Basis Sets

- **Supported SCF references**: Gas-phase and SCRF `RHF`, `UHF`, `RKS`, `UKS`, with atomic-block SAD initial guess (`guess.rs`) and drift-bounded incremental ΔP Fock construction (`fock_builder.rs`, `scrf.rs`).
- **Supported DFT functionals (`XcFunctional`)**: `SVWN5`, `BLYP`, `PBE`, `B3LYP`, `B3LYP5`, `PBE0`, `CAM-B3LYP`, `wB97X`, `wB97X-D` (using native exact and erf/erfc range-separated ERIs in `calchem-int`).
- **UHF / UKS caveats**:
  - Unrestricted calculations report `<S^2>` without spin projection or general SCF stability analysis.
  - Convergence establishes a stationary SCF solution, not necessarily the global electronic ground state.
  - **Open-shell `UB3LYP` local correlation formula**: CalChem implements `B3LYP` with Vosko-Wilk-Nusair Formula V (`VWN_RPA`, Stephens canonical form), whereas Gaussian 16 defaults to VWN Formula III (`VWN3`). Closed-shell `RB3LYP` and relative differences largely cancel, while open-shell `UB3LYP` absolute gas-phase energies differ from Gaussian 16 by **~0.14 kcal/mol** (`-0.1365 kcal/mol` measured on `CH3· / 6-31G(d)`).
- **Embedded basis sets**: `sto-3g`, `6-31g`, `6-31g(d)`, `6-31g(d,p)`, `lanl2dz`, and `lanl2dzdp`. Shell functions use **Cartesian `6D/10F`** conventions (`ShellType::D = 6`, `ShellType::F = 10`), which affects total AO counts relative to spherical `5D/7F` engines whenever explicit `F` functions appear.

---

## 3. DFT Numerical Quadrature & Dispersion

- **DFT grids**: Full radial mapping without fixed radial cutoffs (`Coarse`, `Medium`, `Fine`, `UltraFine`), with automatic `Fine` expansion for diffuse/small-exponent shells and streaming fallback (`MolecularGrid::should_stream`) controlled by `CALCHEM_MAX_MEMORY_BYTES`. Note that CLI `--grid` defaults to `Coarse` and `--conv-e` / `--conv-d` default to `1e-8` / `1e-8`, whereas Gaussian 16 defaults to `UltraFine`; cross-engine benchmarks explicitly match grids/thresholds or disclose differences (e.g. `pm6_l8bo_60` `Coarse` vs G16 `UltraFine`, and `Fine` vs G16 `UltraFine` on Caffeine/IC2F gradients).
- **Dispersion**:
  - `--disp d3bj` and `D3Model` implement **two-body Grimme D3(BJ)/D3(0)** with analytical nuclear gradients across `energy`, `grad`, `opt`, and `freq` (ATM three-body dispersion is excluded).
  - `wB97X-D` includes built-in Chai-Head-Gordon D2 dispersion (`wb97xd_chg_d2_energy_and_gradient`) consistently in both SCF total energy and gradients (`v0.3.0` fixed the `v0.2.3` `+9.1825 kcal/mol` omission in `energy_hartree`, bringing `IC2F wB97X-D/6-31G(d,p) RI-J(Auto)` to `-0.0366 kcal/mol` vs G16); combining `wB97X-D` with `--disp` is explicitly rejected.

---

## 4. RI-J Density Fitting (`--rij`)

- **Auxiliary basis choices**:
  - `--aux-basis auto-jfit` (default): per-angular-momentum AutoAux construction with 2-center Coulomb Cholesky/SPD factorization (`trunc == 0` full rank verified across E2E suites).
  - `--aux-basis def2-jfit-f`: embedded MolSSI BSE `def2-universal-jfit` up to Cartesian F-shells (`L <= 3`, with M6b fix splitting 5 collapsed `[1.0, 1.0]` shells).
- **Verified accuracy vs known boundaries**:
  - On polarized `6-31G(d)` / `6-31G(d,p)` hybrids (`B3LYP`, `wB97X-D`), `auto-jfit` matches G16 / Exact 4c energies within `< 0.1 kcal/mol` (`Caffeine`: `-0.0859 kcal/mol` vs G16; `IC2F B3LYP`: `-0.0608 kcal/mol`; `IC2F wB97X-D`: `-0.0366 kcal/mol`; `CH3CN`/`C2H3F`/`C4H4S`: `-0.026` to `-0.083 kcal/mol`).
  - **Known RI-J precision failures / outliers**:
    - `C4H4S PBE-D3BJ/6-31G(d) auto-jfit`: `-0.1307 kcal/mol` vs G16.
    - `Caffeine PBE-D3BJ/6-31G(d,p) auto-jfit`: `-0.1820 kcal/mol` vs G16.
    - `pm6_l8bo_60 PBE-D3BJ/6-31G(d) Coarse auto-jfit`: `-2.1029 kcal/mol` vs G16 `UltraFine Exact4c`.
    - `C4H4S B3LYP/6-31G(d) def2-jfit-f`: `-0.1249 kcal/mol` vs G16 (`T2_F1_fshell_def2`); `Caffeine def2-jfit-f`: `-0.3586 kcal/mol`.
    - Minimal `STO-3G` (`IC2F / STO-3G`, `Coarse`): `-4.0847 kcal/mol` (`auto-jfit`) vs Exact 4c. Use `--direct` without `--rij` when minimal-basis or pure-GGA sub-0.1 kcal/mol accuracy is required.

---

## 5. CUDA GPU Acceleration Scope (`--gpu`)

- `calchem-cuda` (`sm_120`, FP64) accelerates:
  1. RI-J 3-center integral evaluation and streaming up to Cartesian **F-shells (`L <= 3`)**, eliminating polarized-basis CPU fallback (`gpu_fallback_invocations == 0` verified across all 74 `含 GPU` E2E runs and 3 standalone GPU runs, including `T2_F1_ultrafine_grid`).
  2. RI-J tensor contractions (`contract_v`, `contract_coulomb`) and transient scratch reclamation (`TransientRijGuard` / `release_rij_scratch_if_idle`).
  3. Hybrid exact Hartree-Fock exchange (`Direct-K`) and range-separated exchange in SCF.
  4. Batched AO-Direct TD-DFT response contraction (`calchem_cuda_direct_contract_tddft_batch`).
  5. Analytical nuclear gradients (3c RI-J gradients, 4c ERI gradients `calchem_cuda_grad4c_contract`, and DFT `xc_nuclear_gradient_with_gpu`).
  6. DFT numerical XC grid quadrature (`V_xc`).
- **Telemetry & Parity**: GPU and Pure CPU Direct4c single-point and gradient runs agree within `|E_cpu - E_gpu| <= 1.89e-11 Eh` and `max|g_cpu - g_gpu| <= 5.11e-10 Eh/Bohr`. `scripts/verify_calchem_job.py --gpu` strictly enforces `gpu_kernel_invocations > 0` and `gpu_fallback_invocations == 0` (unless `--allow-gpu-fallback` is explicitly requested).

---

## 6. TD-DFT Excited States

- Native `calchem-tddft` supports closed-shell `RHF` / `RKS` (`B3LYP`, `PBE0`, `CAM-B3LYP`, `wB97X-D`) in both TDA and Full-LR (RPA) for `singlet`, `triplet`, or `both`, using batched two-stage trial-vector contractions on CPU (`build_tddft_response_batch`, `FxcKernel::contract_batch`, root locking) and GPU (`--gpu`), matching G16 `S1..S5` excitation energies within `0.000233 eV` on Caffeine (`174.53 s` Pure CPU 8T / `96.43 s` `含 GPU`, down from `5119.84 s` in `v0.2.3`) and `0.000085 eV` on `IC2F` (`287.82 s` Pure CPU 8T / `105.50 s` `含 GPU`).
- RI-J references, unrestricted (`UHF`/`UKS`) references, and excited-state nuclear gradients remain cleanly rejected before execution.
- **Known TD-DFT failure**: `T1_F3_04_c2h3f_lr_s` (`C2H3F B3LYP/6-31G(d)` Full-LR `nstates=3`) exhibits a state/root ordering mismatch against G16 (`max|Δω| = 0.74776 eV`, `max|Δf| = 0.32131`) in both CPU and GPU modes.

---

## 7. Continuum Solvation (SCRF) & Gradients

- **Models**: `C-PCM`, `COSMO`, `IEF-PCM` (York-Karplus spherical-Gaussian regularized boundary operator), `SMD` (`IEF-PCM` on `SmdIntrinsic` radii + parameterized Minnesota SMD-CDS), and `Onsager`.
- **Defaults vs Gaussian 16**: `C-PCM`, `IEF-PCM`, and `COSMO` default to `Bondi (alpha = 1.20)`, pure electrostatics (`include_cds = false`), and unrenormalized surface charges (`renormalize_charge = false`), whereas Gaussian 16 defaults to `UFF (alpha = 1.10)` for PCM. Known edge case: `ethanol` in `DMSO` under `SMD` yields `dG_CDS = -0.030` vs G16 `+0.140 kcal/mol`.
- **Gradients**: CLI SCRF (`--solv ...`) is restricted to `--task energy`. Library function `calchem_solv::compute_pcm_gradient_fd` evaluates fixed-AO-density central finite differences on the shared `ReactionFieldSolver` + `compute_cds_energy` surface; analytic PCM gradients remain unsupported (`UnsupportedMethod`).

---

## 8. Gradients, Geometry Optimization, & Harmonic Frequencies

- **Gradients**: Analytical gas-phase nuclear gradients (`calchem-grad`) support RHF/UHF/RKS/UKS, ECPs, RI-J, range-separated exchange, and two-body D3/wB97X-D dispersion on both CPU and CUDA GPU (`|E_cpu-E_gpu| <= 1.89e-11 Eh`, `max|g_cpu-g_gpu| <= 5.11e-10 Eh/Bohr`).
  - **Known gradient gap vs G16**: On 22–24 atom polarized systems (`Caffeine`, `IC2F` at `B3LYP-D3BJ/6-31G(d,p)` `Fine` grid), both Direct4c and RI-J analytical gradients differ from G16 default `UltraFine` by `1.39e-4–2.22e-4 Eh/Bohr` (`> 1e-4` gate), and wall time (`261.90 s` / `403.77 s` Pure CPU; `48.52 s` / `62.08 s` `含 GPU`) remains slower than G16 8T (`40.2 s` / `43.9 s`).
- **Optimization & Frequencies**: Native `calchem-opt` implements Cartesian BFGS / P-RFO (redundant internal coordinates / DLC are not yet implemented); `calchem-vib` computes numerical harmonic Hessians with rigid projection, `--freq-jobs N`, and displacement checkpoints (`n_imag = 0` verified on H2O and CH3CN minima).

---

## 9. Memory & Resource Admission Bounds

- Environment-overridable admission bounds (not OS-enforced cgroup limits):
  - `CALCHEM_MAX_ERI_BYTES = 4 GiB` (automatic switch from Stored to Direct SCF; explicit `--stored` above budget returns `ResourceLimit`).
  - `CALCHEM_MAX_MEMORY_BYTES = 8 GiB` (Direct SCF, grid streaming, and RI-J metric working-memory estimate).
  - `CALCHEM_MAX_DF_BYTES = 4 GiB` (cached/packed RI-J 3-center tensor limit).
  - GPU `vxc_cache` & RI-J streaming enforce the VRAM budget (`5 GiB` verified under 8 GiB RTX 5060 Ti).
- **Large-molecule status**: 60-atom `pm6_l8bo_60` (`6-31G(d)` `Coarse`, `734` AO) and 183-atom `D18` (`STO-3G` `Coarse`, `575` AO) single-point SCFs converge identically on CPU and GPU, though without G16 reference energies (`B3LYP`) or polarized/gradient/frequency runs above `6-31G(d)` `Coarse`. Full 72–726 atom polarized validation remains queued for Phase 5.
