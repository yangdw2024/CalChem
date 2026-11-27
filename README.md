# CalChem

Pure Rust (+ optional CUDA) quantum chemistry CLI engine for molecular SCF, gradients, geometry optimization, harmonic frequencies, TD-DFT, and continuum solvation.

> **Release baseline (`v0.3.0`)**: Native pure Rust (+ optional CUDA `sm_120`) `calchem_cli` engine with a strict fail-fast zero-Python Node.js CLI wrapper (`npm/@yangdw2024/calchem@0.3.0`).
> - Frozen `v0.3.0` Release binary (`build_target/release/calchem_cli.exe`, `22,292,480` B): SHA256 `C9F553BB0F96D183F34BD1E4A79563D0FAF750C33EFC3B6B9426DD908F1FD653`
> - CUDA `sm_120` cubin (`calchem_kernels_sm120.cubin`, `7,183,120` B): SHA256 `5A7DBBFC0908E9A40DCD2BFE5DAA34CD9C98F2EEFCA919B7A1E379A4BC30F533`
> - Crate manifest version note: Workspace `crates/*/Cargo.toml` remain at `0.2.2` on tag `v0.3.0` to preserve byte-for-byte source parity with the audited `C9F553BB...` binary.

---

## Architecture

CalChem is organized as a 14-crate Cargo workspace:

| Crate | Responsibility |
|---|---|
| `calchem-model` | Molecule, atom, SCF options, diagnostics, error types |
| `calchem-basis` | Embedded Gaussian basis sets (`sto-3g`, `6-31g`, `6-31g(d)`, `6-31g(d,p)`, `lanl2dz`, `lanl2dzdp`) and `def2-universal-jfit` |
| `calchem-tensor` | Dense matrices, symmetric eigensolvers, tensor primitives |
| `calchem-int` | Obara-Saika 1e/2e/ECP ERIs, range-separated ERIs, Schwarz shell-pair filtering & cache, Direct Fock, RI-J (`auto-jfit` / `def2-jfit-f`), batched TD-DFT contractions |
| `calchem-dft` | Lebedev/Euler-Maclaurin molecular grids, spin-polarized XC (`SVWN5`, `BLYP`, `PBE`, `B3LYP`, `B3LYP5`, `PBE0`, `CAM-B3LYP`, `wB97X`, `wB97X-D`), batched $F_{\text{xc}}$, two-body D3(BJ)/D3(0) dispersion |
| `calchem-scf` | RHF, UHF, RKS, UKS drivers, atomic-block SAD guess, drift-bounded incremental ΔP Fock, DIIS (with optional disk spill) |
| `calchem-grad` | Gas-phase analytical nuclear gradients (RHF/UHF/RKS/UKS + ECP/RI-J/range-separated/D3 + GPU offloading) |
| `calchem-vib` | Numerical harmonic Hessian, vibrational frequencies, rigid-rotor harmonic thermochemistry, displacement checkpoints |
| `calchem-opt` | Cartesian BFGS / P-RFO geometry optimization, wavefunction/guess reuse, atomic checkpoints & `--resume` |
| `calchem-prop` | Population/multipole property analysis |
| `calchem-solv` | SCRF solvation (`C-PCM`, `COSMO`, `IEF-PCM`, `SMD`, `Onsager`) & fixed-density FD solvation gradient |
| `calchem-tddft` | Batched AO-Direct and stored-ERI TDA / Full-LR (RPA) Davidson excited-state solver |
| `calchem-cuda` | Optional FP64 CUDA `sm_120` kernels (`cuda_kernels.cu`) for RI-J 3c (`L <= 3` incl. F-shells), RI-J contractions, hybrid `Direct-K`, TD-DFT batching, analytical nuclear gradients, and DFT $V_{\text{xc}}$ grids |
| `calchem-cli` | Native `calchem_cli` binary (`.xyz` / `.gjf` input, structured atomic JSON output, `--gpu` telemetry) |

---

## Build & Test

Workspace resource discipline caps active threads at `<= 8` (`RAYON_NUM_THREADS=4`, `OMP_NUM_THREADS=1`):

```powershell
$env:RAYON_NUM_THREADS = "4"
$env:OMP_NUM_THREADS = "1"
$env:CARGO_TARGET_DIR = "E:\YDW\Trae_Gaussian\build_target"

# Full Rust workspace test suite
cargo test --workspace --no-fail-fast --jobs 4

# Static lints across all workspace targets
cargo clippy --workspace --all-targets --jobs 4

# Python contract verification tests
python -m unittest discover -s E:\YDW\Trae_Gaussian\scripts -p "test_*.py" -v

# Build release native CLI binary
cargo build --release -p calchem-cli --jobs 4
```

On `v0.3.0`:
- `cargo test --workspace --no-fail-fast --jobs 4`: **230 passed, 0 failed, 1 ignored** across **72** test suites (`profile_60atom_pm6_l8bo_cluster_stages` is explicitly `#[ignore]`).
- `cargo clippy --workspace --all-targets`: **exit 0** (`0` errors; `0` warnings in `calchem-cuda` / `calchem-cli`).
- `scripts/test_verify_calchem_job.py`: **2 passed, 0 failed**.

---

## CLI Usage

```bash
# Exact-4c Direct RKS B3LYP-D3(BJ)/6-31G(d,p) single-point on Fine grid (8 CPU threads)
calchem_cli caffeine.xyz --task energy --threads 8 --method rks --xc b3lyp --basis "6-31g(d,p)" --grid fine --direct --disp d3bj --json --out result.json

# RI-J single-point with optional CUDA GPU acceleration (--gpu)
calchem_cli caffeine.xyz --task energy --threads 8 --method rks --xc b3lyp --basis "6-31g(d,p)" --grid fine --direct --rij --aux-basis auto-jfit --disp d3bj --gpu --json --out result.json

# Full-LR singlet TD-DFT (batched AO-Direct exact Coulomb)
calchem_cli caffeine.xyz --task tddft --threads 8 --method rks --xc b3lyp --basis "6-31g(d,p)" --grid fine --direct --disp d3bj --td-mode lr --spin singlet --nstates 5 --json --out tddft.json

# Implicit SCRF solvation (C-PCM / IEF-PCM / SMD; CLI supports single-point energy only)
calchem_cli acetic.xyz --task energy --threads 4 --method rks --xc b3lyp --basis "6-31g(d)" --grid fine --direct --solv smd --solvent water --json --out solv.json

# Geometry optimization with checkpoint & resume
calchem_cli mol.xyz --task opt --threads 4 --method rks --xc b3lyp --basis "6-31g(d)" --grid fine --checkpoint opt.ckpt.json --json --out opt.json
```

---

## Verified Benchmarks & Reference Comparisons (`v0.3.0`, SHA256 `C9F553BB...`)

All Phase 3 final runs were executed on the frozen `C9F553BB...` binary (`NewerThanExeCount = 0`) and archived under `D:\YDW\Calc\qc_benchmark_runs\phase3_m6\final_benchmarks\`. Pure CPU 8-thread (`AMD Ryzen 7 9700X`, `CALCHEM_DISABLE_GPU=1`) and **`含 GPU`** (`RTX 5060 Ti 8 GiB + 8T CPU`, `gpu_fallback_invocations = 0`) runs are separated strictly per `AGENTS.md` §2. Basis sets use Cartesian `6D/10F`, convergence thresholds `1e-10 / 1e-8`, and `Fine` grid unless noted (`Coarse` for 60/183-atom cases); G16 references run on the same host with 8 CPU threads unless noted.

### Pure CPU 8T Results (vs `v0.2.3` CPU 8T & G16 CPU 8T)

| System & Task | Basis (AO) / Method / Integral | Wall (Iters) & Speedup | Total Energy (Eh) & Accuracy vs G16 | Status |
|---|---|---|---|---|
| Caffeine (`24` at, `4967e90eb680`) SP | `6-31g(d,p)` (260) / `B3LYP-D3BJ` (20% EXX) / `Direct4c` | `31.15 s` (`18`); **`8.16x` v0.2.3** (`254.16 s`), **`1.01x` G16** (`31.5 s`) | `-680.43105426`; `dE = +0.0079 kcal/mol` | **PASS** |
| Caffeine (`24` at, `4967e90eb680`) SP | `6-31g(d,p)` (260) / `B3LYP-D3BJ` (20% EXX) / `RI-J(Auto)` | `26.77 s` (`18`); **`3.54x` v0.2.3** (`94.76 s`), **`1.18x` G16** (`31.5 s`) | `-680.43120380`; `dE = -0.0859 kcal/mol` | **PASS** |
| Caffeine (`24` at, `1acd5f86ae36`) LR-TDDFT 5s | `6-31g(d,p)` (260) / `B3LYP-D3BJ` (20% EXX) / `Direct4c` | `174.53 s` (`18`); **`29.33x` v0.2.3** (`5119.84 s`), **`1.10x` G16** (`192.2 s`) | `-680.43626091`; `dE = +0.0095 kcal/mol`, `max\|Δω\| = 0.000233 eV` | **PASS** |
| IC2F (`22` at, `22ae99e999de`) SP | `6-31g(d,p)` (290) / `B3LYP-D3BJ` (20% EXX) / `Direct4c` | `57.16 s` (`19`); `0.77x` G16 force (`43.9 s`, slower) | `-882.17945675`; `dE = -0.0086 kcal/mol` | **PASS** |
| IC2F (`22` at, `22ae99e999de`) SP | `6-31g(d,p)` (290) / `B3LYP-D3BJ` (20% EXX) / `RI-J(Auto)` | `45.51 s` (`19`); **`4.14x` v0.2.3** (`188.53 s`), `0.96x` G16 force | `-882.17954007`; `dE = -0.0608 kcal/mol` | **PASS** |
| IC2F (`22` at, `22ae99e999de`) LR-TDDFT 5s | `6-31g(d,p)` (290) / `B3LYP-D3BJ` (20% EXX) / `Direct4c` | `287.82 s` (`19`); `0.61x` G16 (`176.0 s`, slower) | `-882.17945675`; `dE = -0.0086`, `max\|Δω\| = 0.000085 eV` | **PASS** |
| IC2F (`22` at, `22ae99e999de`) SP | `6-31g(d,p)` (290) / `wB97X-D` (22.2% EXX) / `RI-J(Auto)` | `251.50 s` (`21`); `1.61x` v0.2.3 (`403.85 s`), `0.22x` G16 (`55.9 s`, slower) | `-881.83332524`; `dE = -0.0366 kcal/mol` | **PASS** |
| `pm6_l8bo_60` (`60` at) / `D18` (`183` at) SP | `6-31g(d)` (734) / `sto-3g` (575), `Coarse`, `RI-J(Auto)` | `2011.78 s` (`pm6` B3LYP); `26.31 s` / `48.87 s` (`D18` PBE/B3LYP) | `-3469.71809382` / `-6760.89305855` / `-6766.71666945` | **PASS (conv only; no G16 ref)** |
| Caffeine Direct4c Grad (`24` at) | `6-31g(d,p)` (260) / `B3LYP-D3BJ` / `Direct4c` | `261.90 s` (`0.15x` G16 `40.2 s`; **speedup VOID**) | `max\|Δg\| = 1.553e-4 > 1e-4 Eh/Bohr` (`Fine` vs G16 `UltraFine`) | **FAIL (Grad Gate)** |
| IC2F Direct4c Grad (`22` at) | `6-31g(d,p)` (290) / `B3LYP-D3BJ` / `Direct4c` | `403.77 s` (`0.11x` G16 `43.9 s`; **speedup VOID**) | `max\|Δg\| = 2.191e-4 > 1e-4 Eh/Bohr` (`Fine` vs G16 `UltraFine`) | **FAIL (Grad Gate)** |

### `含 GPU` Results (RTX 5060 Ti 8 GiB + 8T CPU, `fb=0` everywhere)

| System & Task | Basis (AO) / Method / Integral | Wall (Iters, GPU kernels) & Speedup | Total Energy (Eh) & Parity | Status |
|---|---|---|---|---|
| Caffeine SP (`含 GPU`) | `6-31g(d,p)` (260) / `B3LYP-D3BJ` / `Direct4c` | `14.02 s` (`18`, `k=57, fb=0`); **`2.18x` CPU**, **`18.13x` v0.2.3** | `-680.43105426`; `dE = +0.0079 kcal/mol` vs G16 | **PASS** |
| Caffeine SP (`含 GPU`) | `6-31g(d,p)` (260) / `B3LYP-D3BJ` / `RI-J(Auto)` | `13.45 s` (`18`, `k=96, fb=0`); **`1.98x` CPU**, **`7.05x` v0.2.3** | `-680.43120380`; `dE = -0.0859 kcal/mol` vs G16 | **PASS** |
| Caffeine LR-TDDFT 5s (`含 GPU`) | `6-31g(d,p)` (260) / `B3LYP-D3BJ` / `Direct4c` | `96.43 s` (`18`, `k=69, fb=0`); **`1.81x` CPU**, **`53.09x` v0.2.3** (G16 8T CPU `192.2 s`) | `-680.43626091`; `dE = +0.0095`, `max\|Δω\| = 0.000233 eV` | **PASS** |
| IC2F SP (`含 GPU`) | `6-31g(d,p)` (290) / `B3LYP-D3BJ` / `Direct4c` | `16.14 s` (`19`, `k=60, fb=0`); **`3.54x` CPU** | `-882.17945675`; `dE = -0.0086`, `\|E_cpu-E_gpu\|=1.39e-11 Eh` | **PASS** |
| IC2F SP (`含 GPU`) | `6-31g(d,p)` (290) / `B3LYP-D3BJ` / `RI-J(Auto)` | `15.61 s` (`19`, `k=101, fb=0`); **`2.91x` CPU**, **`12.08x` v0.2.3** | `-882.17954007`; `dE = -0.0608 kcal/mol` vs G16 | **PASS** |
| IC2F LR-TDDFT 5s (`含 GPU`) | `6-31g(d,p)` (290) / `B3LYP-D3BJ` / `Direct4c` | `105.50 s` (`19`, `k=72, fb=0`); **`2.72x` CPU** (G16 8T CPU `176.0 s`) | `-882.17945675`; `dE = -0.0086`, `max\|Δω\| = 0.000085 eV` | **PASS** |
| IC2F `wB97X-D` SP (`含 GPU`) | `6-31g(d,p)` (290) / `wB97X-D` / `RI-J(Auto)` | `61.50 s` (`21`, `k=133, fb=0`); **`4.06x` CPU** (still slower than G16 8T CPU `55.9 s`) | `-881.83332524`; `dE = -0.0366 kcal/mol` vs G16 | **PASS** |
| `pm6_l8bo_60` (`60` at) / `D18` (`183` at) (`含 GPU`) | `6-31g(d)` (734) / `sto-3g` (575), `Coarse`, `RI-J(Auto)` | `pm6` B3LYP `292.75 s` (`k=575, fb=0`, **`6.87x` CPU**); `D18`: `15.67 s` (`1.65x`), `22.22 s` (`2.20x`) | `\|E_cpu-E_gpu\| <= 6.7e-11 Eh` (`pm6` B3LYP) | **PASS (conv only; no G16 ref)** |
| Caffeine Direct4c Grad (`含 GPU`) | `6-31g(d,p)` (260) / `B3LYP-D3BJ` / `Direct4c` | `48.52 s` (`5.40x` CPU, slower than G16 8T CPU `40.2 s`; **speedup VOID**) | `max\|g_cpu-g_gpu\|=4.82e-10`; **`max\|Δg\| = 1.553e-4 > 1e-4`** | **FAIL (Grad Gate)** |
| IC2F Direct4c Grad (`含 GPU`) | `6-31g(d,p)` (290) / `B3LYP-D3BJ` / `Direct4c` | `62.08 s` (`6.50x` CPU, slower than G16 8T CPU `43.9 s`; **speedup VOID**) | `max\|g_cpu-g_gpu\|=5.11e-10`; **`max\|Δg\| = 2.191e-4 > 1e-4`** | **FAIL (Grad Gate)** |

### End-to-End Suite Totals (`C9F553BB...`, `exit_code == 0`, `timed_out == 0`, `skipped == 0`)
- **Pure CPU 8T (`92` cases)**: **`83 passed`**, `2 known_precision_outlier`, **`7 failed`** (`0` timed out).
- **`含 GPU + 8T` (`74` cases)**: **`65 passed`**, `2 known_precision_outlier`, **`7 failed`** (`0` timed out, **`gpu_fallback_invocations = 0` on all 74 cases**).
- **Solvation (`v0.3.0` + `v0.2.3` 23-case sweep)**: `T1_F5` / `T2_F5` E2E cases pass on `C9F553BB...` (`|ΔΔG_solv| <= 0.064 kcal/mol` on checked cases, matching the `v0.2.3` 23-case sweep).

---

## Known Method Boundaries & Failing / Slow Items

1. **7 E2E Precision Failures (identical in CPU and GPU suites vs G16)**:
   - `T1_F2_06_c4h4s_pbe_d3`: `PBE-D3BJ/6-31G(d) RI-J(Auto)` energy diff `-0.1307 kcal/mol` (`> 0.1`).
   - `T4_W1c_caffeine_pbe_aj`: `PBE-D3BJ/6-31G(d,p) RI-J(Auto)` energy diff `-0.1820 kcal/mol` (`> 0.1`).
   - `T4_W5a_pm6_60_pbe_aj_sp`: 60-atom `PBE-D3BJ/6-31G(d) RI-J(Auto)` (`Coarse` vs G16 `UltraFine Exact4c`) energy diff `-2.1029 kcal/mol` (`> 0.1`), and Pure CPU `790.17 s` is slower than G16 `468.2 s`.
   - `T1_F3_04_c2h3f_lr_s`: `C2H3F` Full-LR TD-DFT (`nstates=3`) diverges in low-lying state ordering (`max|Δω| = 0.74776 eV`, `max|Δf| = 0.32131`) vs G16.
   - `T4_W2a_caffeine_b3lyp_aj_grad`, `T4_W2b_caffeine_def2jfit_grad`, `T4_W4a_ic2f_b3lyp_aj_grad`: `B3LYP-D3BJ/6-31G(d,p)` `Fine`-grid gradients differ by `1.389e-4–2.222e-4 Eh/Bohr` vs G16 default `UltraFine` (`def2-jfit-f` also has `-0.3586 kcal/mol` energy error on Caffeine).
2. **2 Known Precision Outliers**: `T2_F1_fshell_def2` (`C4H4S/6-31G(d)` `def2-jfit-f` `-0.1249 kcal/mol`) and `T2_F1_ic2f_sto3g_rij` (`IC2F/STO-3G` `auto-jfit` `-4.0847 kcal/mol` vs Exact 4c).
3. **Analytic Gradient Grid Offset & Speed**: Direct4c analytical gradients on `Fine` grid also differ from G16 `UltraFine` by `1.553e-4` (Caffeine) and `2.191e-4 Eh/Bohr` (IC2F), and remain slower than G16 8T in both Pure CPU (`261.90 s` / `403.77 s`) and `含 GPU` (`48.52 s` / `62.08 s` vs G16 `40.2 s` / `43.9 s`), even though GPU internal parity matches CPU (`<= 5.11e-10 Eh/Bohr`).
4. **Slow Polarized Pure CPU Cases**: `IC2F wB97X-D` (`251.50 s` CPU / `61.50 s` `含 GPU` vs G16 `55.9 s`), `IC2F LR-TDDFT` (`287.82 s` CPU vs G16 `176.0 s`), `IC2F Direct4c SP` (`57.16 s` CPU vs G16 `43.9 s`), and `C4H4S Direct4c Grad` (`9.49 s` CPU vs G16 `5.7 s`) remain slower than G16 8T.
5. **Open-shell `UB3LYP` local correlation (`VWN_RPA` vs `VWN3`)**: `B3LYP` uses VWN Formula V (`VWN_RPA`) rather than G16's VWN3, introducing `~0.14 kcal/mol` gas-phase offset on `CH3·/6-31G(d)` (cancels in relative $\Delta G_{\text{solv}}$).
6. **PCM cavity radii defaults**: `C-PCM`, `IEF-PCM`, and `COSMO` default to `Bondi (alpha = 1.20)` pure electrostatics without charge renormalization rather than G16 `UFF (alpha = 1.10)`. `ethanol` in `DMSO` under `SMD` still has a small CDS sign discrepancy (`-0.030` vs `+0.140 kcal/mol`).
7. **Optimizer & Unverified Large-System Regimes**: Geometry optimization uses Cartesian BFGS / P-RFO rather than redundant internal coordinates (DLC). `pm6_l8bo_60` `B3LYP` and `D18` have no G16 reference energies and have not been benchmarked with gradients, frequencies, or polarized bases above `6-31G(d)` `Coarse`.

See [docs/NATIVE_METHOD_BOUNDARIES.md](docs/NATIVE_METHOD_BOUNDARIES.md) and [docs/ROADMAP_AND_ISSUES.md](docs/ROADMAP_AND_ISSUES.md) for detailed technical boundaries and issue tracking.

## License

MIT / Apache-2.0 — see repository [LICENSE](LICENSE).
