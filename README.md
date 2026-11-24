# CalChem

Pure Rust (+ optional CUDA) quantum chemistry CLI engine for molecular SCF, gradients, geometry optimization, harmonic frequencies, TD-DFT, and continuum solvation.

> **Delivery status notice**: Online `v0.2.2` (`9f57bd7`) predates the fixes on this branch. The current working tree is an uncommitted local development build (`build_target/release/calchem_cli.exe`); Python bindings (`calchem-py` / PyPI) have been removed in favor of a pure Rust native CLI and optional Node.js CLI wrapper (`npm/`).

---

## Architecture

CalChem is organized as a 14-crate Cargo workspace:

| Crate | Responsibility |
|---|---|
| `calchem-model` | Molecule, atom, SCF options, diagnostics, error types |
| `calchem-basis` | Embedded Gaussian basis sets (`sto-3g`, `6-31g`, `6-31g(d)`, `6-31g(d,p)`, `lanl2dz`, `lanl2dzdp`) and `def2-universal-jfit` |
| `calchem-tensor` | Dense matrices, symmetric eigensolvers, tensor primitives |
| `calchem-int` | Obara-Saika 1e/2e/ECP ERIs, range-separated ERIs, Schwarz screening, Direct Fock, RI-J (`auto-jfit` / `def2-jfit-f`) |
| `calchem-dft` | Lebedev/Euler-Maclaurin molecular grids, spin-polarized XC (`SVWN5`, `BLYP`, `PBE`, `B3LYP`, `B3LYP5`, `PBE0`, `CAM-B3LYP`, `wB97X`, `wB97X-D`), two-body D3(BJ)/D3(0) dispersion |
| `calchem-scf` | RHF, UHF, RKS, UKS drivers, SAD guess, DIIS (with optional disk spill) |
| `calchem-grad` | Gas-phase analytical nuclear gradients (RHF/UHF/RKS/UKS + ECP/RI-J/range-separated/D3) |
| `calchem-vib` | Numerical harmonic Hessian, vibrational frequencies, rigid-rotor harmonic thermochemistry, displacement checkpoints |
| `calchem-opt` | Cartesian BFGS / P-RFO geometry optimization, wavefunction/guess reuse, atomic checkpoints & `--resume` |
| `calchem-prop` | Population/multipole property analysis |
| `calchem-solv` | SCRF solvation (`C-PCM`, `COSMO`, `IEF-PCM`, `SMD`, `Onsager`) & fixed-density FD solvation gradient |
| `calchem-tddft` | AO-Direct and stored-ERI TDA / Full-LR (RPA) Davidson excited-state solver |
| `calchem-cuda` | Optional FP64 CUDA kernels (`cuda_kernels.cu`) for RI-J 3c integrals (`L <= 2`), RI-J contractions, and DFT $V_{\text{xc}}$ grids |
| `calchem-cli` | Native `calchem_cli` binary (`.xyz` / `.gjf` input, structured atomic JSON output) |

---

## Build & Test

Workspace resource discipline caps active threads at `<= 8` (`RAYON_NUM_THREADS=4`, `OMP_NUM_THREADS=1`):

```powershell
$env:RAYON_NUM_THREADS = "4"
$env:OMP_NUM_THREADS = "1"
$env:CARGO_TARGET_DIR = "E:\YDW\Trae_Gaussian\build_target"

# Full Rust workspace test suite
cargo test --workspace --no-fail-fast --jobs 4

# Build release native CLI binary
cargo build --release -p calchem-cli --jobs 4
```

On the current working tree, `cargo test --workspace --no-fail-fast --jobs 4` passes **160 passed, 0 failed, 1 ignored** across **72** test suites (`profile_60atom_pm6_l8bo_cluster_stages` is explicitly `#[ignore]`).

---

## CLI Usage

```bash
# Exact-4c Direct RKS B3LYP-D3(BJ)/6-31G(d,p) single-point on Fine grid (8 CPU threads)
calchem_cli caffeine.xyz --task energy --threads 8 --method rks --xc b3lyp --basis "6-31g(d,p)" --grid fine --direct --disp d3bj --json --out result.json

# RI-J single-point (per-L AutoAux default: --aux-basis auto-jfit, or --aux-basis def2-jfit-f)
calchem_cli caffeine.xyz --task energy --threads 8 --method rks --xc b3lyp --basis "6-31g(d,p)" --grid fine --direct --rij --aux-basis auto-jfit --disp d3bj --json --out result.json

# Full-LR singlet TD-DFT (AO-Direct exact Coulomb)
calchem_cli caffeine.xyz --task tddft --threads 8 --method rks --xc b3lyp --basis "6-31g(d,p)" --grid fine --direct --disp d3bj --td-mode lr --spin singlet --nstates 5 --json --out tddft.json

# Implicit SCRF solvation (C-PCM / IEF-PCM / SMD; CLI supports single-point energy only)
calchem_cli acetic.xyz --task energy --threads 4 --method rks --xc b3lyp --basis "6-31g(d)" --grid fine --direct --solv smd --solvent water --json --out solv.json

# Geometry optimization with checkpoint & resume
calchem_cli mol.xyz --task opt --threads 4 --method rks --xc b3lyp --basis "6-31g(d)" --grid fine --checkpoint opt.ckpt.json --json --out opt.json
```

---

## Verified Benchmarks & Reference Comparisons

The runs below were recorded on `D:\YDW\Calc\qc_benchmark_runs\pilot\` prior to the final post-audit cleanup edits (`gradient.rs` / `matrix.rs` / `scrf.rs` / `density_fitting.rs` and `calchem-py` stripping); re-verification against a frozen single release binary SHA256 is scheduled before tagging `v0.2.3`.

| System & Task | Method / Basis / Grid | Mode & Threads | CalChem Result | Reference Comparison (G16 / Exact 4c) | Parity / Scope Notes |
|---|---|---|---|---|---|
| Caffeine (24 atoms, 260 AO, `6D/7F`) SP | `B3LYP-D3BJ/6-31G(d,p)`, `Fine` | Direct Exact 4c, 8 CPU | `-680.4310537200 Eh`, `265.56 s` (20 iters) | vs G16 (`31.5 s`): **`+0.0082 kcal/mol`** | CPU 8t vs G16 UltraFine 8t (`calchem_postfix_20261123_*`) |
| Caffeine (24 atoms, 260 AO, `6D/7F`) SP | `B3LYP-D3BJ/6-31G(d,p)`, `Fine` | RI-J `auto-jfit`, 8 CPU | `-680.4312032596 Eh`, `91.94 s` (`trunc=0`) | vs Exact 4c: **`-0.0938 kcal/mol`** (`-0.0856` vs G16) | `open01_caffeine_autojfit_20261124_v2` |
| Caffeine (24 atoms, 260 AO, `6D/7F`) Grad | `B3LYP-D3BJ/6-31G(d,p)`, `Fine` | RI-J `def2-jfit-f`, 8 CPU | `max_g=0.025204 Eh/Bohr`, `307.35 s` (`214.44 s` grad) | vs G16 SP: `-0.5813 kcal/mol` (`def2-jfit-f`) | `calchem_postfix_20261123_*` |
| `CH3CN`, `C2H3F`, `C4H4S` SP+Grad | `B3LYP/6-31G(d)`, `Fine` | RI-J `auto-jfit`, 4 CPU | `max\|ΔG\| = 9.4e-6 – 4.5e-5 Eh/Bohr` | vs Exact 4c: **`-0.028` to `-0.087 kcal/mol`** | `open01_rij_task3_20261124_v2` |
| Caffeine 5-root Full-LR TD-DFT (`6D/7F`) | `B3LYP/6-31G(d,p)`, `Fine` | AO-Direct LR, 8 CPU | `S1..S5`: `4.6959, 4.9140, 5.6900, 5.9785, 6.0324 eV` | vs G16 (`4.6961..6.0326 eV`): **`0.0001–0.0002 eV`** | **`4998.0 s` (~83 min)** vs G16 `192.2 s` (~25x slower); ran on pre-cleanup binary `0fb868d4...` |
| Solvation (`IEF-PCM`, `C-PCM`, `SMD`, 23 cases) | `(U)B3LYP/6-31G(d)`, `Fine` | SCRF Direct 4c, 4 CPU | 23/23 converged across neutrals, `acetate-`, `CH3·` | `\|Δ(ΔG_solv)\|` vs G16: **`0.002–0.318 kcal/mol`** | G16 used `UltraFine` and explicit `Radii=Bondi Alpha=1.20` for PCM (`open03_solvation_20261124_v2`) |

---

## Known Method Boundaries & Limitations

1. **RI-J on minimal `STO-3G` vs polarized bases**: On `IC-2F / STO-3G` (22 atoms, 94 AO, `B3LYP/Coarse`), RI-J exhibits a **`-4.08 kcal/mol`** error (`auto-jfit`) or **`-4.58 kcal/mol`** (`def2-jfit-f`) relative to Exact 4c. On polarized valence bases (`6-31G(d)` / `6-31G(d,p)`), verified organic systems fall within **`< 0.1 kcal/mol`** (`CH3CN`, `C2H3F`, `C4H4S`, Caffeine).
2. **Pure CPU Full-LR TD-DFT speed**: Although 8-thread Caffeine 5-root Full-LR TD-DFT matches Gaussian 16 excitation energies within `0.0002 eV`, AO-Direct trial-vector sweeps currently run only on CPU and take **`4998 s` (~83 min)** on 260 AOs, roughly **25× slower** than Gaussian 16 (`192.2 s`).
3. **Open-shell `UB3LYP` local correlation (`VWN_RPA` vs `VWN3`)**: `B3LYP` uses VWN Formula V (`VWN_RPA`) rather than Gaussian 16's VWN Formula III (`VWN3`). For closed-shell `RB3LYP` and relative solvation $\Delta G_{\text{solv}}$ differences, the offset cancels (`~0.014 kcal/mol` on `CH3·`), but open-shell `UB3LYP` absolute gas-phase energies differ from Gaussian 16 by **`~0.14 kcal/mol`** (`-0.1365 kcal/mol` on `CH3·/6-31G(d)`).
4. **CUDA GPU acceleration scope & F-shell fallback**: `calchem-cuda` currently accelerates RI-J 3-center integral generation **only up to D auxiliary/primary shells (`L <= 2`)**, RI-J tensor contractions (`contract_v`, `contract_coulomb`), and DFT $V_{\text{xc}}$ grid evaluation. When polarized basis sets generate F-type auxiliary shells (`L_aux = 3`), 3-center integral construction **falls back to CPU** (tracked in `gpu_fallback_invocations`). Hybrid exact exchange $K$, TD-DFT, and nuclear gradients run on CPU.
5. **PCM cavity radii defaults**: Continuum `C-PCM`, `IEF-PCM`, and `COSMO` default to **`Bondi (alpha = 1.20)`** radii with pure electrostatics (`include_cds = false`, `renormalize_charge = false`) rather than Gaussian 16's default **`UFF (alpha = 1.10)`** radii. `SMD` uses `SmdIntrinsic` radii and parameterized SMD-CDS.
6. **Optimizer & unverified large-molecule regimes**: Geometry optimization uses Cartesian BFGS / P-RFO rather than redundant internal coordinates (DLC); large systems (72–726 atoms with polarized bases) have not yet completed end-to-end verified runs.

See [docs/NATIVE_METHOD_BOUNDARIES.md](docs/NATIVE_METHOD_BOUNDARIES.md) and [docs/ROADMAP_AND_ISSUES.md](docs/ROADMAP_AND_ISSUES.md) for detailed technical boundaries and issue tracking.

## License

MIT / Apache-2.0 — see repository [LICENSE](LICENSE).
