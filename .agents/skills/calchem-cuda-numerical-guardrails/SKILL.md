---
name: calchem-cuda-numerical-guardrails
description: >-
  Domain-specific CUDA kernel safety, GPU/host memory budgeting, and quantum
  chemistry numerical guardrails learned from CalChem M2–M6 and Tier-5 white-box
  hardening. Use whenever modifying calchem-cuda, calchem-int, calchem-dft,
  calchem-scf, calchem-tddft, calchem-grad, calchem-opt, calchem-vib,
  calchem-solv, or calchem-basis.
---

# CalChem CUDA & Numerical Guardrails

Use this checklist whenever editing native CUDA kernels, Rust FFI bindings, integral/DFT engines, SCF/TD-DFT solvers, analytical gradients, optimizers, Hessians, or solvation modules.

---

## 1. CUDA Kernel & Rust/C++ FFI Guardrails (`calchem-cuda`, `calchem-int`)

1. **Sanitizer Verification on Every `.cu` Change**:
   - Run `compute-sanitizer --tool memcheck` and `compute-sanitizer --tool initcheck` targeting `0 errors` before accepting kernel changes.
   - Verify warp-level synchronization explicitly (`__syncwarp()` in warp-cooperative paths such as `eval_d_warp_impl`).
2. **Boundary & Tight-VRAM Adversarial Tests**:
   - Never validate GPU paths only with generous VRAM defaults; run tight-budget stress tests (e.g., `1 MiB` / `10 MiB` caps, `CALCHEM_MAX_DF_BYTES=1`) and forced small-stride batches to catch single-bra or scratch OOB errors (`dir_wl_stride`, `d_dir_f_scratch`).
3. **RAII Transient GPU Scratch Lifecycle**:
   - Never leave transient buffers (such as `scratch_p_ptr` / `scratch_ao_ptr`) dependent on caller discipline or special-cased IDs (e.g., skipping `tensor_id == 0` cleanup). Guard transient allocations with RAII (`TransientRijGuard`) and release idle scratch (`release_rij_scratch_if_idle()`).
4. **ID-Keyed Multi-Context Maps (No Single-Slot Globals)**:
   - Never store active driver/grid/RI-J/direct contexts in a single global slot (eviction corrupts concurrent SCF + TD-DFT/Grad workflows). Key contexts by ID in dedicated maps and add multi-context coexistence tests.
5. **FFI Bounds & Integer-Overflow Protection**:
   - Guard all element/byte size computations with `checked_mul` (Rust) and `safe_mul_sz` / `safe_add_sz` (C++), returning clean errors on overflow (test with `usize::MAX / 8 + 1` or malformed spans).
   - Validate streaming chunk bounds (`ao_start`, `ao_end`, `col_offset`, shell span) at the host/kernel FFI boundary before launch.
   - Count `gpu_fallback_invocations` exactly once per true fallback event; never double-count across wrapper layers.

---

## 2. Host RAM Budgeting & CPU/GPU Performance Boundaries

1. **Unified Host Memory Cap**:
   - All large host buffers (`vxc_cache` in `RksVxcEngine` and `exchange_rows` in `DirectFockEngine`) must share `CALCHEM_MAX_MEMORY_BYTES` cleanly and trigger `should_stream` fallback cleanly before hitting OOM.
2. **Accurate GPU Speedup & Telemetry Reporting**:
   - Never extrapolate small-molecule GPU launch overhead (where GPU can be slower than CPU) or pure-functional GPU RI-J single-point speedups to hybrid functionals, gradients, or larger/smaller regimes without measurements.
   - Verify `--gpu` runs execute real GPU kernels (`gpu_kernel_invocations > 0`) and track whether fallback stayed zero (`gpu_fallback_invocations == 0`).

---

## 3. Quantum Chemistry Numerical & Input Guardrails

1. **Scale-Aware Float64 SCF Plateau (`calchem-scf`, `calchem-vib`)**:
   - Never clamp SCF energy/density convergence thresholds below the IEEE-754 float64 rounding floor for total energy $|E|$ (e.g., `1e-13/1e-12` in Hessian setups causes 100-iteration stalls). Use scale-aware `e_plateau` logic alongside practical `1e-11/1e-10` targets.
2. **Decouple `Task::Sp` vs `Task::Opt` Convergence Thresholds (`calchem-opt`)**:
   - Loosening or tuning default single-point SCF tolerances must never relax `Task::Opt` convergence (`1e-11/1e-9` class) needed to keep line searches and geometry steps noise-free.
3. **Residual-Aware Davidson Buffer Roots (`calchem-tddft`)**:
   - Do not use fixed hard energy gaps or small fixed buffer-root caps (`k_roots + 2`), which drop strongly coupled low-lying states (`S1`). Check residuals across `k_roots..active_roots` with residual-aware `eff_buffer_tol`.
   - Enforce pre-SCF parameter validation (`n_states == 0`, `max_iter`, `conv_tol`, unsupported DF+TDDFT combinations) and preserve dense fast-path matrix symmetrization.
4. **Gradient Discrepancy Attribution Protocol (`calchem-grad`, `calchem-dft`)**:
   - Before modifying analytical gradient equations when matching Gaussian/ORCA/REST, run a **two-probe diagnosis**:
     1. Check analytical vs central finite-difference gradient parity (`~1e-8–1e-9 Eh/Bohr`).
     2. Compare `Fine` vs `UltraFine` grid error vs the reference engine.
   - If central FD matches to ~1e-9 and `Fine -> UltraFine` drops reference delta by ~10x, the delta comes from Becke vs SSF quadrature grid differences — do **not** break correct analytical gradient formulas.
5. **Physical Completeness, Basis Integrity & Pre-SCF Contracts**:
   - Verify dispersion terms (e.g., `D3` / `wB97X-D`) are included consistently across `energy_hartree`, `grad`, `opt`, and `freq`.
   - Audit embedded basis JSON data (e.g., reject/detect collapsed `[1.0, 1.0]` shells) and large auxiliary contraction paths.
   - Validate geometry, basis/element valence, multiplicity (`--mult 0` rejected), solvation parameters (`epsilon`, `scaling_factor`, `probe_radius`), thermo parameters (`T`, `P`, `symmetry_number`), and CLI flag conflicts **before** entering SCF; return non-zero exit codes whenever `!all_converged` or `NotConverged`.
   - Cross-platform I/O failure tests (such as disk DIIS spill errors) must trigger errors portably (e.g., passing an existing regular file path as target directory rather than assuming Windows-only unwritable paths).
