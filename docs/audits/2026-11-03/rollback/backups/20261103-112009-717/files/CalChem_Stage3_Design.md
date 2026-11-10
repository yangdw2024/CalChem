# CalChem Stage 3 — analytical RHF gradients and geometry optimization

This is an implementation handoff against the supplied Stage 2 archive. Paths below are relative to its root. Coordinates and gradients use Bohr and Eh/Bohr. The public APIs return `Result` so invalid or unconverged inputs cannot silently yield a force.

## 0. Prerequisite: a consistent RHF wavefunction

**Files:** `crates/calchem-scf/src/rhf.rs`; regression test in `crates/calchem-scf/tests/rhf_sto3g.rs`.

The current finalization computes `f` and `e_tot_final` from one density, diagonalizes `f`, then returns the *new* density and orbitals with the *old* Fock matrix and energy. Gradients require the energy-weighted density and total density to describe the same stationary solution. Tighten the final fixed-point step: build `F[P]`, diagonalize it, and iterate until both `P_new-P` and the commutator are below gradient-grade tolerances; then rebuild `F[P]`, compute the variational energy, and return a consistent canonical `(C, epsilon, P, F)`. If the fixed-point step does not converge, return `CalcError::NotConverged`. In particular, verify `max |P - 2 C_occ C_occ^T|`, `||FPS-SPF||`, and the energy recomputed from the returned `P` are small. Do not rely on the energy convergence test alone.

Stage 2 names `density_alpha` but RHF stores the **spin-summed** density `P=2 C_occ C_occ^T` there and leaves `density_beta=None`. Document this compatibility behavior in `crates/calchem-model/src/lib.rs`; do not multiply it by two again. `mo_coefficients` is row-major, with index `mu*n_mo+i`.

## 1. Integral derivatives and RHF gradient

### Files and signatures

| File | Change | Signatures |
| --- | --- | --- |
| `crates/calchem-grad/Cargo.toml` | Add direct tensor dependency. | `calchem-tensor = { path = "../calchem-tensor" }` |
| `crates/calchem-grad/src/lib.rs` | Public gradient, input checks, contraction. | `pub fn rhf_gradient(mol: &Molecule, basis: &BasisSet, wf: &ScfWavefunction) -> Result<Vec<[f64; 3]>, CalcError>` |
| `crates/calchem-grad/src/derivatives.rs` | Derivative shell blocks and nuclear term; keep derivative logic here. | `pub(crate) fn one_electron_derivative(mol: &Molecule, basis: &BasisSet, atom: usize, axis: usize) -> (Matrix2D, Matrix2D, Matrix2D)`; `pub(crate) fn eri_derivative_quartet(shells: [&Shell; 4], atom: usize, axis: usize) -> Vec<f64>`; `pub(crate) fn nuclear_repulsion_derivative(mol: &Molecule, atom: usize, axis: usize) -> f64` |
| `crates/calchem-int/src/primitive.rs`, `crates/calchem-int/src/eri.rs` | Reuse existing primitive evaluators for raised/lowered angular momenta. | No new public API required; `overlap_primitive`, `kinetic_primitive`, `nuclear_vrr`, and `eri_primitive` are already public. |
| `crates/calchem-grad/tests/rhf_gradient.rs` | Primitive and molecular tests. | Integration tests described below. |

`one_electron_derivative` returns `(dS,dT,dV)`, where `dV` includes both basis-center motion and explicit movement of the nuclear charge. A faster later version can contract blocks directly instead of allocating three dense matrices per coordinate. The needed primitive helpers are already re-exported by `calchem-int/src/lib.rs`; keep all derivative routines in `calchem-grad`.

For an unnormalized Cartesian primitive `g_l(r; A,alpha)`, with axis unit vector `e_q`, use the center identity

```text
∂g_l/∂A_q = 2 alpha g_(l+e_q) - l_q g_(l-e_q).
```

At the contracted AO level, apply this **inside the primitive loops** with the *original* contraction coefficient and normalization for `l`. The raised/lowered primitive is an algebraic integral argument; do not renormalize it as a new basis function. Skip the lower term when `l_q=0`. The analogous identity applies separately to each of the two centers of `S`, `T`, and each of the four centers of an ERI. Sum contributions from every center attached to the differentiated atom, including coincident centers.

For a nuclear attraction term `V^K_{mu nu}=-Z_K <mu|1/r_K|nu>`, its charge-center derivative follows translational invariance:

```text
∂_(R_K,q) V^K = - ∂_(A,q) V^K - ∂_(B,q) V^K,
```

where the right side denotes the two *formal AO-center* derivatives of the **same** `K` term. Add this term when `K` is the moved atom, alongside the basis-center derivatives when `mu` or `nu` belongs to it. Precompute Boys values to the raised angular-momentum order needed by each shifted integral. For two-electron integrals, request recurrence storage for the raised angular momentum; the current `eri_primitive` constructor computes the order from its arguments.

For `A != B`, the nuclear repulsion derivative is

```text
∂E_nn/∂R_A = - Σ_(B != A) Z_A Z_B (R_A-R_B) / |R_A-R_B|^3.
```

Reject coincident nuclei rather than returning infinities. The derivative of overlap contributes the energy-weighted density (Pulay term), while the nuclear attraction derivative includes the explicit Hellmann–Feynman term. With the spin-summed `P` used by this code, define

```text
W_mu,nu = Σ_(i occupied) 2 epsilon_i C_mu,i C_nu,i

G_Aq = dE_nn/dR_Aq
     + Σ_mu,nu P_mu,nu (dT_mu,nu + dV_mu,nu)
     + 1/2 Σ_mu,nu,lam,sig P_mu,nu P_lam,sig d(mu nu|lam sig)
     - 1/4 Σ_mu,nu,lam,sig P_mu,nu P_lam,sig d(mu lam|nu sig)
     - Σ_mu,nu W_mu,nu dS_mu,nu.
```

These are derivatives of **total energy**, so forces are `-G`. Contract the full AO index ranges initially to avoid symmetry factors and multiplicity errors. Once correct, a unique-quartet contraction may use `eri_permutations`, but deduplicate equal permutations before adding weights. Do not store a dense `3N_atom × N_AO^4` derivative tensor: compute one quartet and contract it immediately.

**Pseudocode:**

```text
check wf.converged, RHF singlet, AO dimensions, basis centers/atom indices,
      finite values, and matching mol/basis coordinates
P = row-major Matrix2D from wf.density_alpha    # already spin summed
W[mu,nu] = sum_i wf.mo_occupations[i] * eps[i] * C[mu,i] * C[nu,i]
check P ≈ C diag(occ) C^T and P/F/S stationary to gradient tolerance
for atom A, axis q:
    dS,dT,dV = derivative one-electron shell blocks
    G[A,q] = dE_nn + dot(P,dT+dV) - dot(W,dS)
    for each shell quartet: evaluate its four center derivatives for A
        contract Coulomb and exchange terms with P in the equation above
return G
```

**Validation:** At H₂ at 1.4 Bohr and at a displaced H₂ (for example 1.6 Bohr), compare every Cartesian coordinate to the independent Python reference central difference `(E(R+h)-E(R-h))/(2h)` with `h=0.002` Bohr and tighter SCF settings; assert max absolute error `<1e-4` Eh/Bohr. Repeat for the supplied asymmetric H₂O geometry plus a second distorted H₂O geometry, covering its bond and angle motions. Test all coordinates, including nominal zeros; verify `Σ_A G_A≈0` (translation), `Σ_A R_A×G_A≈0` (rotation), and `G_A=-force_A`. At a single atom, the gradient must vanish. Check the shell-level center derivative against finite differences of *the individual integral* before molecular tests. Gradient FD tolerances should account for SCF residual and step-size error; check convergence at `h` and `h/2`. Run the Python reference engine at every displaced point, avoiding a wrapper route that silently selects Rust.

## 2. BFGS geometry optimization

### Files and signatures

| File | Change | Signatures |
| --- | --- | --- |
| `crates/calchem-opt/Cargo.toml` | Add basis and SCF dependencies. | `calchem-basis = { path = "../calchem-basis" }`; `calchem-scf = { path = "../calchem-scf" }` |
| `crates/calchem-opt/src/lib.rs` | Optimizer and options. | `pub fn optimize_geometry(mol: &Molecule, basis: &BasisSet, method: &str) -> Result<Molecule, CalcError>`; `pub fn optimize_geometry_with_options(mol: &Molecule, basis_name: &str, opts: &OptOptions) -> Result<OptResult, CalcError>` |
| `crates/calchem-opt/tests/optimization.rs` | Geometry, descent, convergence tests. | Tests described below. |

`OptOptions` fields: `max_steps: usize` (default 100), `max_force: f64` (1e-4), `rms_force: f64` (5e-5), `max_displacement: f64` (1e-3), `max_step_bohr: f64` (e.g. 0.2), `scf_options: ScfOptions`, and line-search controls. `OptResult` contains `molecule`, `energy_hartree`, `gradient`, `steps`, `converged`, and the iteration history. The requested short API accepts a basis for the initial geometry, checks its name and shell ordering, then rebuilds `BasisSet::from_name(&basis.name, &trial_mol)` **at every trial point**; never reuse its old centers. `method` accepts RHF case-insensitively and returns `UnsupportedMethod` otherwise. The options API permits convergence settings without changing the requested short API.

Use inverse-Hessian BFGS on the flattened `3N` Bohr coordinates and the energy gradient. Initialize `H_0` to a positive diagonal scale; cap the trial step and use Armijo backtracking with actual converged SCF energies. Reset `H` to a positive diagonal when curvature `y·s` is too small or nonpositive. If the initial gradient is already below force tolerances, treat displacement as zero. Otherwise require **all three** tests at an accepted step: maximum absolute gradient component `<1e-4`, RMS over `3N` components `<5e-5`, and maximum atom Cartesian displacement (Euclidean norm) `<1e-3` Bohr. A stalled line search or `max_steps` exhaustion returns a descriptive error, not a falsely converged molecule.

```text
x = input coordinates; H = scaled identity; last_step = zero
repeat:
    mol(x); rebuild basis; wf = rhf(mol,basis,scf_options)
    E = wf.total_energy; g = rhf_gradient(mol,basis,wf)
    if force and displacement tests pass: return OptResult
    p = -H*g; if g·p >= 0: reset H; p = -g
    cap p to max_step_bohr; backtrack alpha until E(x+alpha*p)
        <= E(x) + c1*alpha*(g·p), with each trial's basis rebuilt
    calculate g_new at accepted point; s=x_new-x; y=g_new-g
    if s·y safely positive: H = (I-rho*s*y^T) H (I-rho*y*s^T) + rho*s*s^T
    else reset H
    last_step=s; x=x_new
```

**Validation:** H₂ from initial separations on either side of the minimum, and bent H₂O from two starting geometries. Require decreasing accepted energies, convergence flags, all three criteria, and agreement of final energy and geometry with an independently optimized **same-method, same-basis** Python result; rerun the final RHF single point. The suggested 0.741 Å H₂ bond and H₂O 0.958 Å/104.5° are useful experimental sanity landmarks, **not fixed golden values for RHF/STO-3G**. Derive numerical optimization goldens from the same RHF basis and code-independent reference before setting tolerances; do not assert those experimental numbers as a pass condition.

## 3. Performance, after correctness gates

| File | Signatures and approach | Validation |
| --- | --- | --- |
| `crates/calchem-int/Cargo.toml`, `src/lib.rs`, `src/eri.rs`, `src/one_electron.rs` | Add `rayon = { workspace = true }`. `pub fn compute_eri_with_options(basis: &BasisSet, opts: &IntegralOptions) -> Tensor4D`; retain `compute_eri` as an unscreened compatibility wrapper until energy and gradient equivalence passes. Parallelize *unique shell quartets* into thread-local `(quartet_id, block)` buffers; scatter permutations sequentially or into disjoint partitions to prevent races. One-electron shell-pair blocks can use the same indexed-buffer pattern. | Compare serial/parallel matrices and energies at fixed geometry; exercise 1 and several threads and repeated runs. |
| `crates/calchem-int/src/screening.rs` | `pub fn shell_pair_bounds(basis: &BasisSet) -> Vec<f64>`; for pair `(a,b)`, compute `B_ab = max_(mu in a,nu in b) sqrt(max(0,(mu nu|mu nu)))`. Skip `(ab|cd)` when `B_ab*B_cd < 1e-10`. Sort work by shell-pair bound if useful. | Check screened versus unscreened energies `<1e-6` Eh across H₂O/STO-3G and 6-31G and distorted geometries; record number of skipped quartets. The threshold is an integral bound, not by itself a rigorous *total-energy* bound. |
| `crates/calchem-tensor/src/lib.rs` | Preserve `pub fn matmul(a: &Matrix2D, b: &Matrix2D) -> Matrix2D`; implement row-major loop ordering `i,k,j` or tiled blocks with contiguous `j` writes, retaining the existing pure Rust Jacobi solver. | Compare against the old implementation on rectangular, odd-sized, and symmetric matrices, then benchmark end-to-end SCF. |
| `crates/calchem-scf/benches/rhf.rs` (or a repository-local release benchmark) | Time repeated *complete* H₂O/STO-3G single points including basis build, SCF, integrals and DIIS. Record CPU, Rust profile, threads, median and dispersion. | `<10 ms` is a measured target, not an assumed result. Keep the baseline and each isolated change; revert regressions. |

The ordinary Schwarz bound controls ERI **values**, not their nuclear derivatives. Initially keep gradient ERIs unscreened even when energy SCF uses screening, or derive and test a separate derivative bound. For a reliable stationary gradient, use an unscreened SCF/gradient pair until the screened energy discrepancy and derivative error are both established. Rayon overhead may exceed any gain on the five-AO H₂O/STO-3G case, so benchmark serial versus parallel and use an AO/shell-count threshold.

## 4. 6-31G validation and Python routing

**Files:** `crates/calchem-basis/tests/basis_normalization.rs`, `crates/calchem-scf/tests/rhf_631g.rs` (new), `crates/calchem-py/src/lib.rs`, `python_reference/native.py` (only after the Rust route is validated).

The supplied JSON has five O shells (`s,s,p,s,p`, 9 Cartesian AOs) and two `s` shells per H (2 each): H₂O has **13 AOs**. Assert shell offsets, centers, component ordering, `S_ii≈1`, normalized contractions, and positive definite overlap. Test H₂O/6-31G RHF energy against a pinned, independently calculated same-geometry, same-basis reference to `<1e-6` Eh. The approximate `-76 Eh` in the prompt is a coarse sanity check and cannot be used as a 1e-6 golden. Record geometry, units, Cartesian/spherical convention, basis source, SCF settings, and full precision of the reference. Also compare 6-31G gradients to independent Python finite differences before enabling the route.

Keep `run_single_point` and the existing Python dispatch unchanged while developing. Add a separate PyO3 gradient/optimization entry point only after standalone Rust tests pass; route only the supported RHF/basis combinations through Rust and preserve Python fallback for everything else. Verify that the RHF/STO-3G single-point path still reports the original Stage 2 energies and unsupported methods retain their Python behavior. Do not add DFT, UHF, or BLAS/LAPACK in Stage 3.

## Dependency order and acceptance

1. Fix final SCF consistency and confirm the original H₂, He, H₂O golden energies remain within `1e-6` Eh.
2. Implement integral center derivatives, then explicit charge-center and nuclear-repulsion terms; test each against centered differences at the integral level.
3. Contract the analytic RHF gradient and compare every component to independent Python finite differences to `<1e-4` Eh/Bohr; check translation and rotation invariance.
4. Implement BFGS with basis rebuilds and all three convergence tests; validate against same-level Python optimization.
5. Pin and validate H₂O/6-31G energy, AO count, normalization and gradients.
6. Add screening, parallelism and matrix multiplication changes one at a time, rerunning energy, gradient and optimization gates; measure the `<10 ms` release target.
7. Expose verified routes through PyO3 and check the dual-track Python backend.

The archive contains only `python_reference/hf.py`, `one_electron.py`, `two_electron.py`, and `native.py`; the imported Python `grad/finite_difference.py` and `opt/numeric_optimizer.py` are **not in this zip**. Obtain the actual Python reference package or a pinned independent reference dataset before claiming cross-language gradient/optimizer acceptance.
