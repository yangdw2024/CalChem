# CalChem v1.0 — Rust Engine Implementation Specification
## Stage 2 through Stage 5+ | Based on Stage 1.5 handoff ZIP

**Status:** Design and source inspection; implementation and Rust execution are NOT claimed.  
**Source inspected:** `CalChem_Rust_Handoff_v2_Stage1.5.zip`, 29 members, including 9 Rust crates, 5 Python reference snapshots, two embedded basis JSON files, `BASELINE.md`, and the handoff prompt.  
**Constraints:** no libcint; no Python runtime dependency in Rust core; Python fallback remains available; energy tolerance 1e-8 Eh; gradient tolerance 1e-6 Eh/Bohr; no unvalidated integral code merged.

---

# 1. Executive decisions

1. **Keep all nine existing crates.** Extend them with modules before adding crates. Introduce `calchem-grid`, `calchem-xc`, `calchem-dft`, and `calchem-posthf` only when those subsystems are implemented.
2. **Use `faer` as the default dense linear algebra backend.** The workspace already declares `faer = "0.20"` but `calchem-tensor` still uses nested `Vec<Vec<f64>>` and a home-grown Jacobi eigensolver. Do not make OpenBLAS a mandatory Windows/MSVC dependency; an optional BLAS feature can be benchmarked later. Pin and test the actual `faer` API version.
3. **Implement native Obara–Saika (OS) ERI VRR/HRR.** A C-integral wrapper does not solve the portability goal and libcint is excluded. Keep small independent analytic s-type and parity/finite-difference tests to detect correlated implementation mistakes.
4. **Make Stage 2 a numerical-integrity stage, not just a p-type stage.** Fix kinetic derivatives, Boys accuracy, Cartesian normalization, basis integrity, and SCF status/DIIS before enabling additional molecules.
5. **Make the dual-track Python router capability-based and strict.** Fall back for `Unsupported`; do not silently fall back on wrong energies, `NotConverged`, numerical failures, or programmer errors. For compatibility, provide a separately configurable `prefer_native=False` / `fallback_on_failure` policy if desired, with observable backend provenance.
6. **Keep `calchem-int` independent of SCF, gradients, and Python.** All AO integral and derivative primitives live in `calchem-int`; `calchem-scf` consumes an integral-provider interface; `calchem-grad` uses converged SCF state plus derivative integrals; `calchem-opt` consumes a generic energy-and-gradient callback; `calchem-prop` consumes converged results plus property-integral builders.
7. **Scope performance claims to explicit benchmark families.** “Faster than Gaussian/ORCA/PySCF” is a product aspiration, not a demonstrated outcome. Start with a reproducible 10x H2O/STO-3G vs the pinned Python reference on the specified CPU and record end-to-end and kernel speed separately.

## 1.1 Actual code issues to fix

| Path / location in ZIP | Finding | Release implication |
|---|---|---|
| `crates/calchem-int/src/lib.rs`, `nuclear_vrr` | p-type attraction = `V00 × center_offsets`, not OS | All molecules with p shells blocked |
| `crates/calchem-int/src/lib.rs`, `compute_eri` | non-ssss quartet values left zero | RHF water/methane energies invalid |
| `crates/calchem-int/src/lib.rs`, `kinetic_general` | incorrect factors (`la*la` instead of `la*lb`), invalid negative component cases and over-restrictive cross-term tests | Must replace, not patch only one coefficient |
| `crates/calchem-int/src/lib.rs`, `boys` | alternating Taylor sum at T≈25–30 is unstable; simple test gives F0(30) absolute error around 4.7e-6 vs SciPy special-function reference | Can invalidate higher-angular attraction/ERI |
| `crates/calchem-basis/src/lib.rs`, `ShellType::from_str` | unknown type maps silently to S | Must return `Result` |
| `crates/calchem-basis/src/lib.rs`, `norm_d` | one normalization for all Cartesian d components | dxx and dxy need different factors |
| `crates/calchem-basis/src/lib.rs`, `BasisSet::build` | raw primitive contraction coefficients, not the reference normalization | Normalize and compare diagonal overlap |
| `crates/calchem-basis/data/6-31g.json` | several carbon 6-31G entries differ from standard published values (e.g. C valence exponent 2.9412494 vs canonical 7.86827235...) | Replace and independently verify before activating |
| `crates/calchem-scf/src/lib.rs`, `solve_linear` | singular DIIS returns all-zero coefficients | Use robust solve and backoff |
| `crates/calchem-scf/src/lib.rs`, `rhf` | assumes closed shell without full charge/spin checks, returns last iteration energy evaluated at previous density, allows `converged=false` as an ordinary success | Explicit input and convergence semantics |
| `crates/calchem-py/src/lib.rs` | every basis string loads STO-3G JSON | Exact named dispatch or `Unsupported` |
| `crates/calchem-tensor/src/lib.rs` | Jacobi capped at 100 rotations without accuracy-status reporting | Replace via `faer`, validate eigensystems |

`BASELINE.md` explicitly identifies the water fixture as O(0,0,0), H(0,1.432,0), H(1.112,-0.716,0), **all in Bohr**. Do not compare it to equilibrium-water numbers.

## 1.2 Module graph / public contracts

```text
calchem-model ─────► calchem-basis ─────► calchem-int
       │                     │                │
       └─────────────────────┴─────► calchem-scf ◄── calchem-tensor
                                        │
                        ┌───────────────┼──────────────────────┐
                        ▼               ▼                      ▼
                    calchem-grad    calchem-prop          future calchem-dft
                        │
                        ▼
                    calchem-opt (callback driven)
                        └─────────────────────────────────────► calchem-py
```

Dependency arrows in this sketch mean “consumed by”; the actual Rust crates should retain a directed acyclic dependency graph. `calchem-int` must never import `calchem-scf`. `calchem-opt` should not import a concrete SCF implementation.

**Global conventions:** all coordinates are in Bohr and energies in Hartree; explicit units in field names; `Result<T, CalcError>` for fallible Rust API; AO ordering and Cartesian ordering frozen and documented; `f64` production numerics; thread-safe shell-local scratch; deterministic test mode; no silent synthetic zero integral; `NonConverged` distinct from `Unsupported`; all exposed capabilities must have integration tests.

---

# 2. Stage 2 — Full restricted Hartree–Fock (P0)

## 2.1 Task/file matrix

| Task | Exact path (new unless noted) | Main signatures / responsibilities | Prerequisite |
|---|---|---|---|
| S2.01 | `crates/calchem-model/src/lib.rs` (modify) | `fn validate_rhf(&self)->Result<(usize,usize),CalcError>`; charge/multiplicity/electron/geometry checks | None |
| S2.02 | `crates/calchem-basis/src/lib.rs` (modify) | `pub fn from_name(name:&str, mol:&Molecule)->Result<BasisSet,CalcError>`; strict shell parsing, component norms, deterministic AO map | S2.01 |
| S2.03 | `crates/calchem-basis/src/normalization.rs` | `pub fn primitive_norm(alpha:f64, lmn:[u8;3])->f64`; `pub fn normalized_contraction(exps:&[f64], coeffs:&[f64],lmn:[u8;3])->Result<Vec<f64>,CalcError>` | S2.02 |
| S2.04 | `crates/calchem-basis/data/6-31g.json` (replace, verify) | Adopt correct BSE data; preserve source version and SHA-256 in fixtures | S2.03 |
| S2.05 | `crates/calchem-int/src/boys.rs` | `pub fn boys_batch(max_n:usize,t:f64,out:&mut[f64])->Result<(),CalcError>` | None |
| S2.06 | `crates/calchem-int/src/primitive.rs` | `pub(crate) fn gaussian_pair(a:&PrimitiveGto,b:&PrimitiveGto)->GaussianPair` | S2.03 |
| S2.07 | `crates/calchem-int/src/one_electron.rs` | `pub fn shell_overlap(a:&Shell,b:&Shell)->Result<ShellMatrix,CalcError>`; same for kinetic; `pub fn shell_nuclear(a:&Shell,b:&Shell,nuclei:&[Atom])->Result<ShellMatrix,CalcError>` | S2.05–S2.06 |
| S2.08 | `crates/calchem-int/src/eri.rs` | `pub fn eri_shell_quartet(a:&Shell,b:&Shell,c:&Shell,d:&Shell,out:&mut[f64])->Result<(),CalcError>` | S2.05–S2.07 |
| S2.09 | `crates/calchem-int/src/symmetry.rs` | `pub fn unique_shell_quartets(n_shells:usize)->impl Iterator<Item=[usize;4]>`; `pub fn eri_permutations(i:usize,j:usize,k:usize,l:usize)->[(usize,usize,usize,usize);8]` | S2.08 |
| S2.10 | `crates/calchem-int/src/lib.rs` (modify) | `pub fn IntegralEngine::build(basis:&BasisSet, mol:&Molecule)->Result<Self,CalcError>`; thin orchestration, correct indexed scatter | S2.07–S2.09 |
| S2.11 | `crates/calchem-tensor/src/lib.rs` (modify) | expose checked dense `Matrix`/faer wrappers; `pub fn eigh_symmetric(a:&Matrix)->Result<Eigensystem,CalcError>`; `solve_diis` | None |
| S2.12 | `crates/calchem-scf/src/diis.rs` | `pub struct Diis { ... }`; `pub fn extrapolate(&mut self, fock:&Matrix, residual:&Matrix)->Result<Option<Matrix>,CalcError>` | S2.11 |
| S2.13 | `crates/calchem-scf/src/rhf.rs` | `pub fn rhf(mol:&Molecule,basis:&BasisSet,opts:&ScfOptions)->Result<ScfResult,CalcError>` | S2.10–S2.12 |
| S2.14 | `crates/calchem-scf/src/lib.rs` (modify) | reexport API and converged-only result contract | S2.13 |
| S2.15 | `crates/calchem-py/src/lib.rs` (modify) | keep `run_single_point` Python-facing signature; dispatch exact basis name; map typed failures | S2.13 |
| S2.16 | `calchem/backends/native.py` (modify in full project; ZIP has snapshot at `python_reference/native.py`) | broaden native RHF support only for validated elements/bases; preserve Python fallback | S2.15 |
| S2.17 | `tests/rust_integration/stage2_rhf.rs`, `tests/python/test_native_parity.py` | frozen fixture/SCF parity and unsupported-path tests | all |

Place crate-specific unit tests in `crates/calchem-basis/tests/{basis_normalization,basis_data}.rs`, `crates/calchem-int/tests/{boys,one_electron,eri_symmetry,eri_reference}.rs`, and `crates/calchem-scf/tests/{rhf_golden,diis_recovery}.rs`. Since `tests/rust_integration/` at a workspace root does not automatically run as a Rust test crate, either add a small `calchem-tests` workspace member or place integration tests in the consuming crate's `tests/` directory. **Preferred:** place each integration test in the appropriate crate and use `tests/python/` for bridge testing.

## 2.2 Primitive and contraction normalization

Use Cartesian `lmn`, not only shell angular momentum L:

\[
N_{\ell_x\ell_y\ell_z}(\alpha)=
(2\alpha/\pi)^{3/4}\sqrt{
(4\alpha)^{\ell_x+\ell_y+\ell_z}/
[(2\ell_x-1)!!(2\ell_y-1)!!(2\ell_z-1)!!]}.
\]

Consequently \(N_{d_{xx}}/N_s=4\alpha/\sqrt{3}\), whereas \(N_{d_{xy}}/N_s=4\alpha\). Normalize contraction coefficients by the actual unnormalized contracted self-overlap for the selected component; the shell's final overlap diagonal should be ~1. Compare **all** basis components to the Python reference, which has `_normalize_contraction()` and per-primitive `GTOPrimitive._compute_norm()`.

`Shell` must retain `atom_index` for nuclear-gradient center attribution and a stable list of `CartesianExponent` components.

**Pseudocode:**
```text
validate positive, finite exponents and equal nonempty coeff/exponent lengths
for every Cartesian lmn component:
    for primitive i: component_norm[i] = primitive_norm(alpha[i], lmn)
    S_contract = sum_ij c[i]c[j]N[i]N[j] <primitive_i|primitive_j>
    reject S_contract <= epsilon or nonfinite
    norm_coeff[i] = c[i] / sqrt(S_contract)
assert resulting contracted S(ii) ~ 1
```
Cost per shell \(O(n_{\rm prim}^2 n_{\rm cart})\), memory \(O(n_{\rm prim}n_{\rm cart})\).

## 2.3 Boys function: stable scheme

Use \(F_n(T)=\int_0^1 u^{2n}e^{-Tu^2}\,du\), \(F_n(0)=1/(2n+1)\). For e.g. \(T\le40\) use a **positive** series (not the current alternating series):
\[
F_n(T)=e^{-T}\sum_{k\ge0} a_k,\quad
a_0={1\over 2n+1},\quad
a_{k+1}=a_k {2T\over 2n+2k+3}.
\]
For \(T>40\), compute \(F_0(T)=\frac12\sqrt{\pi/T}\operatorname{erf}\sqrt T\) with a high-quality `erf` implementation and use
\(F_{n+1}=[(2n+1)F_n-e^{-T}]/(2T)\).
Benchmark and test the switch, tiny T, n up to at least 12, and T as large as 1000. Use vectorized `boys_batch` so primitive quartets calculate all m in one call, reuse scratch to avoid allocation.

**Pseudocode:**
```text
reject T < 0 or nonfinite; check output size
if T <= 40:
    for n in 0..max_n:
        a = 1/(2n+1); sum = a
        for k=0..until relative_term < 1e-16:
            a *= 2*T/(2*n+2*k+3)
            sum += a
        out[n] = exp(-T)*sum
else:
    out[0] = 0.5*sqrt(pi/T)*erf(sqrt(T))
    for n in 0..max_n-1:
        out[n+1] = ((2n+1)*out[n]-exp(-T))/(2T)
```
Cost \(O(m_{\max}\times k(T))\) at moderate T; \(O(m_{\max})\) for large T. `erf` approximations limited to ~1e-7 are not enough for general 1e-8 Eh energy guarantees without error-bound analysis.

## 2.4 Overlap, kinetic, and nuclear attraction

**One-electron strategy:** keep the existing correct Gaussian product theorem and analytic `overlap_ss`, implement axis-aware s/p/d overlap via one-dimensional OS or validated binomial moments, then express kinetic energy by valid angular shifts:

\[
T_{ab} = -\frac12\sum_{i=x,y,z}
\left[
b_i(b_i-1)S_{a,b-2e_i}
-2\beta(2b_i+1)S_{a,b}
+4\beta^2 S_{a,b+2e_i}
\right].
\]

These are *unnormalized primitive integrals*; multiply by the original normalized primitive coefficients once. A negative angular index contributes zero. This simple Laplacian-on-ket formula avoids the current malformed mixed-derivative implementation and is equivalent to the gradient formula after correct integration by parts. Validate both to detect mistakes.

For nuclear attraction define
\[
V_{00}^{(m)}(C)= -Z_C(2\pi/p)K_{AB}F_m(p|P-C|^2)
\]
with \(p=\alpha+\beta\), \(PA=P-A\), \(PB=P-B\), \(PC=P-C\). To raise angular momentum on A along axis i:
\[
V_{a+e_i,b}^{(m)} =
PA_i V_{a,b}^{(m)}-PC_i V_{a,b}^{(m+1)}
+\frac{a_i}{2p}(V_{a-e_i,b}^{(m)}-V_{a-e_i,b}^{(m+1)})
+\frac{b_i}{2p}(V_{a,b-e_i}^{(m)}-V_{a,b-e_i}^{(m+1)}).
\]
Similarly, raising B uses \(PB_i\), \(PC_i\) and the same lower-angular terms. The `-PC_i V^{m+1}` sign is crucial and is directly testable with p-s attraction.

**Pseudocode `shell_nuclear`:**
```text
allocate zero shell block [ncart(a), ncart(b)]
for primitive pair (a_prim,b_prim):
    gaussian_pair -> p, P, K
    for atom nucleus C:
        fill Boys F_0..F_{La+Lb} for T = p*|P-C|^2
        create per-primitive memo/table V(a_lmn,b_lmn,m)
        set V(0,0,m) = -Z*2*pi/p*K*F_m
        recurse strictly from lower total angular momentum:
            if A carries momentum, peel one A component and apply A formula
            else peel one B component and apply B formula
        accumulate c_a*c_b*N_a*N_b*V(target, 0) in all Cartesian entries
return block
```

For fixed maximal shell angular momenta s/p/d, one shell pair costs \(O(n_a n_b n_{\rm cart,a}n_{\rm cart,b} N_{\rm nuclei} C_{\rm VRR})\) with memoized recurrence; do not regenerate Cartesian index vectors inside every primitive/nuclear loop.

## 2.5 Four-center ERI: OS VRR followed by HRR

A generic implementation is preferable to manually coding only ssss, spss, and pppp. The targets are supported automatically once Cartesian angular momentum raising and contraction are correct.

Definitions:
\[
p=\alpha+\beta,\ q=\gamma+\delta,\ \rho=pq/(p+q);
\quad P=(\alpha A+\beta B)/p,\quad Q=(\gamma C+\delta D)/q,
\quad W=(pP+qQ)/(p+q).
\]
Base:
\[
[00|00]^{(m)}=
\frac{2\pi^{5/2}}{pq\sqrt{p+q}}\,
K_{AB}K_{CD}F_m(\rho |P-Q|^2).
\]

Raise A, with B and D initially zero:
\[
[a+e_i,0|c,0]^{(m)} =
(P_i-A_i)[a0|c0]^{(m)}
+(W_i-P_i)[a0|c0]^{(m+1)}
+\frac{a_i}{2p}\left([a-e_i,0|c0]^{(m)}
-\frac{\rho}{p}[a-e_i,0|c0]^{(m+1)}\right)
+\frac{c_i}{2(p+q)}[a0|c-e_i,0]^{(m+1)}.
\]
Raise C analogously by swapping (A,p,P,a) with (C,q,Q,c) and \(W-P\) with \(W-Q\). Then shift angular momentum from A to B, and from C to D, by the horizontal recurrence:
\[
(a,b+e_i|c,d)=(a+e_i,b|c,d)+(A_i-B_i)(a,b|c,d),
\]
with analogous C/D shift. Every angular-momentum state must be computed from initialized dependencies, not an iteration order that can silently encounter a missing term. Use memoized recursion during Stage 2; introduce fixed-size scratch recurrence tables for performance later.

**Pseudocode `eri_shell_quartet`:**
```text
for primitive a,b,c,d:
    compute P,Q,W,p,q,rho,prefactor; Boys F[0..La+Lb+Lc+Ld]
    initialize [00|00]^m for required m
    recursively form [a+b,0|c+d,0]^m with OS VRR:
         choose an axis with positive momentum
         apply lower-angular and m+1 recurrence terms
    apply HRR A->B and C->D to target (a,b|c,d)
    multiply original primitive coefficients and per-component norms
    add into shell-block entries
return block
```

**Eightfold symmetry** is eight permutations of one integral, not eight separate formulas:
\((ij|kl)=(ji|kl)=(ij|lk)=(ji|lk)=(kl|ij)=(lk|ij)=(kl|ji)=(lk|ji)\).
Generate canonical shell quartets with `pair(i,j) >= pair(k,l)`, compute once, scatter to all distinct permutations (some coincide for repeated indices). Verify all eight explicitly for mixed p locations, e.g. `(px,s|py,s)` and `(px,py|px,py)`. Add arbitrary s/p/d primitive tests, even if Stage 2 SCF uses only s/p orbital bases.

Full dense ERI storage costs \(8N_{\rm bf}^4\) bytes; at \(N_{\rm bf}=200\), ~12.8 GB before matrices or scratch. Use a checked allocation limit and plan a shell-quartet provider/direct-Fock interface now. Time for unoptimized quartet assembly is approximately \(O(N_{\rm bf}^4 n_{\rm prim}^4 C_{\rm rec})\) for bounded contraction/angular complexity; AO Fock contraction costs \(O(N_{\rm bf}^4)\) per iteration.

## 2.6 SCF / `faer` / DIIS

**Recommended API:**
```rust
pub struct ScfOptions {
    pub max_iter: usize,
    pub energy_tol: f64,
    pub density_rms_tol: f64,
    pub commutator_tol: f64,
    pub diis_max: usize,
    pub linear_dep_tol: f64,
}
pub struct ScfResult {
    pub energy: f64,
    pub iterations: usize,
    pub density: Matrix,
    pub fock: Matrix,
    pub mo_coeffs: Matrix,
    pub mo_energies: Vec<f64>,
    pub overlap: Matrix,
    pub energy_weighted_density: Matrix,
}
pub fn rhf(mol: &Molecule, basis: &BasisSet, opts: &ScfOptions)
    -> Result<ScfResult, CalcError>;
```
Adapt `ScfResult` to the existing PyO3 outward-facing fields; Rust may retain additional internal state.

**Input guard:** `nelec = sum(Z)-charge`; require positive even electron count, multiplicity 1, \(n_{\rm occ}=n_e/2 \le n_{\rm independent\,AO}\), finite nuclear positions, and no coincident distinct nuclei. Reject negative/near-zero S eigenvalues and handle linear dependencies by canonical orthogonalization with a clearly documented threshold (drop near-null columns; remap dimensions).

Use \(X=U_{\rm kept}\Lambda_{\rm kept}^{-1/2}\), which satisfies \(X^\top S X=I\). The current code calls this “S^-1/2” but actually forms a rectangular/full canonical orthogonalizer, not necessarily a symmetric one; document the distinction.

**SCF pseudocode:**
```text
validate molecule and basis; construct S, H, E_nuc and ERI provider
diagonalize S; build canonical orthogonalizer X
diagonalize X.T*H*X; construct P=2*C_occ*C_occ.T
for iter in 1..max_iter:
    F = H + J[P] - 0.5*K[P]
    E = 0.5*sum_ij P_ij*(H_ij+F_ij) + E_nuc
    R = F*P*S - S*P*F
    if at least two safe DIIS vectors:
        solve constrained linear system [B -1;-1 0] [c;lambda] = [0;-1]
        on near-singularity drop oldest/most dependent vector and retry;
        otherwise use damping/current F (never all-zero coefficients)
    diagonalize X.T*F_extrap*X; form P_new
    compute RMS(P_new-P), delta E, ||R|| and check all thresholds
    update P and continue
on apparent convergence:
    recompute F[P_final], total energy, residual and orbital eigenvectors;
    verify Tr(P*S)=nelec and consistency before success
otherwise return CalcError::NotConverged
```

Use `faer` matrices internally, check `||A U - U diag(e)||`, `||U^T U - I||`, `C^T S C=I`, density electron count, and DIIS recovery on repeated singular vectors. Memory for dense SCF matrices \(O(N^2)\), diagonalization \(O(N^3)\), Fock contraction \(O(N^4)\)/iteration.

## 2.7 Stage 2 tests and merge gates

| Case | Exact geometry / method / basis | Frozen expected | Gate |
|---|---|---|---|
| H2 | bond = 1.4 Bohr, RHF/STO-3G | -1.116714325063 Eh | total absolute error ≤ 1e-8 Eh |
| He | isolated atom, RHF/STO-3G | -2.8077839575 Eh | ≤ 1e-8 Eh |
| Ne | isolated atom, RHF/STO-3G | -126.6045291715 Eh | ≤ 1e-8 Eh, catches p terms |
| H2O | O(0,0,0), H(0,1.432,0), H(1.112,-0.716,0) Bohr | -74.6953191771 Eh | ≤ 1e-8 Eh |
| CH4 | exact geometry must first be frozen from full Python project | **not provided in ZIP** | Python-reference difference ≤ 1e-6 Eh |
| H2/6-31G | same 1.4 Bohr geometry | **generate after basis data correction** | ≤ 1e-8 Eh against matched reference |
| H2O/6-31G | same Szabo geometry | **generate after basis data correction** | ≤ 1e-8 Eh against matched reference |
| mixed p, synthetic d primitive | fixed exponents and non-coincident centers | Python primitive/shell reference values to be generated | s/p/d S,T,V and ERI componentwise |
| Boys n=0..12, T=0,1e-10,0.1,1,10,25,30,40,100,1000 | analytic or SciPy special function | reference arrays | abs/rel ≤ 1e-12 where numerically meaningful |

Every integral must pass Python-reference comparison before merge, plus eightfold ERI symmetry, S/T/V transpose symmetry, molecular translation and rotation of scalar integrals, on-center normalization, and finite-difference identities. For tiny integrals test absolute errors (relative errors near zero are misleading). An independent BSE/PySCF/ORCA check at identical geometry and basis should supplement—not secretly replace—the frozen Python reference.

**Exit criterion:** all Stage 2 gates pass, native support for H/He/C/N/O/Ne RHF/STO-3G is explicitly enabled, correct 6-31G is supported for validated elements, and unsupported features continue to use the Python reference.

---

# 3. Stage 3 — Analytical RHF gradients and native BFGS

## 3.1 Files and signatures

| Path | Tasks / signatures | Depends on |
|---|---|---|
| `crates/calchem-int/src/derivative.rs` (new) | `pub fn one_electron_derivative(engine:&IntegralEngine, atom:usize)->Result<OneElectronDerivative,CalcError>`; `pub fn eri_shell_derivative(quartet:[&Shell;4],atom:usize,out:&mut [f64])->Result<(),CalcError>` | all S2 integral kernels |
| `crates/calchem-grad/src/lib.rs` (replace placeholder) | `pub fn rhf_gradient(mol:&Molecule,basis:&BasisSet,scf:&ScfResult,engine:&IntegralEngine)->Result<Vec<[f64;3]>,CalcError>` | S2 converged SCF |
| `crates/calchem-grad/src/nuclear.rs` | `pub fn nuclear_repulsion_gradient(mol:&Molecule)->Result<Vec<[f64;3]>,CalcError>` | model validation |
| `crates/calchem-opt/src/lib.rs` (replace placeholder) | `pub fn optimize_bfgs<F>(initial:&Molecule,evaluate:F,options:&OptOptions)->Result<OptResult,CalcError> where F: FnMut(&Molecule)->Result<(f64,Vec<[f64;3]>),CalcError>` | S3 gradients |
| `crates/calchem-opt/src/line_search.rs` | `pub fn strong_wolfe(...)` or backtracking Armijo with displacement safeguards | optimizer |
| `crates/calchem-py/src/lib.rs` | new Python-facing `run_gradient`, `optimize_geometry`, preserving old `run_single_point` | S3 completion |
| `crates/calchem-grad/tests/gradient_fd.rs` | H2, H2O, CH4 finite difference gradient validation | all |
| `crates/calchem-opt/tests/h2_bfgs.rs` | H2 equilibrium optimization and monotonic line search | all |

The `F` callback makes the optimizer independently testable on Rosenbrock/quadratic toy functions and avoids a crate cycle. Add an adapter in `calchem-py` or application-level Rust code that reuses the SCF density as the next geometry's initial guess if consistent.

## 3.2 Derivative integral algorithm

For unnormalized Gaussian on a center A and axis i:
\[
\frac{\partial\chi_a}{\partial A_i}
=2\alpha\,\chi_{a+e_i}-a_i\chi_{a-e_i}.
\]
For normalized contracted shells, **use original primitive norm multipliers** when evaluating shifted angular states; if the shifted helper returns a normalized integral, correct for the norm ratio. Do not double-normalize.

Nuclear-attraction derivatives require BOTH derivatives of basis centers and **explicit derivatives of nuclear Coulomb centers**. Apply \(\partial_{C_i}F_m(p|P-C|^2)=2p(P_i-C_i)F_{m+1}\) to the base, and propagate through attraction recurrence. All atoms must be attributed by `shell.atom_index`, and repeated centers correctly summed.

ERI derivatives require differentiating up to four basis-function centers. Gradient angular shifts therefore need integrals one angular momentum higher than energy kernels, even if the production orbital basis ends at p/d.

## 3.3 Total RHF gradient

Let \(P=2C_{\rm occ}C_{\rm occ}^{T}\), \(W=2C_{\rm occ}\operatorname{diag}(\epsilon_{\rm occ})C_{\rm occ}^{T}\), \(h=T+V\). Then the converged, finite-basis gradient includes the Pulay term:
\[
\begin{aligned}
\partial_A E={}&
\sum_{\mu\nu}P_{\mu\nu}\partial_A h_{\mu\nu}
-\sum_{\mu\nu}W_{\mu\nu}\partial_A S_{\mu\nu}\\
&+\tfrac12\sum_{\mu\nu\lambda\sigma}P_{\mu\nu}P_{\lambda\sigma}
\left[\partial_A(\mu\nu|\lambda\sigma)
-\tfrac12\partial_A(\mu\lambda|\nu\sigma)\right]
+\partial_A E_{\rm NN}.
\end{aligned}
\]
For \(E_{\rm NN}=\sum_{A<B}Z_AZ_B/R_{AB}\):
\[
\nabla_A E_{\rm NN}=-\sum_{B\ne A} Z_A Z_B (R_A-R_B)/R_{AB}^{3}.
\]

**Pseudocode `rhf_gradient`:**
```text
require converged RHF result and matched basis/geometry provenance
allocate grads[natoms][3] = nuclear_repulsion_gradient()
construct energy-weighted density W from occupied coefficients/energies
for atom A and Cartesian axis x,y,z:
    dS,dT,dV = derivative AO matrices including nuclear-center dV
    grads[A][axis] += trace(P*(dT+dV)) - trace(W*dS)
    stream dERI shell quartets, accumulating dJ - 0.5*dK
return gradient after all finite/value/symmetry checks
```
Naive cached ERI gradient requires \(O(N_{\rm atom}N_{\rm bf}^4)\) operations and can use \(O(N_{\rm bf}^4)\) temporary memory; streaming derivative quartets retains roughly \(O(N^2)\) persistent memory plus shell scratch. The derivative integral work includes an additional angular level.

**Hellmann–Feynman is not sufficient** for finite atom-centered Gaussian bases: Pulay overlap derivatives are needed. The derivative-integral approach incorporates both.

## 3.4 BFGS optimization algorithm

```text
x = initial Cartesian positions (Bohr); Hinv = scaled identity
repeat max_steps:
    (E,g) = evaluate(x), require finite and SCF converged
    if max_force/rms_force and displacement/energy checks pass: return
    p = -Hinv*g
    if g.p >= 0: reset Hinv=I and use -g
    find safe alpha via Armijo/Wolfe, rejecting SCF failures and atom collisions
    s = alpha*p; y = g_new-g
    if y.s > curvature_threshold:
        Hinv = (I-rho*s*y.T)*Hinv*(I-rho*y*s.T)+rho*s*s.T
    else: skip/reset update
    x += s; record trajectory and convergence statistics
on failure return NotConverged, not successful geometry
```
Dense Cartesian BFGS update: \(O((3N_{\rm atom})^2)\) time/memory per step, excluding expensive SCF/gradient evaluation. L-BFGS can follow for large systems.

## 3.5 Stage 3 tests

- H2 1.4-Bohr, H2O specified Szabo geometry, and a frozen CH4 geometry: compare each gradient component to central differences \([E(R+h)-E(R-h)]/(2h)\), with \(h=10^{-4}\) Bohr and a second step \(5\times10^{-5}\) Bohr; maximum component error ≤ \(10^{-6}\) Eh/Bohr.
- Translational invariance: \(\|\sum_A\nabla_A E\|\) near roundoff; rotational covariance and torque conservation for isolated molecules; exchange-equivalent nuclei give consistent forces.
- H2/STO-3G native BFGS optimized bond length must match a **new frozen Python-reference fixture** (the ZIP does not give a reference equilibrium bond length). Suggested geometry acceptance target: ≤ \(10^{-3}\) Bohr, energy ≤ \(10^{-8}\) Eh, and residual max force within configured tolerance. Run optimized H2 from at least two starting bond lengths.

**Exit criterion:** verified S/T/V/ERI derivatives, Pulay inclusion, molecular gradients to tolerance, native H2 optimizer with no Python finite differences, and a retained Python fallback.

---

# 4. Stage 4 — Performance on Ryzen 7 9700X / 15.1 GB RAM

## 4.1 Files and signatures

| Path | Task / API | Dependency |
|---|---|---|
| `crates/calchem-int/src/screening.rs` | `pub fn schwarz_bounds(basis:&BasisSet)->Result<Vec<f64>,CalcError>`; `pub fn should_skip(pair_ab:f64,pair_cd:f64,threshold:f64)->bool` | S2 generic ERI |
| `crates/calchem-int/src/shell_pairs.rs` | `pub fn build_shell_pairs(basis:&BasisSet)->Vec<ShellPair>` with cached centers, exponents, Gaussian prefactors | basis + S2 |
| `crates/calchem-int/src/direct.rs` | `pub fn build_fock_direct(basis:&BasisSet,mol:&Molecule,density:&Matrix,opts:&IntegralOptions)->Result<Matrix,CalcError>` | S2 + screening |
| `crates/calchem-int/src/parallel.rs` | `pub fn eri_blocks_parallel(...,threads:usize)->Result<...>`; deterministic task batching/scatter | S2 canonical quartets |
| `crates/calchem-tensor/src/simd.rs` | `pub fn matmul_dispatch(a:&Matrix,b:&Matrix)->Result<Matrix,CalcError>` with generic and optional accelerated kernels | S2 faer |
| `crates/calchem-int/src/scratch.rs` | `pub struct IntegralScratch{...}`; per-thread reusable Boys/VRR/HRR buffers | S2 |
| `crates/calchem-scf/src/fock.rs` | choose `DenseStored` vs `DirectScreened` and track work counters | S4 direct |
| `benches/` or `crates/calchem-int/benches/` | shell-ERI, Fock, linear algebra benches using Criterion or `std::time` harness | all |
| `tests/python/test_performance_h2o.py` | pinned Python-ref comparison, medians and environment metadata | all |

## 4.2 Schwarz screening and direct Fock

Precompute pair bounds \(B_{ij}=\sqrt{\max(0,(ij|ij))}\), with a bound interpreted as the appropriate shell-pair operator norm / conservative maximal component bound. Screen shell quartet `(ij|kl)` only when its conservative product bound is below a configured threshold. For strict correctness, test screening error by comparison to unscreened reference and expose the threshold through options.

**Pseudocode:**
```text
pairs = canonical shell pairs, with cached bounds and integral sizes
quartets = canonical pair-pair combinations
parallel by bounded blocks/chunks of quartets:
    for quartet:
        if bound_product < threshold: skip
        compute shell ERI block in worker-local scratch
        accumulate shell-local contributions for J and K
    return worker-local updates
deterministically merge worker updates into global Fock
```
Do **not** allow multiple workers to scatter eightfold copies to the same dense tensor without synchronization. Prefer thread-local accumulation and stable chunk-level reduction for correct results and repeatable tests. The full dense-ERI mode can coexist for small systems, while direct SCF uses \(O(N^2)\) global density/Fock storage plus thread-local scratch rather than \(O(N^4)\) ERI storage.

## 4.3 Parallelism, SIMD, and memory

- Rayon scheduling should work at shell-pair/quartet granularity, not at individual scalar integral or H2O-size microtasks. Default to serial when work is too small; use configurable thread count.
- `faer` provides a suitable pure-Rust matrix backend; benchmark it before writing bespoke AVX-512 GEMM. Restrict hand-written SIMD to demonstrably hot, contiguous primitive/recurrence loops.
- Support a generic x86-64 wheel; build a separate local `-C target-cpu=native` artifact for Zen 5. Only enable AVX-512 paths under runtime CPU detection / safe function-level targeting. Confirm `znver5` availability using `rustc --print target-cpus` on the actual toolchain; do not assume every Rust version recognizes it.
- Use a pool of reusable per-thread scratch arrays sized for the maximum active shell angular momentum. Put hard limits on workspace allocation and avoid `nbf.pow(4)` unchecked overflow.
- Keep numerical test flags and fast-math strategies separate. Do not introduce reassociation that violates the 1e-8 Eh energy gate unless tested, documented, and explicitly opted in.

Asymptotic cost: shell screening worst-case \(O(N_{\rm shell}^4)\) quartet checks, unscreened ERI build \(O(N_{\rm bf}^4)\) at fixed angular complexity, direct Fock persistent memory \(O(N_{\rm bf}^2 + N_{\rm threads}M_{\rm scratch})\), conventional diagonalization \(O(N_{\rm bf}^3)\). Practical screening reduces work but does not change worst-case \(N^4\).

## 4.4 Benchmark protocol / acceptance

Test on Ryzen 7 9700X, Windows 11/MSVC, fixed CPU/power mode, pinned release build, warm-up, 30+ repetitions, and recorded Python/Rust/compiler versions, BLAS threads, hash of input fixture, and tolerances.

**Measure separately:** AO integral generation; Fock build; total SCF; Python-call-to-return path; peak memory; 1/2/4/8 Rust worker scaling. Compare exactly matched H2O geometry, STO-3G, RHF settings, requested outputs and precision.

**Target:** median end-to-end **native H2O/STO-3G speedup ≥10x** against the pinned Python-reference path, but record any failure honestly. The tiny water problem may run faster serially because thread scheduling dominates. Never change tolerances or omit needed work to win the benchmark. Regression gate remains energy ≤1e-8 Eh, gradient ≤1e-6 Eh/Bohr once gradients are optimized.

---

# 5. Stage 5+ — UHF, DFT, TDDFT, and post-HF

## 5.1 Proposed dependency order

1. **Stage 2.5 (optional but recommended before DFT): UHF** to support odd-electron/open-shell molecules; requires a validated ERI/Fock backend and spin-resolved DIIS/density.
2. **Stage 5A: LDA RKS**, with molecular atom-centered quadrature, AO values, density and local XC potentials, and electron-count grid validation.
3. **Stage 5B: PBE RKS/UKS**, adding AO gradients, \(\nabla\rho\), spin densities, and generalized-gradient XC potential.
4. **Stage 5C: B3LYP RKS/UKS**, adds hybrid exact exchange and correctly parameterized functional composition.
5. **Stage 5D: MP2 (conventional → RI-MP2)** after accurate occupied/virtual MOs and AO→MO transformation. MP2 need not wait for DFT; it may run alongside Stage 5A.
6. **Stage 5E: TDDFT/TDA** after validated DFT ground state and functional response kernels; full TDDFT follows TDA.
7. **Stage 5F: CCSD then CCSD(T)** after MP2 infrastructure, memory/blocking architecture, and full amplitude tests.

## 5.2 Files and planned signatures

| Path | Planned function signatures | Complexity / prerequisites |
|---|---|---|
| `crates/calchem-scf/src/uhf.rs` | `pub fn uhf(mol:&Molecule,basis:&BasisSet,opts:&ScfOptions)->Result<UhfResult,CalcError>` | \(O(N^4)\) Fock/iter; Stage 2 |
| `crates/calchem-grid/src/lebedev.rs` | `pub fn angular_grid(order:usize)->Result<Vec<GridPoint>,CalcError>` | \(O(N_{\rm ang})\) generation; weights integrate \(4\pi\) |
| `crates/calchem-grid/src/radial.rs` | `pub fn radial_grid(z:u8,n:usize)->Vec<RadialPoint>` | \(O(N_{\rm rad})\); choose documented Mura–Knowles/Euler–Maclaurin |
| `crates/calchem-grid/src/becke.rs` | `pub fn partition_weights(atoms:&[Atom],points:&mut[GridPoint])` | \(O(N_{\rm atom}^2N_{\rm grid})\) direct |
| `crates/calchem-grid/src/ao.rs` | `pub fn evaluate_ao_batch(basis:&BasisSet,points:&[GridPoint],need_grad:bool)->AoBatch` | \(O(N_{\rm grid}N_{\rm bf}n_{\rm prim})\) |
| `crates/calchem-xc/src/lda.rs` | `pub fn lda_xc(rho_a:f64,rho_b:f64)->XcValue` | \(O(N_{\rm grid})\) |
| `crates/calchem-xc/src/pbe.rs` | `pub fn pbe_xc(rho_a:f64,rho_b:f64,grad_a:[f64;3],grad_b:[f64;3])->XcValue` | \(O(N_{\rm grid})\) |
| `crates/calchem-xc/src/b3lyp.rs` | `pub fn b3lyp_coefficients()->HybridXcParameters` | exact exchange from Stage 2; validate conventions |
| `crates/calchem-dft/src/rks.rs` | `pub fn rks(mol:&Molecule,basis:&BasisSet,xc:&dyn XcFunctional,opts:&DftOptions)->Result<DftResult,CalcError>` | typical XC build \(O(N_{\rm grid}N_{\rm bf}^2)\)/iter |
| `crates/calchem-dft/src/uks.rs` | `pub fn uks(...)->Result<DftResult,CalcError>` | after UHF and RKS |
| `crates/calchem-posthf/src/mp2.rs` | `pub fn mp2(rhf:&ScfResult,integrals:&dyn IntegralProvider)->Result<Mp2Result,CalcError>` | canonical transform often \(O(N^5)\); \(O(N^4)\) possible storage without blocking |
| `crates/calchem-posthf/src/ccsd.rs` | `pub fn ccsd(...)->Result<CcsdResult,CalcError>` | \(O(N^6)\) time, \(O(N^4)\) amplitudes/integrals |
| `crates/calchem-posthf/src/tddft.rs` | `pub fn tda(dft:&DftResult,opts:&TddftOptions)->Result<ExcitationResult,CalcError>`; then `tddft` | requires functional response kernel, costly matrix products |
| `crates/calchem-py/src/lib.rs` | `run_dft`, `run_mp2`, `run_tddft` facade while preserving legacy APIs | after validated native methods |

## 5.3 DFT grid / XC algorithm

```text
for each atom:
    generate positive radial nodes/weights and Lebedev angular nodes/weights
    transform spherical nodes to Bohr Cartesian grid around atom
    assign Becke partition weight so all atom partitions sum to unity
for grid points in small batches:
    evaluate basis values and, for GGA, gradients
    compute spin densities rho_a/rho_b and gradient components from P
    evaluate epsilon_xc and functional partial derivatives
    add weighted XC energy and V_xc (and GGA gradient terms) to AO matrix
SCF: F_RKS = H + J + V_xc; for hybrid B3LYP include specified fraction of HF K
perform checked DIIS/convergence and integrate density electron count
```

Avoid hand-entered Lebedev tables without independent degree/weight validation. Test grid integration of known spherical harmonics/polynomials, partition-of-unity, grid convergence for H2O and CH4, and XC derivatives by finite differences. LDA → PBE → B3LYP only after separate validated fixtures.

The ZIP's `BASELINE.md` lists **unverified** historical PBE -75.5322 and B3LYP -75.5566 Eh for H2O/STO-3G. Do not make these hard acceptance standards until geometry, functional conventions, grid, and matched reference are independently confirmed.

## 5.4 MP2 / CCSD / TDDFT algorithm summaries

**Canonical MP2**
```text
transform AO (mu nu|lambda sigma) to required occupied/virtual MO blocks
for occupied i,j and virtual a,b:
    denom = eps_i+eps_j-eps_a-eps_b
    guard near-zero denominator; do not hide instability
    E2 += (ia|jb) * [2*(ia|jb)-(ib|ja)] / denom
return E_HF + E2, frozen closed-shell reference
```
AO→MO transformation by successive one-index contractions is normally \(O(N^5)\) time and should be blocked to cap memory; direct full four-index transform at once is inefficient.

**CCSD**
```text
start T1/T2 from MBPT amplitudes; form intermediates
build singles/doubles residuals, update amplitudes and DIIS
check residual norm AND energy changes and solver stability
after converged CCSD optionally compute perturbative triples (T)
```
CCSD \(O(N^6)\), perturbative triples usually \(O(N^7)\). The first release should limit molecule/basis size based on estimated memory.

**TDA/TDDFT**
```text
build KS occupied-virtual excitation space
form or apply TDDFT response matrix using Coulomb, exchange and XC kernel
iterative Davidson solve for low excitations; verify oscillator strengths
after TDA works, implement full non-Hermitian response with stability guards
```

**Validation:** use fresh method-specific reference fixtures, exactly matched basis/grid conventions, and at least a second independent engine for representative test cases. Do not promote README-only post-HF numbers into golden tests without rechecking their geometries and definitions.

---

# 6. Build, packaging, fallback, portability

## 6.1 Files and configuration

| Exact path | Action |
|---|---|
| `pyproject.toml` (full repository; create or modify) | PEP 517 maturin backend; select `crates/calchem-py/Cargo.toml`, Python package directory, and `module-name="calchem._core"` |
| `crates/calchem-py/Cargo.toml` | keep `cdylib`, pinned PyO3 0.22 compatibility; preserve abi3-py310 if Python >=3.10 is required |
| `crates/calchem-py/src/lib.rs` | thin argument validation, release GIL during Rust-only calculation with `py.allow_threads` on pinned PyO3, translate typed errors |
| `crates/calchem-basis/src/registry.rs` (new) | compile-time `include_str!` registry of validated embedded basis datasets |
| `crates/calchem-basis/data/*.json` | checked-in canonical data with provenance and hash |
| `calchem/backends/native.py` | feature-aware native→reference routing with explicit `backend_used`, no silent numerical failure masking |
| `.github/workflows/rust-python.yml` (optional new) | native Windows MSVC / Linux / macOS cargo tests, Python wheel build/import and fixture parity |
| `scripts/generate_reference_fixtures.py` (new in full repo) | stable input geometry/basis + expected arrays/energies/gradient plus package versions and input SHA-256 |

Example root `pyproject.toml` fragment (merge with the project's existing configuration, do not blindly overwrite):
```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[tool.maturin]
manifest-path = "crates/calchem-py/Cargo.toml"
python-source = "."
module-name = "calchem._core"
```

Windows workflow in the full source tree:
```powershell
python -m pip install maturin
maturin develop --release -m crates/calchem-py/Cargo.toml
python -c "import calchem._core as c; print(c)"
cargo test --workspace
python -m pytest tests/python/test_native_parity.py -q
```
Check compatible options for the installed maturin version and ensure an activated Python environment for `maturin develop`.

`include_str!` is an excellent default for small, fixed canonical datasets: hermetic Rust binary, no wheel data-file path issue, no runtime Python dependency. Keep version/sha metadata and add JSON parsing validation during tests. An optional external basis-file loader can be added later but should not silently override embedded standard names.

**ABI/build rules:** no manual `.pyd` copy; generic x86-64 binary for distribution, host-optimized local build only for the specified Ryzen CPU; compile and test Windows MSVC natively (Linux→Windows MSVC cross-compilation is not the default safe release path); use `maturin build --release` to create wheels; ensure source distributions contain workspace crates and embedded JSON. Do not upgrade PyO3/faer without dedicated API and regression commits.

## 6.2 Strict Python fallback pseudocode

```text
if request is a supported, numerically certified native method+basis+element/spin:
    try:
        result = rust.run_single_point(...)
        assert result.converged and finite(result.energy)
        result.backend_used = "calchem-rust"
        return result
    except Unsupported:
        pass  # deliberate fallback
    except (NotConverged, Numerical, InvalidInput):
        raise/report: no invisible fallback masking failure
return python_reference.calculate(request)
```

For UX compatibility, if policy supports forced fallback on calculation errors, make it user-opt-in and mark both attempted native and actual Python backend. A backend that returns a numerically wrong yet “converged” energy must be caught by regression; a runtime router cannot prove energy accuracy on its own.

---

# 7. Dependency and PR sequence

| PR | Scope | Must merge after | Critical acceptance |
|---|---|---|---|
| PR-01 | baseline fixtures, strict model/basis validation and exact geometry/unit registry | none | H2/He unaffected; bad inputs rejected |
| PR-02 | correct basis normalization and audited 6-31G data | PR-01 | all S(ii) ~1; published basis verified |
| PR-03 | stable Boys + corrected S/T/V s/p/d | PR-02 | primitive/shell Python parity; no VRR sign issue |
| PR-04 | generic OS ERI + HRR and eightfold symmetry | PR-03 | ssss/spss/pppp and mixed component parity |
| PR-05 | faer SCF, stable DIIS, converged-only result | PR-04 | H2, He, Ne, H2O, CH4 passes |
| PR-06 | Rust/Python bridge + dual-track routing, proper wheels | PR-05 | imported wheel, fallback compatibility, 6-31G parity |
| PR-07 | derivative primitives and full RHF gradient | PR-06 | finite-difference gradients ≤1e-6 Eh/Bohr |
| PR-08 | native BFGS optimization | PR-07 | H2 optimized geometry parity |
| PR-09 | screening/direct-Fock/Rayon and scratch | PR-05, PR-08 for gradient parity | speed + strict accuracy gates |
| PR-10 | UHF + first DFT LDA/grid | PR-09 (or earlier correctness gate) | open-shell and DFT reference validation |
| PR-11 | PBE and hybrid B3LYP | PR-10 | grid/functional reference tests |
| PR-12 | MP2, then TDA/TDDFT and CCSD | stable SCF and per-method prerequisites | method-specific numerical tests |

**Immediate recommended next coding order:** basis normalization → Boys → kinetic → p nuclear VRR → generic p ERI → SCF convergence/DIIS → audited 6-31G → handoff tests. Do not start Rayon, AVX-512, DFT, or analytics until H2O/STO-3G meets the frozen \(10^{-8}\) Eh gate.

---

# 8. Complexity summary

Let \(N\) = AO basis-function count; \(M\) = number of shells; \(K\) = average primitives per shell; \(A\) = atoms; \(G\) = quadrature points; \(I\) = SCF iterations; fixed maximum angular momentum.

| Component | Time, approximate | Persistent memory |
|---|---|---|
| Basis normalization | \(O(MK^2)\) | \(O(MK)\) |
| S/T/V matrices | \(O(M^2K^2 A)\) for attraction, fixed angular recurrence | \(O(N^2)\) |
| Four-center ERIs | \(O(M^4K^4)\) fixed angular recurrence | \(O(N^4)\) if stored |
| RHF (stored ERI) | \(O(M^4K^4 + I(N^4+N^3))\) | \(O(N^4)\) |
| RHF (direct shell Fock) | \(O(I M^4K^4)\) before screening benefits | \(O(N^2)\) plus scratch |
| Analytic full RHF gradient | worst \(O(A N^4)\) contraction plus derivative recurrence | \(O(N^2)\) plus scratch if streamed |
| BFGS optimizer | \(O((3A)^2)\) per optimizer update + evaluation | \(O(A^2)\) |
| RKS XC potential | \(O(GN^2)\)/SCF iteration | \(O(N^2+G_{\rm batch}N)\) |
| Canonical MP2 | \(O(N^5)\) conventional transform | implementation-dependent \(O(N^4)\) unblocked |
| CCSD / (T) | \(O(N^6)\) / \(O(N^7)\) | at least \(O(N^4)\) |

These are worst-case/asymptotic planning estimates, not measured benchmarks. Exact shell recurrence constants can dominate small molecules.

---

# 9. Definition of done / verification status

**Verified by inspection of supplied files:** nine-crate layout, code-level deficiencies listed in §1.1, presence of 6-31G JSON, specified frozen baseline values, code path currently using STO-3G regardless of requested name, and Python reference snapshot interfaces.

**Independently checked without executing Rust:** the existing alternating Boys implementation's cancellation behavior using a SciPy-based numerical comparison; published 6-31G carbon values versus the supplied JSON.

**Not performed:** `cargo test`, MSVC compilation, direct invocation of the complete Python reference application, PyO3 wheel assembly, quantum-chemistry independent external-engine validation, native H2O/CH4 results, gradient/optimization or performance measurements. No such result is claimed. The supplied ZIP omits the full Python package and its original test suite, so new fixture generation belongs in the complete working repository.

**Release gate:** stage claims are accepted only when the committed code, deterministic fixtures, reference-parity tests, and cross-platform build artifacts demonstrate them.

---

# References (external documentation used for implementation recommendations)

- [faer symmetric eigendecomposition documentation](https://docs.rs/faer/latest/faer/)
- [Maturin mixed Rust/Python layout](https://github.com/PyO3/maturin/blob/main/guide/src/project_layout.md)
- [Maturin configuration](https://www.maturin.rs/config)
- [PyO3 0.22 parallelism / `allow_threads`](https://docs.rs/crate/pyo3/0.22.4/source/guide/src/parallelism.md)
- [Basis Set Exchange official API reference](https://molssi-bse.github.io/basis_set_exchange/web_api.html)
- [Basis Set Exchange usage and versioned data](https://molssi-bse.github.io/basis_set_exchange/usage.html)
