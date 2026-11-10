# Native scientific method boundaries — 2026-11-10

Native gas-phase RHF/UHF and SVWN5/BLYP/PBE/B3LYP/B3LYP5/PBE0 RKS/UKS
use Gaussian integrals and numerical XC quadrature. UHF/UKS report `<S^2>`;
they do not perform spin projection or a general SCF stability analysis.
Convergence establishes an SCF stationary solution, not the global electronic
ground state. Broken-symmetry guesses and near-degeneracies require independent
checks. Embedded basis sets are STO-3G, 6-31G, 6-31G(d), 6-31G(d,p), LANL2DZ
and LANL2DZdp; other names are not evidence of native support.

**CAM-B3LYP and wB97X(-D) are unavailable in the native engine.** The required
erf/erfc range-separated ERIs have not been implemented. Request Gaussian/ORCA
explicitly. Neither a global hybrid with a fixed exchange fraction nor an
extra dispersion term can reproduce these functionals. Legacy Rust enum values
are rejected by native SCF entry points.

Native D3(BJ)/D3(0) uses standard pair reference C6 tables, CN weighting, r4/r2
moments, functional parameters and analytic derivatives. It is **two-body D3**;
ATM three-body dispersion is absent. Coordinates and gradients use Bohr and
Hartree/Bohr. Unknown functionals/damping and wrapper custom parameters raise
errors. There is no approximate Python fallback. Reference-data source, hashes,
licenses and corresponding source are under `crates/calchem-dft/data/d3/` and
shipped in `calchem/data/d3/`.

C-PCM/COSMO are research implementations with discrete Bondi/Lebedev cavities.
Only closed-shell RHF/RKS SCRF is implemented. Open-shell SCRF, full IEF-PCM,
Onsager and standard SMD are not provided by the native boundary solver.
Post-SCF PCM and SCRF are distinct calculations. The independent gas energy
must converge; failure propagates instead of changing the reference.
`run_solvation_gradient` is a **fixed-AO-density partial finite difference**
(electrostatic + CDS), not a relaxed SCF gradient. Its analytic alternative is
explicitly unavailable. Solvent geometry optimization uses fully reconverged
total SCRF energy differences, including CDS. This is computationally expensive
and remains sensitive to cavity topology changes. No analytic solvent gradient
or smooth standard SMD surface is claimed.

Gas-phase gradients are analytic; harmonic Hessians differentiate those
gradients numerically. TS acceptance recomputes a fresh Hessian, projects rigid
motion in mass-weighted coordinates, and requires exactly one resolved negative
vibrational curvature. Modes with magnitude below 10 cm^-1 are treated as
unresolved for certification. A Bofill search Hessian is not an acceptance
certificate. IRC returns coherent coordinates/energies/forces and reports an
incomplete branch if constrained steps or endpoints do not converge. Partial
paths do not establish reaction barriers or endpoint thermochemistry.

DFT radial grids cover the full radial mapping without a fixed 18/50-Bohr cut.
Fine grids have at least 150 radial nodes, with extra nodes for diffuse basis
exponents. This does not certify arbitrary multi-center/angular quadrature:
verify electron counts and grid convergence for diffuse/charged/CT systems.

Stored ERIs still have quartic worst-case storage. SCF uses a conservative
admission estimate (default 4 GiB; explicit `CALCHEM_MAX_ERI_BYTES` override).
This is not an operating-system memory/time quota or a direct/RI algorithm.
Run large or untrusted jobs in isolated worker processes with host resource
limits. GIL release enables Python concurrency but does not make OOM recoverable.

Windows CPython acceptance does not establish Linux/macOS wheel correctness.
Source packaging rebuilds Rust into the build directory and emits a platform
wheel with the abi3-py310 contract; installation must still be tested on each
supported OS/Python version before release. The repository remains research
software until the independent external acceptance matrix is sufficiently broad.
