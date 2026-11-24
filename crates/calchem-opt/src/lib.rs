use calchem_basis::BasisSet;
use calchem_dft::{GridLevel, XcFunctional};
use calchem_model::{CalcError, Molecule, ScfOptions};
use calchem_tensor::Matrix2D;
use serde::{Deserialize, Serialize};

pub mod irc;
pub mod prfo;

pub use irc::{run_irc, IrcBranch, IrcDirection, IrcOptions, IrcPoint, IrcResult};
pub use prfo::{search_transition_state, TsOptions, TsResult};

/// SCF / Kohn-Sham reference used to evaluate energies and analytical gradients during optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Reference {
    /// RHF for closed-shell singlets, UHF otherwise.
    #[default]
    Auto,
    Rhf,
    Uhf,
    /// Restricted Kohn-Sham DFT with the specified XC functional.
    Rks(XcFunctional),
    /// Unrestricted Kohn-Sham DFT with the specified XC functional.
    Uks(XcFunctional),
    /// Closed-shell RKS for singlets, open-shell UKS otherwise.
    DftAuto(XcFunctional),
}

impl Reference {
    pub fn from_method_and_xc(method: &str, xc: Option<&str>) -> Result<Self, CalcError> {
        let m = method.trim().to_ascii_lowercase();
        let resolve_xc = |default_xc: &str| -> Result<XcFunctional, CalcError> {
            XcFunctional::from_name(xc.unwrap_or(default_xc))
        };
        match m.as_str() {
            "rhf" => Ok(Self::Rhf),
            "uhf" => Ok(Self::Uhf),
            "hf" | "auto" => Ok(Self::Auto),
            "rks" => Ok(Self::Rks(resolve_xc("b3lyp")?)),
            "uks" => Ok(Self::Uks(resolve_xc("b3lyp")?)),
            "dft" => Ok(Self::DftAuto(resolve_xc("b3lyp")?)),
            other => {
                let func = XcFunctional::from_name(other)?;
                Ok(Self::DftAuto(func))
            }
        }
    }

    fn resolve(self, mol: &Molecule) -> Reference {
        match self {
            Reference::Auto => {
                if mol.multiplicity == 1 {
                    Reference::Rhf
                } else {
                    Reference::Uhf
                }
            }
            Reference::DftAuto(xc) => {
                if mol.multiplicity == 1 {
                    Reference::Rks(xc)
                } else {
                    Reference::Uks(xc)
                }
            }
            r => r,
        }
    }
}

mod point;
pub use point::GasPoint;

pub(crate) fn scf_energy(mol: &Molecule, bs: &BasisSet, r: Reference, grid: GridLevel, opts: &ScfOptions) -> Result<f64, CalcError> {
    Ok(GasPoint::solve(mol, bs, r, grid, opts)?.energy())
}

pub fn scf_energy_gradient(mol: &Molecule, bs: &BasisSet, r: Reference, grid: GridLevel, opts: &ScfOptions) -> Result<(f64, Vec<[f64; 3]>), CalcError> {
    let point = GasPoint::solve(mol, bs, r, grid, opts)?;
    Ok((point.energy(), point.gradient(mol, bs)?))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptOptions {
    pub max_steps: usize,
    pub max_force: f64,
    pub rms_force: f64,
    pub max_displacement: f64,
    pub max_step_bohr: f64,
    pub scf_options: ScfOptions,
    pub c1: f64,
    pub backtrack_factor: f64,
    pub max_line_searches: usize,
    #[serde(default)]
    pub reference: Reference,
    #[serde(default)]
    pub grid_level: GridLevel,
    #[serde(default)]
    pub dispersion: Option<String>,
    #[serde(default)]
    pub solvent: Option<String>,
}

impl Default for OptOptions {
    fn default() -> Self {
        let mut scf_opts = ScfOptions::default();
        scf_opts.max_iter = 150;
        scf_opts.conv_energy = 1e-11;
        scf_opts.conv_density = 1e-9;
        Self {
            max_steps: 100,
            max_force: 1e-4,
            rms_force: 5e-5,
            max_displacement: 1e-3,
            max_step_bohr: 0.2,
            scf_options: scf_opts,
            c1: 1e-4,
            backtrack_factor: 0.5,
            max_line_searches: 20,
            reference: Reference::Auto,
            grid_level: GridLevel::Medium,
            dispersion: None,
            solvent: None,
        }
    }
}

pub fn scf_energy_gradient_with_options(
    mol: &Molecule,
    bs: &BasisSet,
    opts: &OptOptions,
) -> Result<(f64, Vec<[f64; 3]>), CalcError> {
    // Gas-phase derivatives are analytic. The optional SCRF surface uses explicit
    // central differences of *fully reconverged total energies*, including CDS.
    // A frozen gas density is not the derivative of either relaxed surface.
    let (mut e, mut g) = if opts.solvent.is_some() {
        let e = solvent_total_energy(mol, bs, opts)?;
        let h = 1e-4;
        let mut g = vec![[0.0; 3]; mol.atoms.len()];
        for a in 0..mol.atoms.len() {
            for q in 0..3 {
                let mut plus = mol.clone();
                let mut minus = mol.clone();
                plus.atoms[a].position_bohr[q] += h;
                minus.atoms[a].position_bohr[q] -= h;
                let bp = BasisSet::from_name(&bs.name, &plus)?;
                let bm = BasisSet::from_name(&bs.name, &minus)?;
                g[a][q] = (solvent_total_energy(&plus, &bp, opts)?
                    - solvent_total_energy(&minus, &bm, opts)?)
                    / (2.0 * h);
            }
        }
        (e, g)
    } else {
        scf_energy_gradient(mol, bs, opts.reference, opts.grid_level, &opts.scf_options)?
    };

    // 1. Dispersion correction (e.g. D3(BJ) or D3(0))
    if opts.dispersion.is_some() {
        let d3 = dispersion_model(opts)?;
        let (e_d3, g_d3) = d3.energy_and_gradient(&mol.atoms, true)?;
        e += e_d3;
        for (ga, gd) in g.iter_mut().zip(g_d3.iter()) {
            ga[0] += gd[0];
            ga[1] += gd[1];
            ga[2] += gd[2];
        }
    }

    Ok((e, g))
}

pub fn scf_energy_with_options(
    mol: &Molecule,
    bs: &BasisSet,
    opts: &OptOptions,
) -> Result<f64, CalcError> {
    let mut e = if opts.solvent.is_some() {
        solvent_total_energy(mol, bs, opts)?
    } else {
        scf_energy(mol, bs, opts.reference, opts.grid_level, &opts.scf_options)?
    };
    if opts.dispersion.is_some() {
        e += dispersion_model(opts)?.energy(&mol.atoms)?;
    }
    Ok(e)
}

fn dispersion_model(opts: &OptOptions) -> Result<calchem_dft::D3Model, CalcError> {
    let functional = match opts.reference {
        Reference::Rks(xc) | Reference::Uks(xc) | Reference::DftAuto(xc) => xc.canonical_name(),
        _ => "hf",
    };
    calchem_dft::D3Model::from_scheme(opts.dispersion.as_deref().unwrap_or(""), functional)
}

fn solvent_total_energy(
    mol: &Molecule,
    bs: &BasisSet,
    opts: &OptOptions,
) -> Result<f64, CalcError> {
    let solvent = calchem_solv::Solvent::from_name(opts.solvent.as_deref().unwrap_or(""))
        .ok_or_else(|| CalcError::InvalidMolecule("Unknown solvent".into()))?;
    let (method, xc) = match opts.reference.resolve(mol) {
        Reference::Rhf | Reference::Auto => ("rhf", None),
        Reference::Rks(xc) => ("rks", Some(xc)),
        _ => {
            return Err(CalcError::UnsupportedMethod(
                "Open-shell SCRF gradients are not implemented; use an external backend".into(),
            ))
        }
    };
    let solv_opts = calchem_solv::SolvationOptions {
        solvent,
        ..Default::default()
    };
    let res = calchem_solv::run_scrf(mol, bs, &opts.scf_options, &solv_opts, method, xc)?;
    if !res.converged {
        return Err(CalcError::NotConverged {
            iterations: opts.scf_options.max_iter,
            delta_e: f64::INFINITY,
            rms_d: f64::INFINITY,
        });
    }
    Ok(res.solvated_energy)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptStep {
    pub step: usize,
    pub energy_hartree: f64,
    pub max_force: f64,
    pub rms_force: f64,
    pub max_displacement: f64,
    #[serde(default)]
    pub scf_iterations: usize,
    #[serde(default)]
    pub line_search_trials: usize,
    #[serde(default)]
    pub step_alpha: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptResult {
    #[serde(default)]
    pub final_scf: Option<calchem_model::ScfSummary>,
    pub molecule: Molecule,
    pub energy_hartree: f64,
    pub gradient: Vec<[f64; 3]>,
    pub steps: usize,
    pub converged: bool,
    pub history: Vec<OptStep>,
}

pub(crate) fn coords_from_mol(mol: &Molecule) -> Vec<f64> {
    let mut coords = Vec::with_capacity(mol.atoms.len() * 3);
    for atom in &mol.atoms {
        coords.extend_from_slice(&atom.position_bohr);
    }
    coords
}

pub(crate) fn mol_with_coords(base_mol: &Molecule, coords: &[f64]) -> Molecule {
    let mut mol = base_mol.clone();
    for (i, atom) in mol.atoms.iter_mut().enumerate() {
        atom.position_bohr = [coords[3 * i], coords[3 * i + 1], coords[3 * i + 2]];
    }
    mol
}

pub(crate) fn mat_vec_mul(mat: &Matrix2D, vec: &[f64]) -> Vec<f64> {
    let n = vec.len();
    let mut res = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += mat.get(i, j) * vec[j];
        }
        res[i] = sum;
    }
    res
}

pub(crate) fn dot_prod(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Optimizes the molecular geometry using the requested HF or DFT method and basis set.
/// Returns the optimized `Molecule`.
pub fn optimize_geometry(
    mol: &Molecule,
    basis: &BasisSet,
    method: &str,
) -> Result<Molecule, CalcError> {
    let reference = Reference::from_method_and_xc(method, None)?;
    let opts = OptOptions {
        reference,
        ..OptOptions::default()
    };
    let res = optimize_geometry_with_options(mol, &basis.name, &opts)?;
    Ok(res.molecule)
}

/// Restart state immediately after an accepted step, before evaluating the next direction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptCheckpoint {
    pub molecule: Molecule,
    pub inverse_hessian: Vec<f64>,
    pub history: Vec<OptStep>,
    pub gradient: Vec<[f64; 3]>,
    pub final_scf: Option<calchem_model::ScfSummary>,
}

pub trait OptObserver {
    fn step(&mut self, _mol: &Molecule, _h: &Matrix2D, _history: &[OptStep], _gradient: &[[f64; 3]], _scf: Option<&calchem_model::ScfSummary>) -> Result<(), CalcError> { Ok(()) }
    fn trial(&mut self, _step: usize, _attempt: usize, _alpha: f64, _energy: Result<f64, &CalcError>, _accepted: bool) {}
}
impl OptObserver for () {}

struct TrialPoint {
    basis: BasisSet,
    gas: Option<GasPoint>,
    energy: f64,
}

impl TrialPoint {
    fn solve(
        mol: &Molecule,
        basis_name: &str,
        opts: &OptOptions,
        guess: Option<&calchem_scf::InitialGuess>,
    ) -> Result<Self, CalcError> {
        let basis = BasisSet::from_name(basis_name, mol)?;
        let gas = if opts.solvent.is_none() {
            Some(GasPoint::solve_with_guess(mol, &basis, opts.reference, opts.grid_level, &opts.scf_options, guess)?)
        } else { None };
        let mut energy = match &gas {
            Some(g) => g.energy(),
            None => solvent_total_energy(mol, &basis, opts)?,
        };
        if opts.dispersion.is_some() { energy += dispersion_model(opts)?.energy(&mol.atoms)?; }
        Ok(Self { basis, gas, energy })
    }

    fn derivatives(self, mol: &Molecule, opts: &OptOptions) -> Result<(f64, Vec<[f64; 3]>, Option<calchem_model::ScfSummary>, Option<calchem_scf::InitialGuess>), CalcError> {
        if let Some(gas) = self.gas {
            let guess = Some(gas.mo_guess());
            let mut g = gas.gradient(mol, &self.basis)?;
            if opts.dispersion.is_some() { dispersion_model(opts)?.add_gradient(&mol.atoms, &mut g)?; }
            Ok((self.energy, g, Some(gas.summary()), guess))
        } else {
            let (e, g) = scf_energy_gradient_with_options(mol, &self.basis, opts)?;
            Ok((e, g, None, None))
        }
    }
}

pub fn optimize_geometry_with_options(mol: &Molecule, basis_name: &str, opts: &OptOptions) -> Result<OptResult, CalcError> {
    optimize_geometry_observed(mol, basis_name, opts, None, &mut ())
}

pub fn optimize_geometry_observed(
    mol: &Molecule, basis_name: &str, opts: &OptOptions, resume: Option<OptCheckpoint>, observer: &mut impl OptObserver,
) -> Result<OptResult, CalcError> {
    let n_atoms = mol.atoms.len();
    if n_atoms == 0 { return Err(CalcError::InvalidMolecule("No atoms in molecule".into())); }
    if [opts.max_force, opts.rms_force, opts.max_displacement, opts.max_step_bohr].iter().any(|v| !v.is_finite() || *v <= 0.0)
        || !(0.0 < opts.c1 && opts.c1 < 1.0 && 0.0 < opts.backtrack_factor && opts.backtrack_factor < 1.0)
        || opts.max_line_searches == 0 {
        return Err(CalcError::InvalidMolecule("Invalid optimizer thresholds or line search limits".into()));
    }

    let dim = n_atoms * 3;
    let mut x = coords_from_mol(mol);
    let mut h_inv = Matrix2D::zeros(dim, dim);
    for i in 0..dim { h_inv.set(i, i, 0.7); }
    let mut history = Vec::new();
    let mut last_displacement = 0.0;
    let mut last_ls_trials = 0usize;
    let mut last_alpha = 0.0f64;
    let mut cached_eg = None;
    let mut prev_guess: Option<calchem_scf::InitialGuess> = None;
    let mut first_step = 0;
    if let Some(restart) = resume {
        let last = restart.history.last().ok_or_else(|| CalcError::InvalidMolecule("Empty optimizer checkpoint".into()))?;
        if restart.molecule.atoms.len() != n_atoms || restart.gradient.len() != n_atoms || restart.inverse_hessian.len() != dim * dim
            || restart.molecule.charge != mol.charge || restart.molecule.multiplicity != mol.multiplicity
            || !restart.molecule.atoms.iter().zip(&mol.atoms).all(|(a,b)| a.z == b.z && a.symbol == b.symbol && a.position_bohr.iter().all(|x| x.is_finite()))
            || restart.gradient.iter().flatten().chain(&restart.inverse_hessian).any(|v| !v.is_finite())
            || !last.energy_hartree.is_finite() || last.step > opts.max_steps || last.step != restart.history.len() - 1 {
            return Err(CalcError::InvalidMolecule("Incompatible optimizer checkpoint".into()));
        }
        first_step = last.step;
        last_displacement = last.max_displacement;
        last_ls_trials = last.line_search_trials;
        last_alpha = last.step_alpha;
        cached_eg = Some((last.energy_hartree, restart.gradient, restart.final_scf, None));
        history = restart.history;
        history.pop();
        x = coords_from_mol(&restart.molecule);
        h_inv.data = restart.inverse_hessian;
    }
    for step in first_step..=opts.max_steps {
        let cur_mol = mol_with_coords(mol, &x);
        let (current_e, current_g_3d, final_scf, step_guess) = match cached_eg.take() {
            Some(eg) => eg,
            None => TrialPoint::solve(&cur_mol, basis_name, opts, prev_guess.as_ref())?.derivatives(&cur_mol, opts)?,
        };
        if step_guess.is_some() {
            prev_guess = step_guess;
        }
        let mut g = Vec::with_capacity(dim);
        for atom_g in &current_g_3d {
            g.extend_from_slice(atom_g);
        }

        let max_force = g.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let sum_sq_g: f64 = g.iter().map(|v| v * v).sum();
        let rms_force = (sum_sq_g / (dim as f64)).sqrt();

        history.push(OptStep {
            step,
            energy_hartree: current_e,
            max_force,
            rms_force,
            max_displacement: last_displacement,
            scf_iterations: final_scf.as_ref().map(|s| s.iterations).unwrap_or(0),
            line_search_trials: last_ls_trials,
            step_alpha: last_alpha,
        });

        observer.step(&cur_mol, &h_inv, &history, &current_g_3d, final_scf.as_ref())?;
        if max_force < opts.max_force && rms_force < opts.rms_force
            && (step == 0 || last_displacement < opts.max_displacement) {
            return Ok(OptResult {
                final_scf,
                molecule: cur_mol,
                energy_hartree: current_e,
                gradient: current_g_3d,
                steps: step,
                converged: true,
                history,
            });
        }

        if step == opts.max_steps {
            return Err(CalcError::Internal(format!("Geometry optimization reached step limit {step} (max_force={max_force:.6e}, rms_force={rms_force:.6e}, displacement={last_displacement:.6e})")));
        }

        let mut p = mat_vec_mul(&h_inv, &g);
        for v in p.iter_mut() {
            *v = -*v;
        }

        let mut g_dot_p = dot_prod(&g, &p);
        if g_dot_p >= 0.0 {
            for i in 0..dim {
                for j in 0..dim {
                    h_inv.set(i, j, if i == j { 0.7 } else { 0.0 });
                }
            }
            p = g.iter().map(|v| -0.7 * v).collect();
            g_dot_p = dot_prod(&g, &p);
        }

        let p_norm = dot_prod(&p, &p).sqrt();
        if p_norm > opts.max_step_bohr && p_norm > 1e-12 {
            let scale = opts.max_step_bohr / p_norm;
            for v in p.iter_mut() {
                *v *= scale;
            }
            g_dot_p *= scale;
        }

        let mut alpha = 1.0;
        let mut accepted = None;
        let mut ls_trials_used = 0usize;
        let mut accepted_alpha = 0.0f64;
        let mut x_new = x.clone();

        for ls in 0..opts.max_line_searches {
            ls_trials_used = ls + 1;
            for i in 0..dim {
                x_new[i] = x[i] + alpha * p[i];
            }

            let trial_mol = mol_with_coords(mol, &x_new);
            let trial = TrialPoint::solve(&trial_mol, basis_name, opts, prev_guess.as_ref());
            let ok = trial.as_ref().is_ok_and(|t| t.energy <= current_e + opts.c1 * alpha * g_dot_p);
            observer.trial(step, ls + 1, alpha, trial.as_ref().map(|p| p.energy), ok);
            match trial {
                Ok(t) if ok => {
                    accepted_alpha = alpha;
                    accepted = Some(t);
                    break;
                }
                Ok(_) | Err(CalcError::NotConverged { .. }) => {}
                Err(e) => return Err(e),
            }
            alpha *= opts.backtrack_factor;
        }

        if accepted.is_none() {
            let mut is_diagonal = true;
            for i in 0..dim {
                for j in 0..dim {
                    if i != j && h_inv.get(i, j).abs() > 1e-10 {
                        is_diagonal = false;
                        break;
                    }
                }
                if !is_diagonal {
                    break;
                }
            }

            if !is_diagonal {
                for i in 0..dim {
                    for j in 0..dim {
                        h_inv.set(i, j, if i == j { 0.7 } else { 0.0 });
                    }
                }
                let small_step = 0.05;
                ls_trials_used = opts.max_line_searches + 1;
                for i in 0..dim {
                    x_new[i] = x[i] - small_step * g[i];
                }
                let trial_mol = mol_with_coords(mol, &x_new);
                let trial = TrialPoint::solve(&trial_mol, basis_name, opts, prev_guess.as_ref());
                let ok = trial.as_ref().is_ok_and(|p| p.energy <= current_e);
                observer.trial(step, ls_trials_used, small_step, trial.as_ref().map(|p| p.energy), ok);
                match trial {
                    Ok(t) if ok => {
                        accepted_alpha = small_step;
                        accepted = Some(t);
                    }
                    Ok(_) | Err(CalcError::NotConverged { .. }) => {}
                    Err(e) => return Err(e),
                }
            }

            if accepted.is_none() {
                return Err(CalcError::Internal(format!(
                    "Geometry optimization line search stalled at step {step}"
                )));
            }
        }

        let mut s = vec![0.0; dim];
        for i in 0..dim {
            s[i] = x_new[i] - x[i];
        }

        let mut max_atom_disp = 0.0;
        for a in 0..n_atoms {
            let dx = s[3 * a];
            let dy = s[3 * a + 1];
            let dz = s[3 * a + 2];
            let disp = (dx * dx + dy * dy + dz * dz).sqrt();
            if disp > max_atom_disp {
                max_atom_disp = disp;
            }
        }
        last_displacement = max_atom_disp;
        last_ls_trials = ls_trials_used;
        last_alpha = accepted_alpha;

        let new_mol = mol_with_coords(mol, &x_new);
        let (new_e, new_g_3d, scf, new_guess) = accepted.ok_or_else(|| CalcError::Internal("Missing accepted trial".into()))?.derivatives(&new_mol, opts)?;
        let mut g_new = Vec::with_capacity(dim);
        for ag in &new_g_3d {
            g_new.extend_from_slice(ag);
        }
        cached_eg = Some((new_e, new_g_3d, scf, new_guess));

        let mut y = vec![0.0; dim];
        for i in 0..dim {
            y[i] = g_new[i] - g[i];
        }

        let s_dot_y = dot_prod(&s, &y);
        if s_dot_y > 1e-10 {
            let rho = 1.0 / s_dot_y;

            // Symmetric rank-two inverse BFGS update, O((3N)^2).
            let hy = mat_vec_mul(&h_inv, &y);
            let coeff = rho * (1.0 + rho * dot_prod(&y, &hy));
            for i in 0..dim {
                for j in 0..=i {
                    let next = h_inv.get(i,j) + coeff * s[i] * s[j] - rho * (hy[i]*s[j] + s[i]*hy[j]);
                    h_inv.set(i,j,next);
                    h_inv.set(j,i,next);
                }
            }
        } else {
            for i in 0..dim {
                for j in 0..dim {
                    h_inv.set(i, j, if i == j { 0.7 } else { 0.0 });
                }
            }
        }

        x = x_new;
    }

    Err(CalcError::NotConverged {
        iterations: opts.max_steps,
        delta_e: 0.0,
        rms_d: last_displacement,
    })
}
