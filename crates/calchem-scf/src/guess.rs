//! Superposition of Atomic Densities (SAD) initial guess engine in pure Rust.
//!
//! Provides a spherically averaged, atomic-UHF-solved initial density matrix $P^{(0)}$
//! in each atom's exact molecular basis subset (including ECPs), preserving strict $SO(3)$
//! rotational invariance and satisfying $\text{Tr}(P S) = N_{\text{elec}}$.

use crate::diis::DiisState;
use calchem_basis::{AtomEcp, BasisSet, Primitive, Shell, ShellType};
use calchem_int::{compute_ecp, compute_kinetic, compute_nuclear, compute_overlap, compute_sparse_eri};
use calchem_model::{Atom, CalcError, Molecule};
use calchem_tensor::{eigh_sym, frobenius_norm, matmul, rms_diff, transpose, Matrix2D};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

const MAX_ATOMIC_SAD_CACHE_ENTRIES: usize = 256;

#[inline]
fn f64_bits(v: f64) -> u64 {
    if v == 0.0 {
        0.0f64.to_bits()
    } else {
        v.to_bits()
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct EcpKeyBits {
    core_electrons: u8,
    l_max: u8,
    local_terms: Vec<(u8, u64, u64)>,
    semilocal_channels: Vec<(u8, Vec<(u8, u64, u64)>)>,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct AtomicSadKey {
    z: u8,
    core_electrons: u8,
    shells: Vec<(u8, Vec<(u64, u64)>)>,
    ecp: Option<EcpKeyBits>,
}

static ATOMIC_SAD_CACHE: OnceLock<Mutex<HashMap<AtomicSadKey, Arc<Vec<f64>>>>> = OnceLock::new();

fn shell_type_id(st: ShellType) -> u8 {
    match st {
        ShellType::S => 0,
        ShellType::P => 1,
        ShellType::D => 2,
        ShellType::F => 3,
    }
}

fn make_atomic_sad_key(
    z: u8,
    n_core: u8,
    atom_shells: &[&Shell],
    atom_ecp: Option<&AtomEcp>,
) -> AtomicSadKey {
    let shells = atom_shells
        .iter()
        .map(|sh| {
            let prims = sh
                .primitives
                .iter()
                .map(|p| (f64_bits(p.exponent), f64_bits(p.coeff)))
                .collect();
            (shell_type_id(sh.shell_type), prims)
        })
        .collect();
    let ecp = atom_ecp.map(|e| EcpKeyBits {
        core_electrons: e.core_electrons,
        l_max: e.l_max,
        local_terms: e
            .local_terms
            .iter()
            .map(|t| (t.r_power, f64_bits(t.exponent), f64_bits(t.coeff)))
            .collect(),
        semilocal_channels: e
            .semilocal_channels
            .iter()
            .map(|ch| {
                (
                    ch.angular_momentum,
                    ch.terms
                        .iter()
                        .map(|t| (t.r_power, f64_bits(t.exponent), f64_bits(t.coeff)))
                        .collect(),
                )
            })
            .collect(),
    });
    AtomicSadKey {
        z,
        core_electrons: n_core,
        shells,
        ecp,
    }
}

/// Optional converged orbital coefficients from a nearby geometry used to warm-start SCF.
#[derive(Debug, Clone)]
pub enum InitialGuess {
    /// Restricted MO coefficients (row-major `n_ao x n_mo`, occupied orbitals in columns `0..n_occ`).
    Restricted {
        n_ao: usize,
        mo_coefficients: Vec<f64>,
    },
    /// Unrestricted alpha and beta MO coefficients (each row-major `n_ao x n_mo`).
    Unrestricted {
        n_ao: usize,
        mo_alpha: Vec<f64>,
        mo_beta: Vec<f64>,
    },
}

/// Projects occupied orbitals `C_old[:, 0..n_occ]` onto the new geometry's overlap matrix `s_new`
/// via symmetric Löwdin orthogonalization: `C_proj = C_old * (C_old^T S_new C_old)^(-1/2)`,
/// returning `P = scale * C_proj C_proj^T` (`scale = 2.0` for restricted, `1.0` for unrestricted).
///
/// Validates positive-definiteness of the projected metric and electron count conservation
/// `|Tr(P S_new) - scale * n_occ| < 1e-6`, returning `None` so the caller safely falls back to SAD.
pub fn project_occupied_density(
    c_old: &[f64],
    n_ao: usize,
    n_occ: usize,
    s_new: &Matrix2D,
    scale: f64,
) -> Option<Matrix2D> {
    if n_ao == 0 || n_occ == 0 || n_occ > n_ao || s_new.rows != n_ao || s_new.cols != n_ao {
        return None;
    }
    if c_old.len() != n_ao * n_ao || c_old.iter().any(|v| !v.is_finite()) {
        return None;
    }

    // Extract C_occ (n_ao x n_occ)
    let mut c_occ = Matrix2D::zeros(n_ao, n_occ);
    for mu in 0..n_ao {
        let row = &c_old[mu * n_ao..(mu + 1) * n_ao];
        for i in 0..n_occ {
            c_occ.set(mu, i, row[i]);
        }
    }

    // Occupied subspace metric M = C_occ^T S_new C_occ (n_occ x n_occ)
    let sc = matmul(s_new, &c_occ);
    let m = matmul(&transpose(&c_occ), &sc);
    let (evals, evecs) = eigh_sym(&m, 100, 1e-13).ok()?;
    if evals.iter().any(|&v| !v.is_finite() || v <= 1e-8) {
        return None;
    }

    let mut inv_sqrt = Matrix2D::zeros(n_occ, n_occ);
    for (i, &val) in evals.iter().enumerate() {
        inv_sqrt.set(i, i, 1.0 / val.sqrt());
    }
    let m_inv_sqrt = matmul(&matmul(&evecs, &inv_sqrt), &transpose(&evecs));
    let c_proj = matmul(&c_occ, &m_inv_sqrt);

    let mut p = matmul(&c_proj, &transpose(&c_proj));
    for v in &mut p.data {
        *v *= scale;
        if !v.is_finite() {
            return None;
        }
    }

    // Verify exact electron count conservation Tr(P * S_new) == scale * n_occ
    let mut tr_ps = 0.0;
    for mu in 0..n_ao {
        for nu in 0..n_ao {
            tr_ps += p.get(mu, nu) * s_new.get(nu, mu);
        }
    }
    let target = scale * (n_occ as f64);
    if !tr_ps.is_finite() || (tr_ps - target).abs() > 1e-6 {
        return None;
    }
    Some(p)
}

/// Returns standard ground-state Aufbau neutral-atom subshell electron counts for $Z \in [1, 86]$.
/// Returns `(s_subshells, p_subshells, d_subshells, f_subshells)` where:
/// - `s`: 1s, 2s, 3s, 4s, 5s, 6s, 7s
/// - `p`: 2p, 3p, 4p, 5p, 6p, 7p
/// - `d`: 3d, 4d, 5d, 6d
/// - `f`: 4f, 5f
pub fn atomic_subshell_occupations(z: u8) -> ([f64; 7], [f64; 6], [f64; 4], [f64; 2]) {
    let mut s = [0.0; 7];
    let mut p = [0.0; 6];
    let mut d = [0.0; 4];
    let mut f = [0.0; 2];

    // Standard Madelung rule / Aufbau filling order with common ground states:
    let fill_order: &[(u8, usize, f64)] = &[
        (0, 0, 2.0),  // 1s
        (0, 1, 2.0),  // 2s
        (1, 0, 6.0),  // 2p
        (0, 2, 2.0),  // 3s
        (1, 1, 6.0),  // 3p
        (0, 3, 2.0),  // 4s
        (2, 0, 10.0), // 3d
        (1, 2, 6.0),  // 4p
        (0, 4, 2.0),  // 5s
        (2, 1, 10.0), // 4d
        (1, 3, 6.0),  // 5p
        (0, 5, 2.0),  // 6s
        (3, 0, 14.0), // 4f
        (2, 2, 10.0), // 5d
        (1, 4, 6.0),  // 6p
        (0, 6, 2.0),  // 7s
        (3, 1, 14.0), // 5f
        (2, 3, 10.0), // 6d
        (1, 5, 6.0),  // 7p
    ];

    let mut remaining = z as f64;
    for &(stype, idx, cap) in fill_order {
        if remaining <= 0.0 {
            break;
        }
        let take = remaining.min(cap);
        match stype {
            0 => s[idx] = take,
            1 => p[idx] = take,
            2 => d[idx] = take,
            3 => f[idx] = take,
            _ => unreachable!(),
        }
        remaining -= take;
    }

    (s, p, d, f)
}

/// Distributes atomic subshell electron counts across the shells of that angular momentum
/// defined for atom $A$ in `basis`.
fn distribute_subshells(occupied: &[f64], n_shells: usize) -> Vec<f64> {
    if n_shells == 0 {
        return Vec::new();
    }
    // Filter non-zero occupied subshells
    let active: Vec<f64> = occupied.iter().copied().filter(|&v| v > 0.0).collect();
    if active.is_empty() {
        return vec![0.0; n_shells];
    }

    let mut out = vec![0.0; n_shells];
    let n_active = active.len();

    if n_shells == n_active {
        // Exact minimal basis match (e.g. STO-3G)
        out.copy_from_slice(&active);
    } else if n_shells > n_active {
        // Split-valence or polarized basis (e.g. 6-31G, def2-SVP)
        // Core subshells (0..n_active - 1) get 1-to-1 assignment.
        // The valence subshell (last active) is split evenly across remaining valence shells.
        for i in 0..(n_active - 1) {
            out[i] = active[i];
        }
        let valence_electrons = active[n_active - 1];
        let n_valence_shells = n_shells - (n_active - 1);
        let per_shell = valence_electrons / (n_valence_shells as f64);
        for i in (n_active - 1)..n_shells {
            out[i] = per_shell;
        }
    } else {
        // ECP or truncated valence basis: assign to outermost available shells
        let skip = n_active - n_shells;
        for i in 0..n_shells {
            out[i] = active[skip + i];
        }
    }

    out
}

/// Computes valence-adjusted subshell occupations by removing `n_core` closed-shell core electrons.
fn atomic_subshell_occupations_valence(
    z: u8,
    n_core: u8,
) -> ([f64; 7], [f64; 6], [f64; 4], [f64; 2]) {
    let (mut s, mut p, mut d, mut f) = atomic_subshell_occupations(z);
    let mut rem_core = n_core as f64;
    // Remove closed core subshells in physical shell-energy order (1s, 2s, 2p, 3s, 3p, 3d, 4s, 4p, 4d, ...)
    let core_order: &[(u8, usize)] = &[
        (0, 0), // 1s
        (0, 1), // 2s
        (1, 0), // 2p
        (0, 2), // 3s
        (1, 1), // 3p
        (2, 0), // 3d
        (0, 3), // 4s
        (1, 2), // 4p
        (2, 1), // 4d
        (0, 4), // 5s
        (1, 3), // 5p
        (3, 0), // 4f
        (2, 2), // 5d
        (0, 5), // 6s
        (1, 4), // 6p
    ];
    for &(stype, idx) in core_order {
        if rem_core <= 0.0 {
            break;
        }
        let slot = match stype {
            0 => &mut s[idx],
            1 => &mut p[idx],
            2 => &mut d[idx],
            3 => &mut f[idx],
            _ => unreachable!(),
        };
        let drop_e = (*slot).min(rem_core);
        *slot -= drop_e;
        rem_core -= drop_e;
    }
    (s, p, d, f)
}

/// Returns neutral ground-state `(n_alpha_atom, n_beta_atom)` after subtracting `n_core` ECP electrons,
/// following Hund's rule in the outermost partially filled subshell.
fn atomic_valence_spin_counts(z: u8, n_core: u8) -> Option<(usize, usize)> {
    let z_eff = (z as i32) - (n_core as i32);
    if z_eff <= 0 {
        return None;
    }
    let z_eff = z_eff as usize;
    let fill_order: &[(usize, usize)] = &[
        (1, 2),  // 1s
        (1, 2),  // 2s
        (3, 6),  // 2p
        (1, 2),  // 3s
        (3, 6),  // 3p
        (1, 2),  // 4s
        (5, 10), // 3d
        (3, 6),  // 4p
        (1, 2),  // 5s
        (5, 10), // 4d
        (3, 6),  // 5p
        (1, 2),  // 6s
        (7, 14), // 4f
        (5, 10), // 5d
        (3, 6),  // 6p
        (1, 2),  // 7s
        (7, 14), // 5f
        (5, 10), // 6d
        (3, 6),  // 7p
    ];
    let mut rem = z as usize;
    let mut unpaired = 0usize;
    for &(n_orb, cap) in fill_order {
        if rem == 0 {
            break;
        }
        let take = rem.min(cap);
        if take < cap {
            let alpha = take.min(n_orb);
            let beta = take - alpha;
            unpaired += alpha - beta;
        }
        rem -= take;
    }
    if unpaired > z_eff || (z_eff - unpaired) % 2 != 0 {
        unpaired = z_eff % 2;
    }
    let n_beta = (z_eff - unpaired) / 2;
    let n_alpha = n_beta + unpaired;
    Some((n_alpha, n_beta))
}

/// Forms an SO(3)-invariant density matrix `P = sum_cluster (q_cluster / d) sum_{m in cluster} c_m c_m^T`
/// by clustering eigenvalues within `1e-5` Eh.
fn spherical_density_from_fock(
    fock: &Matrix2D,
    x: &Matrix2D,
    xt: &Matrix2D,
    n_elec: usize,
) -> Option<Matrix2D> {
    let m = fock.rows;
    if n_elec == 0 {
        return Some(Matrix2D::zeros(m, m));
    }
    let fp = matmul(&matmul(xt, fock), x);
    let (eps, cp) = eigh_sym(&fp, 100, 1e-14).ok()?;
    let c = matmul(x, &cp);

    let mut p = Matrix2D::zeros(m, m);
    let mut remaining = n_elec as f64;
    let mut start = 0usize;
    while start < m && remaining > 1e-12 {
        let e0 = eps[start];
        let mut end = start + 1;
        while end < m && (eps[end] - e0).abs() < 1e-5 {
            end += 1;
        }
        let deg = end - start;
        let take = remaining.min(deg as f64);
        let occ = take / (deg as f64);
        for mu in 0..m {
            for nu in 0..=mu {
                let mut sum = 0.0;
                for k in start..end {
                    sum += c.get(mu, k) * c.get(nu, k);
                }
                let val = occ * sum;
                p.data[mu * m + nu] += val;
                if mu != nu {
                    p.data[nu * m + mu] += val;
                }
            }
        }
        remaining -= take;
        start = end;
    }
    if remaining > 1e-8 {
        return None;
    }
    Some(p)
}

/// Solves an isolated neutral atom via spherically-averaged UHF in its exact molecular basis subset
/// (preserving 6D/10F and ECPs), returning the flattened `m_ao x m_ao` total density `P_alpha + P_beta`.
fn solve_atomic_sad_block(
    atom: &Atom,
    atom_shells: &[&Shell],
    atom_ecp: Option<&AtomEcp>,
    n_core: u8,
) -> Option<Vec<f64>> {
    let (n_a_atom, n_b_atom) = atomic_valence_spin_counts(atom.z, n_core)?;
    let z_eff = (n_a_atom + n_b_atom) as f64;
    if atom_shells.is_empty() {
        return None;
    }

    let mut local_shells = Vec::with_capacity(atom_shells.len());
    let mut m_ao = 0usize;
    for sh in atom_shells {
        let prims: Vec<Primitive> = sh.primitives.clone();
        local_shells.push(Shell {
            atom_index: 0,
            center: [0.0, 0.0, 0.0],
            shell_type: sh.shell_type,
            primitives: prims,
            ao_offset: m_ao,
        });
        m_ao += sh.n_functions();
    }
    if m_ao == 0 || m_ao < n_a_atom || m_ao < n_b_atom {
        return None;
    }

    let local_ecps = if let Some(ecp) = atom_ecp {
        let mut cloned = ecp.clone();
        cloned.atom_index = 0;
        cloned.center = [0.0, 0.0, 0.0];
        vec![cloned]
    } else {
        Vec::new()
    };

    let atom_mol = Molecule::new(
        vec![Atom::from_bohr(&atom.symbol, [0.0, 0.0, 0.0]).ok()?],
        0,
        1,
    );
    let atom_basis = BasisSet {
        name: "atomic_sad".to_string(),
        shells: local_shells,
        n_ao: m_ao,
        core_electrons: vec![n_core],
        ecp_potentials: local_ecps,
    };

    let s = compute_overlap(&atom_basis);
    if m_ao == 1 {
        let s00 = s.get(0, 0);
        if s00 <= 1e-12 || !s00.is_finite() {
            return None;
        }
        return Some(vec![z_eff / s00]);
    }

    let t = compute_kinetic(&atom_basis);
    let v = compute_nuclear(&atom_basis, &atom_mol);
    let v_ecp = compute_ecp(&atom_basis);
    let mut h_atom = Matrix2D::zeros(m_ao, m_ao);
    for i in 0..m_ao * m_ao {
        h_atom.data[i] = t.data[i] + v.data[i] + v_ecp.data[i];
    }

    let (s_vals, s_vecs) = eigh_sym(&s, 100, 1e-14).ok()?;
    let mut s_inv_sqrt = Matrix2D::zeros(m_ao, m_ao);
    for (i, &val) in s_vals.iter().enumerate() {
        if val <= 1e-10 || !val.is_finite() {
            return None;
        }
        s_inv_sqrt.set(i, i, 1.0 / val.sqrt());
    }
    let x = matmul(&matmul(&s_vecs, &s_inv_sqrt), &transpose(&s_vecs));
    let xt = transpose(&x);

    let mut pa = spherical_density_from_fock(&h_atom, &x, &xt, n_a_atom)?;
    let mut pb = spherical_density_from_fock(&h_atom, &x, &xt, n_b_atom)?;
    let eri = compute_sparse_eri(&atom_basis);

    let stack = |a: &Matrix2D, b: &Matrix2D| -> Matrix2D {
        let mut out = Matrix2D::zeros(m_ao + m_ao, m_ao);
        out.data[..m_ao * m_ao].copy_from_slice(&a.data);
        out.data[m_ao * m_ao..].copy_from_slice(&b.data);
        out
    };
    let unstack = |mat: &Matrix2D| -> (Matrix2D, Matrix2D) {
        let mut a = Matrix2D::zeros(m_ao, m_ao);
        let mut b = Matrix2D::zeros(m_ao, m_ao);
        a.data.copy_from_slice(&mat.data[..m_ao * m_ao]);
        b.data.copy_from_slice(&mat.data[m_ao * m_ao..]);
        (a, b)
    };
    let comm = |f: &Matrix2D, p: &Matrix2D| -> Matrix2D {
        let fps = matmul(&matmul(f, p), &s);
        let spf = matmul(&matmul(&s, p), f);
        let mut e = Matrix2D::zeros(m_ao, m_ao);
        for i in 0..m_ao * m_ao {
            e.data[i] = fps.data[i] - spf.data[i];
        }
        e
    };

    let mut diis = DiisState::new(6);
    let mut e_prev = 0.0f64;
    let mut converged = false;

    for iter in 1..=60usize {
        let mut pt_data = vec![0.0f64; m_ao * m_ao];
        for i in 0..m_ao * m_ao {
            pt_data[i] = pa.data[i] + pb.data[i];
        }
        let j = eri.contract_rks(&pt_data, 0.0);
        let ka = eri.contract_k_gen(&pa.data);
        let kb = if n_b_atom > 0 {
            eri.contract_k_gen(&pb.data)
        } else {
            vec![0.0f64; m_ao * m_ao]
        };

        let mut fa = Matrix2D::zeros(m_ao, m_ao);
        let mut fb = Matrix2D::zeros(m_ao, m_ao);
        let mut e_elec = 0.0f64;
        for i in 0..m_ao * m_ao {
            fa.data[i] = h_atom.data[i] + j[i] - ka[i];
            fb.data[i] = h_atom.data[i] + j[i] - kb[i];
            e_elec += pt_data[i] * h_atom.data[i] + pa.data[i] * fa.data[i] + pb.data[i] * fb.data[i];
        }
        e_elec *= 0.5;

        let err = stack(&comm(&fa, &pa), &comm(&fb, &pb));
        let err_norm = frobenius_norm(&err);
        let (fa_step, fb_step) = if iter >= 2 {
            let ex = diis.try_push_and_extrapolate(&stack(&fa, &fb), &err).ok()?;
            unstack(&ex)
        } else {
            (fa.clone(), fb.clone())
        };

        let mut pa_new = spherical_density_from_fock(&fa_step, &x, &xt, n_a_atom)?;
        let mut pb_new = spherical_density_from_fock(&fb_step, &x, &xt, n_b_atom)?;

        if iter > 10 || (iter > 2 && err_norm > 0.5) {
            let d = 0.2;
            for i in 0..m_ao * m_ao {
                pa_new.data[i] = (1.0 - d) * pa_new.data[i] + d * pa.data[i];
                pb_new.data[i] = (1.0 - d) * pb_new.data[i] + d * pb.data[i];
            }
        }

        let delta_e = if iter == 1 {
            f64::INFINITY
        } else {
            (e_elec - e_prev).abs()
        };
        let rms_d = rms_diff(&pa_new, &pa).max(rms_diff(&pb_new, &pb));
        pa = pa_new;
        pb = pb_new;
        e_prev = e_elec;

        if iter > 1 && delta_e < 1e-8 && rms_d < 1e-6 {
            converged = true;
            break;
        }
    }

    if !converged {
        return None;
    }

    let mut p_tot = vec![0.0f64; m_ao * m_ao];
    for i in 0..m_ao * m_ao {
        let v = pa.data[i] + pb.data[i];
        if !v.is_finite() {
            return None;
        }
        p_tot[i] = v;
    }
    Some(p_tot)
}

fn get_or_solve_atomic_sad_block(
    atom: &Atom,
    atom_shells: &[&Shell],
    atom_ecp: Option<&AtomEcp>,
    n_core: u8,
) -> Option<Arc<Vec<f64>>> {
    let key = make_atomic_sad_key(atom.z, n_core, atom_shells, atom_ecp);
    let cache = ATOMIC_SAD_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(hit) = cache.lock().unwrap().get(&key).cloned() {
        return Some(hit);
    }
    let solved = Arc::new(solve_atomic_sad_block(atom, atom_shells, atom_ecp, n_core)?);
    let mut map = cache.lock().unwrap();
    if map.len() >= MAX_ATOMIC_SAD_CACHE_ENTRIES {
        map.clear();
    }
    map.insert(key, solved.clone());
    Some(solved)
}

fn fill_diagonal_sad_fallback(
    p: &mut Matrix2D,
    atom_z: u8,
    n_core: u8,
    atom_shells: &[&Shell],
) {
    let (s_occ, p_occ, d_occ, f_occ) = atomic_subshell_occupations_valence(atom_z, n_core);
    let mut s_shells = Vec::new();
    let mut p_shells = Vec::new();
    let mut d_shells = Vec::new();
    let mut f_shells = Vec::new();
    for &shell in atom_shells {
        match shell.shell_type {
            ShellType::S => s_shells.push(shell),
            ShellType::P => p_shells.push(shell),
            ShellType::D => d_shells.push(shell),
            ShellType::F => f_shells.push(shell),
        }
    }

    let s_alloc = distribute_subshells(&s_occ, s_shells.len());
    let p_alloc = distribute_subshells(&p_occ, p_shells.len());
    let d_alloc = distribute_subshells(&d_occ, d_shells.len());
    let f_alloc = distribute_subshells(&f_occ, f_shells.len());

    for (shell, &n_e) in s_shells.iter().zip(&s_alloc) {
        let off = shell.ao_offset;
        p.set(off, off, n_e);
    }
    for (shell, &n_e) in p_shells.iter().zip(&p_alloc) {
        let off = shell.ao_offset;
        let per_ao = n_e / 3.0;
        for k in 0..3 {
            p.set(off + k, off + k, per_ao);
        }
    }
    for (shell, &n_e) in d_shells.iter().zip(&d_alloc) {
        let off = shell.ao_offset;
        let n_fn = shell.n_functions();
        let per_ao = n_e / (n_fn as f64);
        for k in 0..n_fn {
            p.set(off + k, off + k, per_ao);
        }
    }
    for (shell, &n_e) in f_shells.iter().zip(&f_alloc) {
        let off = shell.ao_offset;
        let n_fn = shell.n_functions();
        let per_ao = n_e / (n_fn as f64);
        for k in 0..n_fn {
            p.set(off + k, off + k, per_ao);
        }
    }
}

fn build_sad_total_density(
    mol: &Molecule,
    basis: &BasisSet,
    s_mat: &Matrix2D,
    target_electrons: f64,
) -> Result<Matrix2D, CalcError> {
    let n_ao = basis.n_ao;
    let mut p = Matrix2D::zeros(n_ao, n_ao);

    for (atom_idx, atom) in mol.atoms.iter().enumerate() {
        let atom_shells: Vec<&Shell> = basis
            .shells
            .iter()
            .filter(|sh| sh.atom_index == atom_idx)
            .collect();
        let n_core = basis.core_electrons.get(atom_idx).copied().unwrap_or(0);
        let atom_ecp = basis
            .ecp_potentials
            .iter()
            .find(|e| e.atom_index == atom_idx);

        if let Some(block) = get_or_solve_atomic_sad_block(atom, &atom_shells, atom_ecp, n_core) {
            let m_ao: usize = atom_shells.iter().map(|sh| sh.n_functions()).sum();
            let mut local_off1 = 0usize;
            for sh1 in &atom_shells {
                let n1 = sh1.n_functions();
                let mut local_off2 = 0usize;
                for sh2 in &atom_shells {
                    let n2 = sh2.n_functions();
                    for i1 in 0..n1 {
                        let mu = sh1.ao_offset + i1;
                        let r_loc = local_off1 + i1;
                        for i2 in 0..n2 {
                            let nu = sh2.ao_offset + i2;
                            let c_loc = local_off2 + i2;
                            p.set(mu, nu, block[r_loc * m_ao + c_loc]);
                        }
                    }
                    local_off2 += n2;
                }
                local_off1 += n1;
            }
        } else {
            fill_diagonal_sad_fallback(&mut p, atom.z, n_core, &atom_shells);
        }
    }

    let mut tr_ps = 0.0;
    for mu in 0..n_ao {
        for nu in 0..n_ao {
            tr_ps += p.get(mu, nu) * s_mat.get(nu, mu);
        }
    }

    if tr_ps > 1e-12 {
        let scale = target_electrons / tr_ps;
        for val in p.data.iter_mut() {
            *val *= scale;
        }
    } else {
        return Err(CalcError::LinearAlgebra(
            "SAD guess trace is degenerate or zero".into(),
        ));
    }

    Ok(p)
}

/// Generates the Superposition of Atomic Densities (SAD) initial density matrix $P^{(0)}$.
///
/// Features:
/// 1. Spherically averaged atomic-UHF density blocks solved in each atom's exact basis subset.
/// 2. Automatic scaling to conserve molecular total electron count: $\text{Tr}(P S) = 2 \times n_{\text{occ}}$.
/// 3. Zero-off-diagonal interatomic elements, ensuring clean Coulomb screening without false initial bonding.
pub fn sad_guess(
    mol: &Molecule,
    basis: &BasisSet,
    s_mat: &Matrix2D,
    n_occ: usize,
) -> Result<Matrix2D, CalcError> {
    if n_occ == 0 {
        return Err(CalcError::InvalidMolecule(
            "RHF/RKS SAD guess requires at least one occupied orbital".into(),
        ));
    }
    build_sad_total_density(mol, basis, s_mat, (2 * n_occ) as f64)
}

/// Generates open-shell SAD spin-density matrices `(P_alpha, P_beta)` scaled from the
/// spherically-averaged atomic-block total density `P_tot` so `Tr(P_alpha S) = n_alpha`,
/// `Tr(P_beta S) = n_beta`, and `P_alpha + P_beta = P_tot`.
pub fn sad_guess_uhf(
    mol: &Molecule,
    basis: &BasisSet,
    s_mat: &Matrix2D,
    n_alpha: usize,
    n_beta: usize,
) -> Result<(Matrix2D, Matrix2D), CalcError> {
    let n_tot = n_alpha + n_beta;
    if n_tot == 0 {
        return Err(CalcError::InvalidMolecule(
            "UHF/UKS SAD guess requires at least one electron".into(),
        ));
    }
    let p_tot = build_sad_total_density(mol, basis, s_mat, n_tot as f64)?;
    let n = basis.n_ao;
    let fa = (n_alpha as f64) / (n_tot as f64);
    let fb = (n_beta as f64) / (n_tot as f64);
    let mut pa = Matrix2D::zeros(n, n);
    let mut pb = Matrix2D::zeros(n, n);
    for idx in 0..n * n {
        let v = p_tot.data[idx];
        pa.data[idx] = fa * v;
        if n_beta > 0 {
            pb.data[idx] = fb * v;
        }
    }
    Ok((pa, pb))
}

#[cfg(test)]
mod tests {
    use super::*;
    use calchem_int::compute_overlap;

    fn trace_ps(p: &Matrix2D, s: &Matrix2D) -> f64 {
        let n = p.rows;
        let mut tr = 0.0;
        for mu in 0..n {
            for nu in 0..n {
                tr += p.get(mu, nu) * s.get(nu, mu);
            }
        }
        tr
    }

    #[test]
    fn test_sad_guess_trace_neutral_ions_ecp_and_uhf() {
        // 1. Neutral H2O (6-31G(d))
        let h2o = Molecule::new(
            vec![
                Atom::from_angstrom("O", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_angstrom("H", [0.0, 0.757, 0.586]).unwrap(),
                Atom::from_angstrom("H", [0.0, -0.757, 0.586]).unwrap(),
            ],
            0,
            1,
        );
        let bs_h2o = BasisSet::from_name("6-31g(d)", &h2o).unwrap();
        let s_h2o = compute_overlap(&bs_h2o);
        let p_h2o = sad_guess(&h2o, &bs_h2o, &s_h2o, 5).unwrap();
        assert!((trace_ps(&p_h2o, &s_h2o) - 10.0).abs() < 1e-10);
        // Verify intra-atomic off-diagonal 1s-2s block on O is non-zero (atomic UHF block)
        assert!(p_h2o.get(0, 1).abs() > 1e-3);

        // 2. Charged ions (OH- anion and H2O+ cation) + sad_guess_uhf
        let oh_minus = Molecule::new(
            vec![
                Atom::from_angstrom("O", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_angstrom("H", [0.0, 0.0, 0.97]).unwrap(),
            ],
            -1,
            1,
        );
        let bs_oh = BasisSet::from_name("6-31g(d)", &oh_minus).unwrap();
        let s_oh = compute_overlap(&bs_oh);
        let p_oh = sad_guess(&oh_minus, &bs_oh, &s_oh, 5).unwrap();
        assert!((trace_ps(&p_oh, &s_oh) - 10.0).abs() < 1e-10);

        let h2o_cat = Molecule::new(h2o.atoms.clone(), 1, 2);
        let bs_cat = BasisSet::from_name("6-31g(d)", &h2o_cat).unwrap();
        let s_cat = compute_overlap(&bs_cat);
        let (pa_cat, pb_cat) = sad_guess_uhf(&h2o_cat, &bs_cat, &s_cat, 5, 4).unwrap();
        assert!((trace_ps(&pa_cat, &s_cat) - 5.0).abs() < 1e-10);
        assert!((trace_ps(&pb_cat, &s_cat) - 4.0).abs() < 1e-10);

        // 3. ECP system: HBr in LANL2DZ
        let hbr = Molecule::new(
            vec![
                Atom::from_angstrom("Br", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_angstrom("H", [0.0, 0.0, 1.42]).unwrap(),
            ],
            0,
            1,
        );
        let bs_hbr = BasisSet::from_name("lanl2dz", &hbr).unwrap();
        let s_hbr = compute_overlap(&bs_hbr);
        let p_hbr = sad_guess(&hbr, &bs_hbr, &s_hbr, 4).unwrap();
        assert!((trace_ps(&p_hbr, &s_hbr) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_sad_guess_so3_rotational_invariance_6d() {
        // Test that rotating an isolated open-shell atom + d-shell (6D) produces identical eigenvalues
        // of S^{1/2} P S^{1/2} and identical Frobenius norm & rotated density for O + H2O in 6-31G(d).
        let rot = |v: [f64; 3]| -> [f64; 3] {
            // Arbitrary 3D orthogonal rotation (Euler angles)
            let (a, b) = (0.73f64, 0.41f64);
            let (sa, ca) = a.sin_cos();
            let (sb, cb) = b.sin_cos();
            let x1 = ca * v[0] - sa * v[1];
            let y1 = sa * v[0] + ca * v[1];
            let z1 = v[2];
            [cb * x1 + sb * z1, y1, -sb * x1 + cb * z1]
        };

        let atoms = vec![
            Atom::from_angstrom("C", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_angstrom("O", [1.21, 0.0, 0.0]).unwrap(),
            Atom::from_angstrom("H", [-0.55, 0.94, 0.0]).unwrap(),
            Atom::from_angstrom("H", [-0.55, -0.94, 0.0]).unwrap(),
        ];
        let atoms_rot: Vec<Atom> = atoms
            .iter()
            .map(|a| {
                let ang = [
                    a.position_bohr[0] * calchem_model::BOHR_TO_ANGSTROM,
                    a.position_bohr[1] * calchem_model::BOHR_TO_ANGSTROM,
                    a.position_bohr[2] * calchem_model::BOHR_TO_ANGSTROM,
                ];
                Atom::from_angstrom(&a.symbol, rot(ang)).unwrap()
            })
            .collect();

        let mol1 = Molecule::new(atoms, 0, 1);
        let mol2 = Molecule::new(atoms_rot, 0, 1);
        let bs1 = BasisSet::from_name("6-31g(d)", &mol1).unwrap();
        let bs2 = BasisSet::from_name("6-31g(d)", &mol2).unwrap();
        let s1 = compute_overlap(&bs1);
        let s2 = compute_overlap(&bs2);
        let p1 = sad_guess(&mol1, &bs1, &s1, 8).unwrap();
        let p2 = sad_guess(&mol2, &bs2, &s2, 8).unwrap();

        // Check rotational invariance of the density operator eigenvalues (P * S) via Tr((P S)^2)
        // and Tr(P S) == 16.0, which are basis-rotation invariants.
        let ps1 = matmul(&p1, &s1);
        let ps2 = matmul(&p2, &s2);
        let ps1_sq = matmul(&ps1, &ps1);
        let ps2_sq = matmul(&ps2, &ps2);
        let mut tr1 = 0.0;
        let mut tr2 = 0.0;
        let mut tr_sq1 = 0.0;
        let mut tr_sq2 = 0.0;
        for i in 0..bs1.n_ao {
            tr1 += ps1.get(i, i);
            tr2 += ps2.get(i, i);
            tr_sq1 += ps1_sq.get(i, i);
            tr_sq2 += ps2_sq.get(i, i);
        }
        assert!((tr1 - 16.0).abs() < 1e-10);
        assert!((tr2 - 16.0).abs() < 1e-10);
        assert!(
            (tr_sq1 - tr_sq2).abs() < 1e-10,
            "SO(3) invariance failed: Tr((PS)^2) {tr_sq1} vs {tr_sq2}"
        );

        // Also check Frobenius norm of S^{1/2} P S^{1/2} (eigenvalue spectrum invariant)
        let sym_spectrum = |p: &Matrix2D, s: &Matrix2D| -> Vec<f64> {
            let n = s.rows;
            let (sv, su) = eigh_sym(s, 100, 1e-14).unwrap();
            let mut shalf = Matrix2D::zeros(n, n);
            for (i, &v) in sv.iter().enumerate() {
                shalf.set(i, i, v.sqrt());
            }
            let s_sqrt = matmul(&matmul(&su, &shalf), &transpose(&su));
            let op = matmul(&matmul(&s_sqrt, p), &s_sqrt);
            eigh_sym(&op, 100, 1e-14).unwrap().0
        };
        let ev1 = sym_spectrum(&p1, &s1);
        let ev2 = sym_spectrum(&p2, &s2);
        let max_ev_diff = ev1
            .iter()
            .zip(&ev2)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_ev_diff < 1e-10,
            "S^{{1/2}} P S^{{1/2}} spectrum differed under 3D rotation by {max_ev_diff:.3e}"
        );
    }
}
