//! Multicenter molecular DFT quadrature grid:
//! - Murray-Handy-Laming Euler-Maclaurin radial quadrature ($q = 2$)
//! - Radial-shell pruned Lebedev-Laikov spherical quadrature
//! - Becke (1988) $k = 3$ fuzzy-Voronoi atomic partitioning with Bragg-Slater heteronuclear adjustment
//! - Exact analytical nuclear derivatives $\frac{dw_g}{d\mathbf{R}_A}$ of Becke quadrature weights

use calchem_model::{ANGSTROM_TO_BOHR, Molecule};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::lebedev::LebedevDegree;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GridLevel {
    Coarse,
    #[default]
    Medium,
    Fine,
}

impl GridLevel {
    pub fn from_name(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "coarse" | "small" | "1" => Self::Coarse,
            "fine" | "ultrafine" | "3" | "4" => Self::Fine,
            _ => Self::Medium,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GridPoint {
    pub coords: [f64; 3],
    /// Total quadrature weight `raw_weight * becke_weight`.
    pub weight: f64,
    /// Single-center radial * spherical weight (`w_rad * w_ang`) before Becke partitioning.
    pub raw_weight: f64,
    /// Index of the parent nucleus $H(g)$ with which `coords = R_H + u_g` co-moves.
    pub host_atom: usize,
}

#[derive(Debug, Clone)]
pub struct MolecularGrid {
    pub points: Vec<GridPoint>,
    pub level: GridLevel,
}

/// Bragg-Slater empirical atomic radii in Angstrom for `Z = 1..=86`
/// (Becke, *J. Chem. Phys.* **88**, 2547, 1988, Appendix; `H` = 0.35 Å).
pub fn bragg_slater_radius_bohr(z: u8) -> f64 {
    let ang = match z {
        1 => 0.35,
        2 => 0.35,
        3 => 1.45,
        4 => 1.05,
        5 => 0.85,
        6 => 0.70,
        7 => 0.65,
        8 => 0.60,
        9 => 0.50,
        10 => 0.45,
        11 => 1.80,
        12 => 1.50,
        13 => 1.25,
        14 => 1.10,
        15 => 1.00,
        16 => 1.00,
        17 => 1.00,
        18 => 1.00,
        19 => 2.20,
        20 => 1.80,
        21..=30 => 1.40,
        31 => 1.30,
        32 => 1.25,
        33 => 1.15,
        34 => 1.15,
        35 => 1.15, // Br
        36 => 1.10,
        37 => 2.35,
        38 => 2.00,
        39..=48 => 1.55,
        49 => 1.55,
        50 => 1.45,
        51 => 1.45,
        52 => 1.40,
        53 => 1.40, // I
        54 => 1.35,
        55 => 2.60,
        56 => 2.15,
        _ => 1.75,
    };
    ang * ANGSTROM_TO_BOHR
}

/// Becke heteronuclear boundary shift parameter $a_{CD} \in [-0.5, 0.5]$.
#[inline]
pub fn becke_asymmetry_param(z_c: u8, z_d: u8) -> f64 {
    if z_c == z_d {
        return 0.0;
    }
    let rc = bragg_slater_radius_bohr(z_c);
    let rd = bragg_slater_radius_bohr(z_d);
    let chi = rc / rd;
    let u = (chi - 1.0) / (chi + 1.0);
    let a = u / (u * u - 1.0);
    a.clamp(-0.5, 0.5)
}

#[inline]
fn becke_poly1(x: f64) -> f64 {
    0.5 * x * (3.0 - x * x)
}

#[inline]
fn becke_poly1_deriv(x: f64) -> f64 {
    1.5 * (1.0 - x * x)
}

/// Evaluates Becke $k = 3$ step function $s(\nu) = \frac{1}{2}(1 - p_3(\nu))$ and its derivative $s'(\nu)$.
#[inline]
fn becke_step_and_deriv(nu: f64) -> (f64, f64) {
    let x = nu.clamp(-1.0, 1.0);
    let f1 = becke_poly1(x);
    let d1 = becke_poly1_deriv(x);
    let f2 = becke_poly1(f1);
    let d2 = becke_poly1_deriv(f1);
    let f3 = becke_poly1(f2);
    let d3 = becke_poly1_deriv(f2);
    let s = 0.5 * (1.0 - f3);
    let ds_dnu = -0.5 * d3 * d2 * d1;
    (s, ds_dnu)
}

fn radial_shell_count(z: u8, level: GridLevel) -> usize {
    match level {
        GridLevel::Coarse => {
            if z <= 2 {
                25
            } else if z <= 18 {
                35
            } else {
                45
            }
        }
        GridLevel::Medium => {
            if z <= 2 {
                35
            } else if z <= 18 {
                50
            } else {
                65
            }
        }
        GridLevel::Fine => {
            if z <= 2 {
                55
            } else if z <= 18 {
                75
            } else {
                90
            }
        }
    }
}

/// SG-1 / Treutler-style radial-region angular pruning based on `r / R_BS`.
fn pruned_angular_degree(r_bohr: f64, r_bs: f64, level: GridLevel) -> LebedevDegree {
    let ratio = r_bohr / r_bs.max(0.3);
    match level {
        GridLevel::Coarse => {
            if ratio < 0.25 {
                LebedevDegree::N26
            } else if ratio < 0.75 {
                LebedevDegree::N50
            } else if ratio < 4.5 {
                LebedevDegree::N110
            } else {
                LebedevDegree::N50
            }
        }
        GridLevel::Medium => {
            if ratio < 0.20 {
                LebedevDegree::N26
            } else if ratio < 0.55 {
                LebedevDegree::N50
            } else if ratio < 1.10 {
                LebedevDegree::N110
            } else if ratio < 4.50 {
                LebedevDegree::N194
            } else {
                LebedevDegree::N110
            }
        }
        GridLevel::Fine => {
            if ratio < 0.18 {
                LebedevDegree::N50
            } else if ratio < 0.50 {
                LebedevDegree::N110
            } else if ratio < 0.95 {
                LebedevDegree::N194
            } else if ratio < 5.00 {
                LebedevDegree::N302
            } else {
                LebedevDegree::N194
            }
        }
    }
}

/// Computes normalized Becke cell weight $B_H(\mathbf{r})$ for host atom `host` at point `r`.
pub fn becke_partition_weight(mol: &Molecule, host: usize, r: [f64; 3]) -> f64 {
    let n_atoms = mol.atoms.len();
    if n_atoms <= 1 {
        return 1.0;
    }

    let mut dist_r = vec![0.0f64; n_atoms];
    for (c, atom) in mol.atoms.iter().enumerate() {
        let dx = r[0] - atom.position_bohr[0];
        let dy = r[1] - atom.position_bohr[1];
        let dz = r[2] - atom.position_bohr[2];
        dist_r[c] = (dx * dx + dy * dy + dz * dz).sqrt();
    }

    let mut cell_p = vec![1.0f64; n_atoms];
    for c in 0..n_atoms {
        for d in (c + 1)..n_atoms {
            let ac = mol.atoms[c].position_bohr;
            let ad = mol.atoms[d].position_bohr;
            let r_cd =
                ((ac[0] - ad[0]).powi(2) + (ac[1] - ad[1]).powi(2) + (ac[2] - ad[2]).powi(2))
                    .sqrt();
            if r_cd < 1e-14 {
                continue;
            }
            let mu = ((dist_r[c] - dist_r[d]) / r_cd).clamp(-1.0, 1.0);
            let a_cd = becke_asymmetry_param(mol.atoms[c].z, mol.atoms[d].z);
            let nu_cd = mu + a_cd * (1.0 - mu * mu);
            let (s_cd, _) = becke_step_and_deriv(nu_cd);
            // Note: nu_dc = -mu + (-a_cd)*(1 - mu^2) = -nu_cd => s(nu_dc) = 1 - s(nu_cd)!
            let s_dc = 1.0 - s_cd;
            cell_p[c] *= s_cd;
            cell_p[d] *= s_dc;
        }
    }

    let sum_p: f64 = cell_p.iter().sum();
    if sum_p <= 1e-30 {
        0.0
    } else {
        cell_p[host] / sum_p
    }
}

/// Computes the exact total nuclear gradient $\frac{d w_g}{d \mathbf{R}_A}$ (`A = 0..n_atoms`)
/// of a grid point $\mathbf{r}_g = \mathbf{R}_H + \mathbf{u}_g$ co-moving with `host` ($H$).
pub fn becke_weight_nuclear_gradient(
    mol: &Molecule,
    host: usize,
    r: [f64; 3],
    raw_weight: f64,
    out_grad: &mut [[f64; 3]],
    f_xc_val: f64,
) {
    let n_atoms = mol.atoms.len();
    if n_atoms <= 1 || f_xc_val.abs() < 1e-18 || raw_weight == 0.0 {
        return;
    }

    // Small stack/vec buffers? Note n_atoms is typically <= 200; for hot loops we use small Vecs
    // or caller scratch, wait: let's compute cleanly with zero quadratic blowup!
    let mut vec_ra = vec![[0.0f64; 3]; n_atoms];
    let mut dist_r = vec![0.0f64; n_atoms];
    for (c, atom) in mol.atoms.iter().enumerate() {
        let dx = r[0] - atom.position_bohr[0];
        let dy = r[1] - atom.position_bohr[1];
        let dz = r[2] - atom.position_bohr[2];
        vec_ra[c] = [dx, dy, dz];
        dist_r[c] = (dx * dx + dy * dy + dz * dz).sqrt();
    }

    // Store s_mat[c * n + d] and ds_dmu[c * n + d] = ds(nu_cd)/d(mu_cd).
    let mut s_mat = vec![1.0f64; n_atoms * n_atoms];
    let mut ds_dmu = vec![0.0f64; n_atoms * n_atoms];
    let mut cell_p = vec![1.0f64; n_atoms];

    for c in 0..n_atoms {
        for d in (c + 1)..n_atoms {
            let ac = mol.atoms[c].position_bohr;
            let ad = mol.atoms[d].position_bohr;
            let r_cd =
                ((ac[0] - ad[0]).powi(2) + (ac[1] - ad[1]).powi(2) + (ac[2] - ad[2]).powi(2))
                    .sqrt();
            if r_cd < 1e-14 {
                continue;
            }
            let mu = ((dist_r[c] - dist_r[d]) / r_cd).clamp(-1.0, 1.0);
            let a_cd = becke_asymmetry_param(mol.atoms[c].z, mol.atoms[d].z);
            let nu_cd = mu + a_cd * (1.0 - mu * mu);
            let (s_cd, ds_dnu) = becke_step_and_deriv(nu_cd);
            let s_dc = 1.0 - s_cd;
            // d(nu_cd)/d(mu_cd) = 1 - 2 * a_cd * mu
            let deriv_mu_cd = ds_dnu * (1.0 - 2.0 * a_cd * mu);

            s_mat[c * n_atoms + d] = s_cd;
            s_mat[d * n_atoms + c] = s_dc;
            // Note: mu_dc = -mu_cd and s_dc = 1 - s_cd => d(s_dc)/d(mu_dc) = d(1 - s_cd)/d(-mu_cd) = +deriv_mu_cd!
            ds_dmu[c * n_atoms + d] = deriv_mu_cd;
            ds_dmu[d * n_atoms + c] = deriv_mu_cd;

            cell_p[c] *= s_cd;
            cell_p[d] *= s_dc;
        }
    }

    let z_sum: f64 = cell_p.iter().sum();
    if z_sum <= 1e-30 {
        return;
    }
    let w_norm = cell_p[host] / z_sum;
    let pref = raw_weight * f_xc_val / z_sum;

    // For each atom A != host, hold `r = R_host + u` FIXED while moving `R_A`:
    // Only pairs involving A (`(A, B)` and `(B, A)` for `B != A`) depend on `R_A`!
    // Specifically, for B != A:
    //   mu_{AB} = (r_A - r_B) / R_{AB}
    //   grad_{R_A} mu_{AB} = [ -(r - R_A)/r_A - mu_{AB} * (R_A - R_B)/R_{AB} ] / R_{AB}
    // and mu_{BA} = -mu_{AB}, so grad_{R_A} mu_{BA} = -grad_{R_A} mu_{AB}!
    let mut host_accum = [0.0f64; 3];

    for a in 0..n_atoms {
        if a == host {
            continue;
        }
        let ra = dist_r[a];
        if ra < 1e-14 {
            continue;
        }
        let pos_a = mol.atoms[a].position_bohr;
        let dr_a = [
            -vec_ra[a][0] / ra,
            -vec_ra[a][1] / ra,
            -vec_ra[a][2] / ra,
        ];

        // Compute dP_A / dR_A = sum_{B != A} (P_A / s_{AB}) * ds_{AB}/dR_A
        // and for each B != A: dP_B / dR_A = (P_B / s_{BA}) * ds_{BA}/dR_A
        // Notice we can compute exact product-without-one `prod_{D != C, k} s_{CD}` safely even if `s_{Ck} == 0`!
        let prod_except = |c: usize, skip: usize| -> f64 {
            let s_val = s_mat[c * n_atoms + skip];
            if s_val > 1e-12 {
                cell_p[c] / s_val
            } else {
                let mut p = 1.0;
                for d in 0..n_atoms {
                    if d != c && d != skip {
                        p *= s_mat[c * n_atoms + d];
                    }
                }
                p
            }
        };

        let mut dp_host = [0.0f64; 3];
        let mut dz_sum = [0.0f64; 3];

        for b in 0..n_atoms {
            if b == a {
                continue;
            }
            let pos_b = mol.atoms[b].position_bohr;
            let rab_vec = [pos_a[0] - pos_b[0], pos_a[1] - pos_b[1], pos_a[2] - pos_b[2]];
            let rab =
                (rab_vec[0] * rab_vec[0] + rab_vec[1] * rab_vec[1] + rab_vec[2] * rab_vec[2])
                    .sqrt();
            if rab < 1e-14 {
                continue;
            }
            let mu_ab = ((ra - dist_r[b]) / rab).clamp(-1.0, 1.0);
            let inv_rab = 1.0 / rab;
            let dmu_ab = [
                (dr_a[0] - mu_ab * rab_vec[0] * inv_rab) * inv_rab,
                (dr_a[1] - mu_ab * rab_vec[1] * inv_rab) * inv_rab,
                (dr_a[2] - mu_ab * rab_vec[2] * inv_rab) * inv_rab,
            ];

            // ds_{AB} / dR_A = ds_dmu[a, b] * dmu_ab
            // ds_{BA} / dR_A = -ds_{AB} / dR_A  (since s_{BA} = 1 - s_{AB}!)
            let slope = ds_dmu[a * n_atoms + b];
            if slope.abs() < 1e-30 {
                continue;
            }
            let p_a_no_b = prod_except(a, b);
            let p_b_no_a = prod_except(b, a);

            for q in 0..3 {
                let ds_ab = slope * dmu_ab[q];
                let dpa_term = p_a_no_b * ds_ab;
                let dpb_term = -p_b_no_a * ds_ab;
                dz_sum[q] += dpa_term + dpb_term;
                if b == host {
                    dp_host[q] += dpb_term;
                }
            }
        }

        // Note: since `a != host`, `P_host` only depends on `R_A` through `b == host` above!
        // d(P_host / Z) / dR_A = (dp_host - w_norm * dz_sum) / Z
        for q in 0..3 {
            let dw_f = pref * (dp_host[q] - w_norm * dz_sum[q]);
            out_grad[a][q] += dw_f;
            host_accum[q] -= dw_f;
        }
    }

    for q in 0..3 {
        out_grad[host][q] += host_accum[q];
    }
}

impl MolecularGrid {
    pub fn build(mol: &Molecule, level: GridLevel) -> Self {
        Self::build_with_radial_min(mol, level, 0)
    }

    /// Resolve diffuse Gaussian tails from the basis exponents, not element radii.
    pub fn build_for_basis(mol: &Molecule, basis: &calchem_basis::BasisSet, level: GridLevel) -> Self {
        let alpha = basis.shells.iter().flat_map(|s| &s.primitives)
            .map(|p| p.exponent).fold(f64::INFINITY, f64::min);
        let extra = if alpha < 1e-3 {
            (180.0 * (1e-4 / alpha.max(1e-12)).powf(0.25)).ceil() as usize
        } else { 0 };
        Self::build_with_radial_min(mol, level, extra.min(4096))
    }

    fn build_with_radial_min(mol: &Molecule, level: GridLevel, radial_min: usize) -> Self {
        let per_atom: Vec<Vec<GridPoint>> = (0..mol.atoms.len())
            .into_par_iter()
            .map(|atom_idx| {
                let atom = &mol.atoms[atom_idx];
                let r_bs = bragg_slater_radius_bohr(atom.z);
                let base_min = if matches!(level, GridLevel::Fine) { 150 } else { 0 };
                let n_rad = radial_shell_count(atom.z, level).max(radial_min).max(base_min);
                let m = (n_rad + 1) as f64;
                let mut local_pts = Vec::with_capacity(n_rad * 150);

                for i in 1..=n_rad {
                    let t = i as f64;
                    let mt = m - t;
                    // Murray-Handy-Laming q=2 Euler-Maclaurin mapping:
                    let r = r_bs * (t * t) / (mt * mt);
                    let w_rad = 2.0 * r_bs.powi(3) * m * t.powi(5) / mt.powi(7);
                    let ang = pruned_angular_degree(r, r_bs, level).points();

                    for &(u, w_ang) in ang {
                        let raw_w = w_rad * w_ang;
                        let coords = [
                            atom.position_bohr[0] + r * u[0],
                            atom.position_bohr[1] + r * u[1],
                            atom.position_bohr[2] + r * u[2],
                        ];
                        let bw = becke_partition_weight(mol, atom_idx, coords);
                        let w_tot = raw_w * bw;
                        if w_tot > 1e-18 {
                            local_pts.push(GridPoint {
                                coords,
                                weight: w_tot,
                                raw_weight: raw_w,
                                host_atom: atom_idx,
                            });
                        }
                    }
                }
                local_pts
            })
            .collect();

        let total_len: usize = per_atom.iter().map(|v| v.len()).sum();
        let mut points = Vec::with_capacity(total_len);
        for mut sub in per_atom {
            points.append(&mut sub);
        }

        Self { points, level }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use calchem_model::Atom;
    use std::f64::consts::PI;

    #[test]
    fn test_single_center_gaussian_3d_integral() {
        let mol = Molecule::new(vec![Atom::from_bohr("C", [0.0, 0.0, 0.0]).unwrap()], 0, 1);
        let grid = MolecularGrid::build(&mol, GridLevel::Medium);
        for alpha in [0.15_f64, 1.0, 8.5] {
            let exact = (PI / alpha).powf(1.5);
            let num: f64 = grid
                .points
                .iter()
                .map(|p| {
                    let r2 = p.coords[0].powi(2) + p.coords[1].powi(2) + p.coords[2].powi(2);
                    p.weight * (-alpha * r2).exp()
                })
                .sum();
            let rel_err = (num - exact).abs() / exact;
            assert!(
                rel_err < 1e-6,
                "alpha={alpha}: num={num}, exact={exact}, rel_err={rel_err:.3e}"
            );
        }
    }

    #[test]
    fn test_multicenter_becke_gaussian_sum_and_weight_gradient_fd() {
        let mol = Molecule::new(
            vec![
                Atom::from_angstrom("O", [0.02, -0.04, 0.12]).unwrap(),
                Atom::from_angstrom("H", [0.08, 0.78, -0.45]).unwrap(),
                Atom::from_angstrom("H", [-0.05, -0.74, -0.49]).unwrap(),
            ],
            0,
            1,
        );
        let grid = MolecularGrid::build(&mol, GridLevel::Medium);

        // Sum of Gaussians centered on all 3 atoms integrates to 3 * pi^{3/2}
        let exact = 3.0 * PI.powf(1.5);
        let num: f64 = grid
            .points
            .iter()
            .map(|gp| {
                let mut val = 0.0;
                for a in &mol.atoms {
                    let d2 = (gp.coords[0] - a.position_bohr[0]).powi(2)
                        + (gp.coords[1] - a.position_bohr[1]).powi(2)
                        + (gp.coords[2] - a.position_bohr[2]).powi(2);
                    val += (-d2).exp();
                }
                gp.weight * val
            })
            .sum();
        assert!((num - exact).abs() / exact < 2e-5);

        // Exact analytical Becke weight nuclear gradient vs central finite differences:
        // Pick a non-trivial valence grid point belonging to host atom 0 (O) and also host atom 1 (H).
        for test_host in [0usize, 1usize] {
            let u = [0.42_f64, -0.31, 0.57];
            let raw_w = 0.73;
            let f_val = -1.25;
            let r0 = [
                mol.atoms[test_host].position_bohr[0] + u[0],
                mol.atoms[test_host].position_bohr[1] + u[1],
                mol.atoms[test_host].position_bohr[2] + u[2],
            ];
            let mut ana = vec![[0.0; 3]; mol.atoms.len()];
            becke_weight_nuclear_gradient(&mol, test_host, r0, raw_w, &mut ana, f_val);

            let h = 1e-4;
            let eval_shifted = |atom_idx: usize, axis: usize, delta: f64| -> f64 {
                let mut m_s = mol.clone();
                m_s.atoms[atom_idx].position_bohr[axis] += delta;
                let r_s = [
                    m_s.atoms[test_host].position_bohr[0] + u[0],
                    m_s.atoms[test_host].position_bohr[1] + u[1],
                    m_s.atoms[test_host].position_bohr[2] + u[2],
                ];
                raw_w * becke_partition_weight(&m_s, test_host, r_s) * f_val
            };
            for a in 0..mol.atoms.len() {
                for q in 0..3 {
                    let wp2 = eval_shifted(a, q, 2.0 * h);
                    let wp1 = eval_shifted(a, q, h);
                    let wm1 = eval_shifted(a, q, -h);
                    let wm2 = eval_shifted(a, q, -2.0 * h);
                    let fd = (-wp2 + 8.0 * wp1 - 8.0 * wm1 + wm2) / (12.0 * h);
                    assert!(
                        (ana[a][q] - fd).abs() < 1e-11,
                        "host={test_host} atom={a} q={q}: ana={} vs fd={}",
                        ana[a][q],
                        fd
                    );
                }
            }
        }
    }
}
