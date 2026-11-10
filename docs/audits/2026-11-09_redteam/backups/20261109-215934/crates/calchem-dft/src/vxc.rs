//! Parallel Kohn-Sham Exchange-Correlation Matrix Assembly ($E_{\text{xc}}, V^{\text{xc}}$)
//! and Exact Analytical XC Nuclear Gradients $\left.\frac{\partial E_{\text{xc}}}{\partial \mathbf{R}_A}\right|_{P^\alpha, P^\beta}$.
//!
//! Includes all three exact Pople-Johnson-Gill (1992/1993) DFT nuclear gradient contributions:
//! 1. Orbital center derivative via AO gradients $\nabla\phi_\mu$ and Hessians $\nabla\nabla\phi_\mu$.
//! 2. Host-atom quadrature point co-moving translation invariance ($\mathbf{r}_g = \mathbf{R}_{H(g)} + \mathbf{u}_g$).
//! 3. Analytical Becke fuzzy-Voronoi quadrature weight derivatives $\frac{d w_g}{d \mathbf{R}_A}$.

use calchem_basis::BasisSet;
use calchem_model::Molecule;
use calchem_tensor::Matrix2D;
use rayon::prelude::*;

use crate::ao_grid::{AoGradVal, AoHessVal, PrecomputedAoShell, eval_aos_grad, eval_aos_hess};
use crate::grid::{MolecularGrid, becke_weight_nuclear_gradient};
use crate::xc::{XcFunctional, eval_xc_spin};

const GRID_CHUNK_SIZE: usize = 256;
const DENSITY_SCREEN: f64 = 1e-18;

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Evaluates closed-shell Kohn-Sham $(E_{\text{xc}}, V^{\text{xc}})$ from spin-summed total density
/// $P^t = 2 P^\alpha$ (`n_ao x n_ao`).
pub fn evaluate_vxc_rks(
    basis: &BasisSet,
    grid: &MolecularGrid,
    func: XcFunctional,
    density_total: &Matrix2D,
) -> (f64, Matrix2D) {
    let n = basis.n_ao;
    if n == 0 || grid.points.is_empty() {
        return (0.0, Matrix2D::zeros(n, n));
    }

    let shells = PrecomputedAoShell::from_basis(basis);
    let p_slice = &density_total.data;

    let (e_xc, v_upper) = grid
        .points
        .par_chunks(GRID_CHUNK_SIZE)
        .map(|chunk| {
            let mut e_loc = 0.0f64;
            let mut v_loc = vec![0.0f64; n * n];
            let mut active: Vec<AoGradVal> = Vec::with_capacity(n);
            let mut phi_p: Vec<f64> = Vec::with_capacity(n);
            let mut z_vec: Vec<f64> = Vec::with_capacity(n);

            for gp in chunk {
                let w = gp.weight;
                if w.abs() < 1e-24 {
                    continue;
                }
                eval_aos_grad(&shells, gp.coords, &mut active);
                if active.is_empty() {
                    continue;
                }

                phi_p.clear();
                let mut rho_t = 0.0;
                let mut grad_rho_t = [0.0f64; 3];

                for a_i in &active {
                    let row = &p_slice[a_i.mu * n..(a_i.mu + 1) * n];
                    let mut s = 0.0;
                    for a_j in &active {
                        s += row[a_j.mu] * a_j.val;
                    }
                    phi_p.push(s);
                    rho_t += a_i.val * s;
                    grad_rho_t[0] += 2.0 * a_i.grad[0] * s;
                    grad_rho_t[1] += 2.0 * a_i.grad[1] * s;
                    grad_rho_t[2] += 2.0 * a_i.grad[2] * s;
                }

                if rho_t < DENSITY_SCREEN {
                    continue;
                }

                let rho_s = 0.5 * rho_t;
                let g_ss = 0.25 * dot3(grad_rho_t, grad_rho_t);
                let out = eval_xc_spin(func, rho_s, rho_s, g_ss, g_ss, g_ss);

                e_loc += w * out.f_xc;

                // Closed-shell spin symmetry: U_alpha = (v_gamma_aa + 0.5 * v_gamma_ab) * grad_rho_t
                let u_scale = out.v_gamma_aa + 0.5 * out.v_gamma_ab;
                let u_vec = [
                    u_scale * grad_rho_t[0],
                    u_scale * grad_rho_t[1],
                    u_scale * grad_rho_t[2],
                ];

                z_vec.clear();
                let half_wv = 0.5 * w * out.v_rho_a;
                for a_i in &active {
                    let z_i = half_wv * a_i.val + w * dot3(u_vec, a_i.grad);
                    z_vec.push(z_i);
                }

                let m = active.len();
                for i in 0..m {
                    let mu = active[i].mu;
                    let phi_i = active[i].val;
                    let z_i = z_vec[i];
                    let row_out = &mut v_loc[mu * n..(mu + 1) * n];
                    for j in 0..m {
                        let nu = active[j].mu;
                        row_out[nu] += z_i * active[j].val + phi_i * z_vec[j];
                    }
                }
            }

            (e_loc, v_loc)
        })
        .reduce(
            || (0.0f64, vec![0.0f64; n * n]),
            |(ea, mut va), (eb, vb)| {
                for (dst, src) in va.iter_mut().zip(vb.iter()) {
                    *dst += *src;
                }
                (ea + eb, va)
            },
        );

    let mut v_mat = Matrix2D::zeros(n, n);
    v_mat.data = v_upper;
    (e_xc, v_mat)
}

/// Evaluates spin-polarized open-shell Kohn-Sham $(E_{\text{xc}}, V^{\text{xc},\alpha}, V^{\text{xc},\beta})$
/// from per-spin densities $P^\alpha, P^\beta$ (`n_ao x n_ao`).
pub fn evaluate_vxc_uks(
    basis: &BasisSet,
    grid: &MolecularGrid,
    func: XcFunctional,
    pa: &Matrix2D,
    pb: &Matrix2D,
) -> (f64, Matrix2D, Matrix2D) {
    let n = basis.n_ao;
    if n == 0 || grid.points.is_empty() {
        return (0.0, Matrix2D::zeros(n, n), Matrix2D::zeros(n, n));
    }

    let shells = PrecomputedAoShell::from_basis(basis);
    let pa_slice = &pa.data;
    let pb_slice = &pb.data;

    let (e_xc, va_data, vb_data) = grid
        .points
        .par_chunks(GRID_CHUNK_SIZE)
        .map(|chunk| {
            let mut e_loc = 0.0f64;
            let mut va_loc = vec![0.0f64; n * n];
            let mut vb_loc = vec![0.0f64; n * n];
            let mut active: Vec<AoGradVal> = Vec::with_capacity(n);
            let mut za_vec: Vec<f64> = Vec::with_capacity(n);
            let mut zb_vec: Vec<f64> = Vec::with_capacity(n);

            for gp in chunk {
                let w = gp.weight;
                if w.abs() < 1e-24 {
                    continue;
                }
                eval_aos_grad(&shells, gp.coords, &mut active);
                if active.is_empty() {
                    continue;
                }

                let mut rho_a = 0.0;
                let mut rho_b = 0.0;
                let mut grad_a = [0.0f64; 3];
                let mut grad_b = [0.0f64; 3];

                for a_i in &active {
                    let row_a = &pa_slice[a_i.mu * n..(a_i.mu + 1) * n];
                    let row_b = &pb_slice[a_i.mu * n..(a_i.mu + 1) * n];
                    let mut sa = 0.0;
                    let mut sb = 0.0;
                    for a_j in &active {
                        sa += row_a[a_j.mu] * a_j.val;
                        sb += row_b[a_j.mu] * a_j.val;
                    }
                    rho_a += a_i.val * sa;
                    rho_b += a_i.val * sb;
                    for q in 0..3 {
                        grad_a[q] += 2.0 * a_i.grad[q] * sa;
                        grad_b[q] += 2.0 * a_i.grad[q] * sb;
                    }
                }

                let ra = rho_a.max(0.0);
                let rb = rho_b.max(0.0);
                if ra + rb < DENSITY_SCREEN {
                    continue;
                }

                let gaa = dot3(grad_a, grad_a);
                let gab = dot3(grad_a, grad_b);
                let gbb = dot3(grad_b, grad_b);
                let out = eval_xc_spin(func, ra, rb, gaa, gab, gbb);

                e_loc += w * out.f_xc;

                let ua = [
                    2.0 * out.v_gamma_aa * grad_a[0] + out.v_gamma_ab * grad_b[0],
                    2.0 * out.v_gamma_aa * grad_a[1] + out.v_gamma_ab * grad_b[1],
                    2.0 * out.v_gamma_aa * grad_a[2] + out.v_gamma_ab * grad_b[2],
                ];
                let ub = [
                    2.0 * out.v_gamma_bb * grad_b[0] + out.v_gamma_ab * grad_a[0],
                    2.0 * out.v_gamma_bb * grad_b[1] + out.v_gamma_ab * grad_a[1],
                    2.0 * out.v_gamma_bb * grad_b[2] + out.v_gamma_ab * grad_a[2],
                ];

                za_vec.clear();
                zb_vec.clear();
                let half_wva = 0.5 * w * out.v_rho_a;
                let half_wvb = 0.5 * w * out.v_rho_b;
                for a_i in &active {
                    za_vec.push(half_wva * a_i.val + w * dot3(ua, a_i.grad));
                    zb_vec.push(half_wvb * a_i.val + w * dot3(ub, a_i.grad));
                }

                let m = active.len();
                for i in 0..m {
                    let mu = active[i].mu;
                    let phi_i = active[i].val;
                    let za_i = za_vec[i];
                    let zb_i = zb_vec[i];
                    let row_va = &mut va_loc[mu * n..(mu + 1) * n];
                    let row_vb = &mut vb_loc[mu * n..(mu + 1) * n];
                    for j in 0..m {
                        let nu = active[j].mu;
                        let phi_j = active[j].val;
                        row_va[nu] += za_i * phi_j + phi_i * za_vec[j];
                        row_vb[nu] += zb_i * phi_j + phi_i * zb_vec[j];
                    }
                }
            }

            (e_loc, va_loc, vb_loc)
        })
        .reduce(
            || (0.0f64, vec![0.0f64; n * n], vec![0.0f64; n * n]),
            |(ea, mut va1, mut vb1), (eb, va2, vb2)| {
                for (d, s) in va1.iter_mut().zip(va2.iter()) {
                    *d += *s;
                }
                for (d, s) in vb1.iter_mut().zip(vb2.iter()) {
                    *d += *s;
                }
                (ea + eb, va1, vb1)
            },
        );

    let mut va = Matrix2D::zeros(n, n);
    let mut vb = Matrix2D::zeros(n, n);
    va.data = va_data;
    vb.data = vb_data;
    (e_xc, va, vb)
}

/// Evaluates the exact analytical nuclear gradient $\left.\frac{\partial E_{\text{xc}}}{\partial \mathbf{R}_A}\right|_{P^\alpha, P^\beta}$
/// in Eh/Bohr for both closed-shell (`pa = pb = 0.5 * P_total`) and open-shell (`pa, pb`) Kohn-Sham DFT.
pub fn xc_nuclear_gradient(
    mol: &Molecule,
    basis: &BasisSet,
    grid: &MolecularGrid,
    func: XcFunctional,
    pa: &Matrix2D,
    pb: &Matrix2D,
) -> Vec<[f64; 3]> {
    let n_atoms = mol.atoms.len();
    let n = basis.n_ao;
    if n_atoms <= 1 || n == 0 || grid.points.is_empty() {
        return vec![[0.0; 3]; n_atoms];
    }

    let shells = PrecomputedAoShell::from_basis(basis);
    let pa_slice = &pa.data;
    let pb_slice = &pb.data;

    grid.points
        .par_chunks(GRID_CHUNK_SIZE)
        .map(|chunk| {
            let mut local_grad = vec![[0.0f64; 3]; n_atoms];
            let mut active: Vec<AoHessVal> = Vec::with_capacity(n);
            let mut phi_pa: Vec<f64> = Vec::with_capacity(n);
            let mut phi_pb: Vec<f64> = Vec::with_capacity(n);
            let mut gpa: Vec<[f64; 3]> = Vec::with_capacity(n);
            let mut gpb: Vec<[f64; 3]> = Vec::with_capacity(n);

            for gp in chunk {
                if gp.raw_weight.abs() < 1e-24 {
                    continue;
                }
                eval_aos_hess(&shells, gp.coords, &mut active);
                if active.is_empty() {
                    continue;
                }

                phi_pa.clear();
                phi_pb.clear();
                gpa.clear();
                gpb.clear();

                let mut rho_a = 0.0;
                let mut rho_b = 0.0;
                let mut grad_a = [0.0f64; 3];
                let mut grad_b = [0.0f64; 3];

                for a_i in &active {
                    let row_a = &pa_slice[a_i.mu * n..(a_i.mu + 1) * n];
                    let row_b = &pb_slice[a_i.mu * n..(a_i.mu + 1) * n];
                    let mut sa = 0.0;
                    let mut sb = 0.0;
                    let mut ga_vec = [0.0f64; 3];
                    let mut gb_vec = [0.0f64; 3];

                    for a_j in &active {
                        let p_a = row_a[a_j.mu];
                        let p_b = row_b[a_j.mu];
                        sa += p_a * a_j.val;
                        sb += p_b * a_j.val;
                        for q in 0..3 {
                            ga_vec[q] += p_a * a_j.grad[q];
                            gb_vec[q] += p_b * a_j.grad[q];
                        }
                    }

                    phi_pa.push(sa);
                    phi_pb.push(sb);
                    gpa.push(ga_vec);
                    gpb.push(gb_vec);

                    rho_a += a_i.val * sa;
                    rho_b += a_i.val * sb;
                    for q in 0..3 {
                        grad_a[q] += 2.0 * a_i.grad[q] * sa;
                        grad_b[q] += 2.0 * a_i.grad[q] * sb;
                    }
                }

                let ra = rho_a.max(0.0);
                let rb = rho_b.max(0.0);
                if ra + rb < DENSITY_SCREEN {
                    continue;
                }

                let gaa = dot3(grad_a, grad_a);
                let gab = dot3(grad_a, grad_b);
                let gbb = dot3(grad_b, grad_b);
                let out = eval_xc_spin(func, ra, rb, gaa, gab, gbb);

                let w = gp.weight;
                if w.abs() > 1e-24 {
                    let ua = [
                        2.0 * out.v_gamma_aa * grad_a[0] + out.v_gamma_ab * grad_b[0],
                        2.0 * out.v_gamma_aa * grad_a[1] + out.v_gamma_ab * grad_b[1],
                        2.0 * out.v_gamma_aa * grad_a[2] + out.v_gamma_ab * grad_b[2],
                    ];
                    let ub = [
                        2.0 * out.v_gamma_bb * grad_b[0] + out.v_gamma_ab * grad_a[0],
                        2.0 * out.v_gamma_bb * grad_b[1] + out.v_gamma_ab * grad_a[1],
                        2.0 * out.v_gamma_bb * grad_b[2] + out.v_gamma_ab * grad_a[2],
                    ];

                    // 1. Fixed-grid AO center nuclear gradient + 2. Host-atom translation invariance:
                    // Since r_g = R_host + u_g co-moves with host atom H, moving R_H shifts r_g by +dR_H,
                    // contributing +d(f_xc)/dr_g = -sum_B f_AO(B, g) onto atom H!
                    let mut sum_ao_force = [0.0f64; 3];

                    for (idx, a_i) in active.iter().enumerate() {
                        let atom = a_i.atom_index;
                        let sa = phi_pa[idx];
                        let sb = phi_pb[idx];
                        let ua_dot_gpa = dot3(ua, gpa[idx]);
                        let ub_dot_gpb = dot3(ub, gpb[idx]);

                        for q in 0..3 {
                            let dphi_dq = a_i.grad[q];
                            let hrow = a_i.hess_row(q);

                            let term_a = out.v_rho_a * dphi_dq * sa
                                + dot3(ua, hrow) * sa
                                + dphi_dq * ua_dot_gpa;
                            let term_b = out.v_rho_b * dphi_dq * sb
                                + dot3(ub, hrow) * sb
                                + dphi_dq * ub_dot_gpb;

                            let f_aq = -2.0 * w * (term_a + term_b);
                            local_grad[atom][q] += f_aq;
                            sum_ao_force[q] += f_aq;
                        }
                    }

                    for q in 0..3 {
                        local_grad[gp.host_atom][q] -= sum_ao_force[q];
                    }
                }

                // 3. Analytical Becke quadrature weight nuclear derivative:
                becke_weight_nuclear_gradient(
                    mol,
                    gp.host_atom,
                    gp.coords,
                    gp.raw_weight,
                    &mut local_grad,
                    out.f_xc,
                );
            }

            local_grad
        })
        .reduce(
            || vec![[0.0f64; 3]; n_atoms],
            |mut ga, gb| {
                for i in 0..n_atoms {
                    for q in 0..3 {
                        ga[i][q] += gb[i][q];
                    }
                }
                ga
            },
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::GridLevel;
    use calchem_basis::BasisSet;
    use calchem_model::{Atom, Molecule};

    fn distorted_water(atoms_override: Option<Vec<[f64; 3]>>) -> Molecule {
        let coords = atoms_override.unwrap_or_else(|| {
            vec![
                [0.02, -0.04, 0.11],
                [0.81, 0.59, -0.37],
                [-0.74, 0.63, -0.42],
            ]
        });
        Molecule::new(
            vec![
                Atom::from_angstrom("O", coords[0]).unwrap(),
                Atom::from_angstrom("H", coords[1]).unwrap(),
                Atom::from_angstrom("H", coords[2]).unwrap(),
            ],
            0,
            1,
        )
    }

    fn make_positive_density(n: usize, seed_shift: usize) -> Matrix2D {
        // Construct strictly positive-definite P = A * A^T + 0.05 * I so density > 0 everywhere
        // even under off-diagonal finite-difference perturbations.
        let mut a = Matrix2D::zeros(n, 4);
        for mu in 0..n {
            for k in 0..4 {
                let x = ((mu * 7 + k * 13 + seed_shift * 5 + 3) as f64 * 0.37).sin();
                a.set(mu, k, 0.22 * x);
            }
        }
        let mut p = Matrix2D::zeros(n, n);
        for mu in 0..n {
            for nu in 0..n {
                let mut s = if mu == nu { 0.05 } else { 0.0 };
                for k in 0..4 {
                    s += a.get(mu, k) * a.get(nu, k);
                }
                p.set(mu, nu, s);
            }
        }
        p
    }

    #[test]
    fn test_vxc_rks_and_uks_matrix_matches_exact_density_derivative() {
        let mol = distorted_water(None);
        let basis = BasisSet::from_name("6-31g(d)", &mol).expect("6-31g(d)");
        let grid = MolecularGrid::build(&mol, GridLevel::Coarse);
        let n = basis.n_ao;

        let p_tot = make_positive_density(n, 1);
        let eps = 1e-6;

        for func in [XcFunctional::Svwn5, XcFunctional::Pbe, XcFunctional::B3lyp] {
            let (_exc, vxc) = evaluate_vxc_rks(&basis, &grid, func, &p_tot);

            // Check symmetry and exact partial derivative wrt P^t on representative (mu, nu) pairs
            for &(mu, nu) in &[(0, 0), (0, 3), (2, 7), (5, 11), (8, 14), (12, 12)] {
                assert!(
                    (vxc.get(mu, nu) - vxc.get(nu, mu)).abs() < 1e-14,
                    "V_xc not symmetric at ({mu},{nu})"
                );
                let mut p_p = p_tot.clone();
                let mut p_m = p_tot.clone();
                if mu == nu {
                    p_p.set(mu, mu, p_tot.get(mu, mu) + eps);
                    p_m.set(mu, mu, p_tot.get(mu, mu) - eps);
                    let (ep, _) = evaluate_vxc_rks(&basis, &grid, func, &p_p);
                    let (em, _) = evaluate_vxc_rks(&basis, &grid, func, &p_m);
                    let fd = (ep - em) / (2.0 * eps);
                    assert!(
                        (vxc.get(mu, mu) - fd).abs() < 1e-9,
                        "{func:?} RKS diag ({mu},{mu}): ana={} fd={fd}",
                        vxc.get(mu, mu)
                    );
                } else {
                    p_p.set(mu, nu, p_tot.get(mu, nu) + eps);
                    p_p.set(nu, mu, p_tot.get(nu, mu) + eps);
                    p_m.set(mu, nu, p_tot.get(mu, nu) - eps);
                    p_m.set(nu, mu, p_tot.get(nu, mu) - eps);
                    let (ep, _) = evaluate_vxc_rks(&basis, &grid, func, &p_p);
                    let (em, _) = evaluate_vxc_rks(&basis, &grid, func, &p_m);
                    // Perturbing both (mu,nu) and (nu,mu) by +eps changes E_xc by 2 * V_xc[mu,nu] * eps!
                    let fd = (ep - em) / (4.0 * eps);
                    assert!(
                        (vxc.get(mu, nu) - fd).abs() < 1e-9,
                        "{func:?} RKS off-diag ({mu},{nu}): ana={} fd={fd}",
                        vxc.get(mu, nu)
                    );
                }
            }

            // Also verify closed-shell UKS (pa = pb = 0.5 * p_tot) matches RKS identically!
            let mut half_p = p_tot.clone();
            for v in &mut half_p.data {
                *v *= 0.5;
            }
            let (exc_u, va, vb) = evaluate_vxc_uks(&basis, &grid, func, &half_p, &half_p);
            let (exc_r, vr) = evaluate_vxc_rks(&basis, &grid, func, &p_tot);
            assert!((exc_u - exc_r).abs() < 1e-13);
            for idx in 0..n * n {
                assert!((va.data[idx] - vr.data[idx]).abs() < 1e-13);
                assert!((vb.data[idx] - vr.data[idx]).abs() < 1e-13);
            }
        }

        // Open-shell UKS asymmetric spin check (`pa != pb`)
        let pa = make_positive_density(n, 2);
        let pb = make_positive_density(n, 5);
        for func in [XcFunctional::Blyp, XcFunctional::B3lyp, XcFunctional::Pbe0] {
            let (_, va, vb) = evaluate_vxc_uks(&basis, &grid, func, &pa, &pb);
            for &(mu, nu) in &[(1, 1), (1, 6), (4, 10), (9, 15)] {
                let mut pa_p = pa.clone();
                let mut pa_m = pa.clone();
                let mut pb_p = pb.clone();
                let mut pb_m = pb.clone();
                let denom = if mu == nu {
                    pa_p.set(mu, mu, pa.get(mu, mu) + eps);
                    pa_m.set(mu, mu, pa.get(mu, mu) - eps);
                    pb_p.set(mu, mu, pb.get(mu, mu) + eps);
                    pb_m.set(mu, mu, pb.get(mu, mu) - eps);
                    2.0 * eps
                } else {
                    pa_p.set(mu, nu, pa.get(mu, nu) + eps);
                    pa_p.set(nu, mu, pa.get(nu, mu) + eps);
                    pa_m.set(mu, nu, pa.get(mu, nu) - eps);
                    pa_m.set(nu, mu, pa.get(nu, mu) - eps);
                    pb_p.set(mu, nu, pb.get(mu, nu) + eps);
                    pb_p.set(nu, mu, pb.get(nu, mu) + eps);
                    pb_m.set(mu, nu, pb.get(mu, nu) - eps);
                    pb_m.set(nu, mu, pb.get(nu, mu) - eps);
                    4.0 * eps
                };
                let (epa, _, _) = evaluate_vxc_uks(&basis, &grid, func, &pa_p, &pb);
                let (ema, _, _) = evaluate_vxc_uks(&basis, &grid, func, &pa_m, &pb);
                let fda = (epa - ema) / denom;
                assert!(
                    (va.get(mu, nu) - fda).abs() < 1e-9,
                    "{func:?} UKS alpha ({mu},{nu}): ana={} fd={fda}",
                    va.get(mu, nu)
                );

                let (epb, _, _) = evaluate_vxc_uks(&basis, &grid, func, &pa, &pb_p);
                let (emb, _, _) = evaluate_vxc_uks(&basis, &grid, func, &pa, &pb_m);
                let fdb = (epb - emb) / denom;
                assert!(
                    (vb.get(mu, nu) - fdb).abs() < 1e-9,
                    "{func:?} UKS beta ({mu},{nu}): ana={} fd={fdb}",
                    vb.get(mu, nu)
                );
            }
        }
    }

    #[test]
    fn test_xc_nuclear_gradient_matches_exact_finite_differences_with_moving_grid_and_basis() {
        let mol = distorted_water(None);
        let basis = BasisSet::from_name("6-31g(d)", &mol).expect("6-31g(d)");
        let grid = MolecularGrid::build(&mol, GridLevel::Coarse);
        let n = basis.n_ao;

        let pa = make_positive_density(n, 3);
        let pb = make_positive_density(n, 8);
        let h_bohr = 1e-5;

        for func in [XcFunctional::Svwn5, XcFunctional::Pbe, XcFunctional::B3lyp] {
            let ana = xc_nuclear_gradient(&mol, &basis, &grid, func, &pa, &pb);

            // Translational invariance check: sum_A dE_xc/dR_A == 0!
            for q in 0..3 {
                let sum_q: f64 = ana.iter().map(|g| g[q]).sum();
                assert!(
                    sum_q.abs() < 1e-12,
                    "{func:?} translational invariance violated on axis {q}: sum={sum_q:.3e}"
                );
            }

            // Compare every nuclear coordinate against central FD of E_xc(R +- h) at fixed (pa, pb)!
            for atom_idx in 0..mol.atoms.len() {
                for q in 0..3 {
                    let mut mol_p = mol.clone();
                    let mut mol_m = mol.clone();
                    mol_p.atoms[atom_idx].position_bohr[q] += h_bohr;
                    mol_m.atoms[atom_idx].position_bohr[q] -= h_bohr;

                    let basis_p = BasisSet::from_name("6-31g(d)", &mol_p).unwrap();
                    let basis_m = BasisSet::from_name("6-31g(d)", &mol_m).unwrap();
                    let grid_p = MolecularGrid::build(&mol_p, GridLevel::Coarse);
                    let grid_m = MolecularGrid::build(&mol_m, GridLevel::Coarse);

                    let (ep, _, _) = evaluate_vxc_uks(&basis_p, &grid_p, func, &pa, &pb);
                    let (em, _, _) = evaluate_vxc_uks(&basis_m, &grid_m, func, &pa, &pb);
                    let fd = (ep - em) / (2.0 * h_bohr);

                    let diff = (ana[atom_idx][q] - fd).abs();
                    assert!(
                        diff < 1e-8,
                        "{func:?} atom {atom_idx} axis {q}: ana={:.10e} fd={:.10e} diff={diff:.3e}",
                        ana[atom_idx][q],
                        fd
                    );
                }
            }
        }
    }
}
