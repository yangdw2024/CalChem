//! Fast shell-screened evaluation of Cartesian Gaussian atomic orbitals $\phi_\mu(\mathbf{r})$,
//! gradients $\nabla\phi_\mu(\mathbf{r})$, and Hessians $\nabla\nabla\phi_\mu(\mathbf{r})$ on DFT quadrature points.

use calchem_basis::{BasisSet, Shell};

#[derive(Clone, Copy)]
struct PrimData {
    exponent: f64,
    weighted_norm: f64,
}

pub struct PrecomputedAoShell {
    pub atom_index: usize,
    pub center: [f64; 3],
    pub ao_offset: usize,
    pub r2_cutoff: f64,
    prims: Vec<PrimData>,
    comps: &'static [[u8; 3]],
    comp_scales: Vec<f64>,
}

impl PrecomputedAoShell {
    pub fn from_basis(basis: &BasisSet) -> Vec<Self> {
        basis.shells.iter().map(Self::from_shell).collect()
    }

    pub fn from_shell(sh: &Shell) -> Self {
        let comps = sh.components();
        let l = sh.shell_type.angular_momentum() as f64;
        let alpha_min = sh
            .primitives
            .iter()
            .map(|p| p.exponent)
            .fold(f64::INFINITY, f64::min)
            .max(1e-12);
        let r2_cutoff = (40.0 + 4.0 * l) / alpha_min;

        let prims = sh
            .primitives
            .iter()
            .map(|p| PrimData {
                exponent: p.exponent,
                weighted_norm: p.coeff * p.norm,
            })
            .collect();

        let lead_norm = sh.primitives.first().map(|p| p.norm).unwrap_or(1.0);
        let comp_scales = comps
            .iter()
            .map(|&c| {
                if sh.primitives.is_empty() || lead_norm == 0.0 {
                    1.0
                } else {
                    sh.primitive_norm_for(0, c) / lead_norm
                }
            })
            .collect();

        Self {
            atom_index: sh.atom_index,
            center: sh.center,
            ao_offset: sh.ao_offset,
            r2_cutoff,
            prims,
            comps,
            comp_scales,
        }
    }
}

#[inline]
fn pow_d0_d1_d2(x: f64, l: u8) -> (f64, f64, f64) {
    match l {
        0 => (1.0, 0.0, 0.0),
        1 => (x, 1.0, 0.0),
        2 => (x * x, 2.0 * x, 2.0),
        3 => (x * x * x, 3.0 * x * x, 6.0 * x),
        _ => {
            let lf = l as f64;
            (
                x.powi(l as i32),
                lf * x.powi(l as i32 - 1),
                lf * (lf - 1.0) * x.powi(l as i32 - 2),
            )
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AoGradVal {
    pub mu: usize,
    pub atom_index: usize,
    pub val: f64,
    pub grad: [f64; 3],
}

#[derive(Debug, Clone, Copy)]
pub struct AoHessVal {
    pub mu: usize,
    pub atom_index: usize,
    pub val: f64,
    pub grad: [f64; 3],
    /// `[xx, xy, xz, yy, yz, zz]`
    pub hess: [f64; 6],
}

impl AoHessVal {
    /// Returns `[d/dq (dphi/dx), d/dq (dphi/dy), d/dq (dphi/dz)]` for axis `q in 0..3`.
    #[inline]
    pub fn hess_row(&self, q: usize) -> [f64; 3] {
        match q {
            0 => [self.hess[0], self.hess[1], self.hess[2]],
            1 => [self.hess[1], self.hess[3], self.hess[4]],
            2 => [self.hess[2], self.hess[4], self.hess[5]],
            _ => unreachable!(),
        }
    }
}

/// Populates `out` with all non-negligible AOs `(val, grad)` at point `r`.
#[inline]
pub fn eval_aos_grad(shells: &[PrecomputedAoShell], r: [f64; 3], out: &mut Vec<AoGradVal>) {
    out.clear();
    for sh in shells {
        let dx = r[0] - sh.center[0];
        let dy = r[1] - sh.center[1];
        let dz = r[2] - sh.center[2];
        let r2 = dx * dx + dy * dy + dz * dz;
        if r2 > sh.r2_cutoff {
            continue;
        }

        let mut r0 = 0.0;
        let mut r1 = 0.0;
        for p in &sh.prims {
            let e = p.weighted_norm * (-p.exponent * r2).exp();
            r0 += e;
            r1 -= 2.0 * p.exponent * e;
        }
        if r0.abs() < 1e-18 && r1.abs() < 1e-18 {
            continue;
        }

        for (ic, &c) in sh.comps.iter().enumerate() {
            let scale = sh.comp_scales[ic];
            let (px, dpx, _) = pow_d0_d1_d2(dx, c[0]);
            let (py, dpy, _) = pow_d0_d1_d2(dy, c[1]);
            let (pz, dpz, _) = pow_d0_d1_d2(dz, c[2]);

            let m = px * py * pz;
            let dm_x = dpx * py * pz;
            let dm_y = px * dpy * pz;
            let dm_z = px * py * dpz;

            let val = scale * m * r0;
            let gx = scale * (dm_x * r0 + m * dx * r1);
            let gy = scale * (dm_y * r0 + m * dy * r1);
            let gz = scale * (dm_z * r0 + m * dz * r1);

            if val.abs() > 1e-16 || gx.abs() > 1e-16 || gy.abs() > 1e-16 || gz.abs() > 1e-16 {
                out.push(AoGradVal {
                    mu: sh.ao_offset + ic,
                    atom_index: sh.atom_index,
                    val,
                    grad: [gx, gy, gz],
                });
            }
        }
    }
}

/// Populates `out` with all non-negligible AOs `(val, grad, hess)` at point `r`.
#[inline]
pub fn eval_aos_hess(shells: &[PrecomputedAoShell], r: [f64; 3], out: &mut Vec<AoHessVal>) {
    out.clear();
    for sh in shells {
        let dx = r[0] - sh.center[0];
        let dy = r[1] - sh.center[1];
        let dz = r[2] - sh.center[2];
        let r2 = dx * dx + dy * dy + dz * dz;
        if r2 > sh.r2_cutoff {
            continue;
        }

        let mut r0 = 0.0;
        let mut r1 = 0.0;
        let mut r2_rad = 0.0;
        for p in &sh.prims {
            let e = p.weighted_norm * (-p.exponent * r2).exp();
            r0 += e;
            r1 -= 2.0 * p.exponent * e;
            r2_rad += 4.0 * p.exponent * p.exponent * e;
        }
        if r0.abs() < 1e-18 && r1.abs() < 1e-18 && r2_rad.abs() < 1e-18 {
            continue;
        }

        for (ic, &c) in sh.comps.iter().enumerate() {
            let scale = sh.comp_scales[ic];
            let (px, dpx, d2px) = pow_d0_d1_d2(dx, c[0]);
            let (py, dpy, d2py) = pow_d0_d1_d2(dy, c[1]);
            let (pz, dpz, d2pz) = pow_d0_d1_d2(dz, c[2]);

            let m = px * py * pz;
            let dm = [dpx * py * pz, px * dpy * pz, px * py * dpz];
            let d2m_xx = d2px * py * pz;
            let d2m_xy = dpx * dpy * pz;
            let d2m_xz = dpx * py * dpz;
            let d2m_yy = px * d2py * pz;
            let d2m_yz = px * dpy * dpz;
            let d2m_zz = px * py * d2pz;

            let val = scale * m * r0;
            let grad = [
                scale * (dm[0] * r0 + m * dx * r1),
                scale * (dm[1] * r0 + m * dy * r1),
                scale * (dm[2] * r0 + m * dz * r1),
            ];

            let h_xx = scale * (d2m_xx * r0 + (2.0 * dm[0] * dx + m) * r1 + m * dx * dx * r2_rad);
            let h_xy =
                scale * (d2m_xy * r0 + (dm[0] * dy + dm[1] * dx) * r1 + m * dx * dy * r2_rad);
            let h_xz =
                scale * (d2m_xz * r0 + (dm[0] * dz + dm[2] * dx) * r1 + m * dx * dz * r2_rad);
            let h_yy = scale * (d2m_yy * r0 + (2.0 * dm[1] * dy + m) * r1 + m * dy * dy * r2_rad);
            let h_yz =
                scale * (d2m_yz * r0 + (dm[1] * dz + dm[2] * dy) * r1 + m * dy * dz * r2_rad);
            let h_zz = scale * (d2m_zz * r0 + (2.0 * dm[2] * dz + m) * r1 + m * dz * dz * r2_rad);

            out.push(AoHessVal {
                mu: sh.ao_offset + ic,
                atom_index: sh.atom_index,
                val,
                grad,
                hess: [h_xx, h_xy, h_xz, h_yy, h_yz, h_zz],
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use calchem_basis::{Primitive, ShellType};
    use calchem_model::{Atom, Molecule};

    #[test]
    fn test_ao_grad_and_hessian_s_p_d_f_finite_differences() {
        let mol = Molecule::new(
            vec![
                Atom::from_angstrom("O", [0.1, -0.2, 0.05]).unwrap(),
                Atom::from_angstrom("H", [0.8, 0.5, -0.3]).unwrap(),
            ],
            0,
            1,
        );
        let mut bs = BasisSet::from_name("6-31g(d,p)", &mol).unwrap();
        // Also append a synthetic F-shell on atom 0 to verify L=3 Hessian too!
        let f_offset = bs.n_ao;
        let alpha_f = 0.85;
        let f_sh = Shell {
            atom_index: 0,
            center: mol.atoms[0].position_bohr,
            shell_type: ShellType::F,
            primitives: vec![Primitive {
                exponent: alpha_f,
                coeff: 1.0,
                raw_coeff: 1.0,
                norm: 1.0,
            }],
            ao_offset: f_offset,
        };
        bs.shells.push(f_sh);
        bs.n_ao += 10;

        let shells = PrecomputedAoShell::from_basis(&bs);
        let r0 = [0.43, -0.61, 0.29];
        let mut ana = Vec::new();
        eval_aos_hess(&shells, r0, &mut ana);

        let h = 1e-6;
        for item in &ana {
            for q in 0..3 {
                let mut rp = r0;
                rp[q] += h;
                let mut rm = r0;
                rm[q] -= h;

                let mut vp_list = Vec::new();
                let mut vm_list = Vec::new();
                eval_aos_grad(&shells, rp, &mut vp_list);
                eval_aos_grad(&shells, rm, &mut vm_list);

                let vp = vp_list.iter().find(|x| x.mu == item.mu).unwrap();
                let vm = vm_list.iter().find(|x| x.mu == item.mu).unwrap();

                let fd_grad = (vp.val - vm.val) / (2.0 * h);
                assert!(
                    (item.grad[q] - fd_grad).abs() < 1e-9,
                    "mu={} q={q}: grad {} vs fd {}",
                    item.mu,
                    item.grad[q],
                    fd_grad
                );

                let h_row = item.hess_row(q);
                for p in 0..3 {
                    let fd_hess = (vp.grad[p] - vm.grad[p]) / (2.0 * h);
                    assert!(
                        (h_row[p] - fd_hess).abs() < 1e-8,
                        "mu={} ({q},{p}): hess {} vs fd {}",
                        item.mu,
                        h_row[p],
                        fd_hess
                    );
                }
            }
        }
    }
}
