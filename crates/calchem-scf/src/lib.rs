//! RHF SCF 求解器。

use calchem_basis::BasisSet;
use calchem_int::IntegralEngine;
use calchem_model::{CalcError, Molecule};
use calchem_tensor::{eigh_sym, mat_add, mat_mul, mat_scale, trace, transpose};

pub struct ScfResult {
    pub energy: f64,
    pub converged: bool,
    pub n_iter: u32,
    pub mo_energies: Vec<f64>,
    pub mo_coeffs: Vec<Vec<f64>>,
    pub dm: Vec<Vec<f64>>,
}

/// 执行 RHF SCF 迭代
pub fn rhf(mol: &Molecule, basis: &BasisSet) -> Result<ScfResult, CalcError> {
    let nbf = basis.nbf;
    let nelec: i32 = mol.atoms.iter().map(|a| a.z as i32).sum::<i32>() - mol.charge;
    let nocc = (nelec / 2) as usize;

    if nocc == 0 {
        return Err(CalcError::InvalidInput("没有占据轨道".into()));
    }

    // 构建原子列表
    let atoms: Vec<(u8, [f64; 3])> = mol.atoms.iter().map(|a| (a.z, a.position_bohr)).collect();

    // 计算积分
    let engine = IntegralEngine::build(basis, &atoms);

    let s = engine.overlap.clone();
    let h_core = mat_add(&engine.kinetic, &engine.nuclear);

    // 构建 S^{-1/2}
    let (s_eigvals, s_eigvecs) = eigh_sym(&s);
    let mut s_sqrt_inv = vec![vec![0.0; nbf]; nbf];
    for i in 0..nbf {
        for j in 0..nbf {
            s_sqrt_inv[i][j] = s_eigvecs[i][j] / s_eigvals[j].sqrt();
        }
    }

    // 初始猜测：core Hamiltonian
    let f_orth = mat_mul(
        &mat_mul(&transpose(&s_sqrt_inv), &h_core),
        &s_sqrt_inv,
    );
    let (mut mo_energies, c_orth) = eigh_sym(&f_orth);
    let mut mo_coeffs = mat_mul(&s_sqrt_inv, &c_orth);

    // 初始密度矩阵
    let mut dm = build_density(&mo_coeffs, nocc, nbf);

    let mut energy = 0.0;
    let mut converged = false;
    let mut n_iter = 0;

    for iteration in 0..100 {
        n_iter = iteration + 1;

        // 构建 Fock 矩阵: F = H + J - 0.5K
        let fock = build_fock(&h_core, &dm, &engine, nbf);

        // 计算能量: E = trace(P*H) + 0.5*trace(P*G) + nuc_rep
        // G = F - H, so E = 0.5*trace(P*H) + 0.5*trace(P*F) + nuc_rep
        let energy_new = 0.5 * trace(&mat_mul(&dm, &h_core))
            + 0.5 * trace(&mat_mul(&dm, &fock))
            + engine.nuclear_repulsion;

        // 更新 Fock 并对角化
        let f_orth = mat_mul(
            &mat_mul(&transpose(&s_sqrt_inv), &fock),
            &s_sqrt_inv,
        );
        let (eigvals, c_orth_new) = eigh_sym(&f_orth);
        mo_coeffs = mat_mul(&s_sqrt_inv, &c_orth_new);
        mo_energies = eigvals;

        let dm_new = build_density(&mo_coeffs, nocc, nbf);

        // 检查收敛
        let delta_e = (energy_new - energy).abs();
        let mut delta_dm_sum = 0.0;
        for i in 0..nbf {
            for j in 0..nbf {
                delta_dm_sum += (dm_new[i][j] - dm[i][j]).powi(2);
            }
        }
        let delta_dm = delta_dm_sum / (nbf * nbf) as f64;

        energy = energy_new;
        dm = dm_new;

        if iteration > 3 && delta_e < 1e-10 && delta_dm < 1e-8 {
            converged = true;
            break;
        }
    }

    Ok(ScfResult {
        energy,
        converged,
        n_iter,
        mo_energies,
        mo_coeffs,
        dm,
    })
}

fn build_density(mo_coeffs: &[Vec<f64>], nocc: usize, nbf: usize) -> Vec<Vec<f64>> {
    // P = 2 * C_occ * C_occ^T
    let mut dm = vec![vec![0.0; nbf]; nbf];
    for i in 0..nbf {
        for j in 0..nbf {
            let mut sum = 0.0;
            for k in 0..nocc {
                sum += mo_coeffs[i][k] * mo_coeffs[j][k];
            }
            dm[i][j] = 2.0 * sum;
        }
    }
    dm
}

fn build_fock(h_core: &[Vec<f64>], dm: &[Vec<f64>], engine: &IntegralEngine, nbf: usize) -> Vec<Vec<f64>> {
    // F = H + G
    // G_mn = sum_ls P_ls * [(mn|ls) - 0.5(ml|ns)]
    let mut g = vec![vec![0.0; nbf]; nbf];

    for m in 0..nbf {
        for n in 0..nbf {
            let mut j = 0.0;
            let mut k = 0.0;
            for l in 0..nbf {
                for s in 0..nbf {
                    j += dm[l][s] * engine.eri(m, n, l, s);
                    k += dm[l][s] * engine.eri(m, l, n, s);
                }
            }
            g[m][n] = j - 0.5 * k;
        }
    }

    mat_add(h_core, &g)
}
