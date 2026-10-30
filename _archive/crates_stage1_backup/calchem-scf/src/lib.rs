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

    // DIIS storage
    let diis_start = 3;
    let diis_max = 8;
    let mut diis_errors: Vec<Vec<f64>> = Vec::new(); // error vectors (flattened FDS-SDF)
    let mut diis_focks: Vec<Vec<Vec<f64>>> = Vec::new(); // stored Fock matrices

    for iteration in 0..100 {
        n_iter = iteration + 1;

        // 构建 Fock 矩阵: F = H + J - 0.5K
        let fock = build_fock(&h_core, &dm, &engine, nbf);

        // 计算能量
        let energy_new = 0.5 * trace(&mat_mul(&dm, &h_core))
            + 0.5 * trace(&mat_mul(&dm, &fock))
            + engine.nuclear_repulsion;

        // 计算 DIIS 误差: E = FDS - SDF (flattened)
        let fds = mat_mul(&mat_mul(&fock, &dm), &s);
        let sdf = mat_mul(&mat_mul(&s, &dm), &fock);
        let mut err_vec = Vec::with_capacity(nbf * nbf);
        for i in 0..nbf {
            for j in 0..nbf {
                err_vec.push(fds[i][j] - sdf[i][j]);
            }
        }
        let err_norm: f64 = err_vec.iter().map(|x| x*x).sum::<f64>().sqrt();

        // 存储 DIIS 数据
        if iteration >= diis_start {
            diis_errors.push(err_vec);
            diis_focks.push(fock.clone());
            if diis_errors.len() > diis_max {
                diis_errors.remove(0);
                diis_focks.remove(0);
            }
        }

        // DIIS 外推
        let fock_extrap = if diis_errors.len() >= 2 {
            let n_diis = diis_errors.len();
            // 构建 B 矩阵: B_ij = <e_i, e_j>
            let mut b = vec![vec![0.0; n_diis + 1]; n_diis + 1];
            for i in 0..n_diis {
                for j in 0..n_diis {
                    let mut dot = 0.0;
                    for k in 0..nbf*nbf {
                        dot += diis_errors[i][k] * diis_errors[j][k];
                    }
                    b[i][j] = dot;
                }
                b[i][n_diis] = -1.0;
                b[n_diis][i] = -1.0;
            }
            b[n_diis][n_diis] = 0.0;

            // 解线性系统: B * c = [0,0,...,0,-1]^T
            // 使用简单高斯消元
            let mut rhs = vec![0.0; n_diis + 1];
            rhs[n_diis] = -1.0;
            let coeffs = solve_linear(&b, &rhs, n_diis + 1);

            // 构建外推 Fock: F_new = sum c_i * F_i
            let mut f_ext = vec![vec![0.0; nbf]; nbf];
            for i in 0..n_diis {
                for r in 0..nbf {
                    for c in 0..nbf {
                        f_ext[r][c] += coeffs[i] * diis_focks[i][r][c];
                    }
                }
            }
            f_ext
        } else {
            fock.clone()
        };

        // 更新 Fock 并对角化
        let f_orth = mat_mul(
            &mat_mul(&transpose(&s_sqrt_inv), &fock_extrap),
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

        if iteration > 3 && delta_e < 1e-10 && err_norm < 1e-6 {
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

/// 简单高斯消元解线性方程组 Ax = b
fn solve_linear(a: &[Vec<f64>], b: &[f64], n: usize) -> Vec<f64> {
    let mut mat: Vec<Vec<f64>> = a.iter().enumerate().map(|(i, row)| {
        let mut r = row.clone();
        r.push(b[i]);
        r
    }).collect();

    // 前向消元
    for col in 0..n {
        // 找主元
        let mut max_row = col;
        for row in (col+1)..n {
            if mat[row][col].abs() > mat[max_row][col].abs() {
                max_row = row;
            }
        }
        mat.swap(col, max_row);

        // 消元
        let pivot = mat[col][col];
        if pivot.abs() < 1e-14 {
            // 奇异矩阵，返回零解
            return vec![0.0; n];
        }
        for row in (col+1)..n {
            let factor = mat[row][col] / pivot;
            for j in col..=n {
                mat[row][j] -= factor * mat[col][j];
            }
        }
    }

    // 回代
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = mat[i][n];
        for j in (i+1)..n {
            x[i] -= mat[i][j] * x[j];
        }
        x[i] /= mat[i][i];
    }
    x
}
