//! Mulliken / Löwdin 布居分析、Mayer 键级与偶极矩。
//!
//! 约定（闭壳层，密度矩阵 P）：
//! * Mulliken 布居：原子 A 的电子数 = Σ_{μ∈A} (P S)_{μμ}，电荷 = Z_A − 布居。
//! * Löwdin 布居：先对称正交化 P' = S^{1/2} P S^{1/2}，再取对角，电荷 = Z_A − 布居。
//! * 偶极矩：μ_x = Σ_A Z_A R_Ax − Σ_{μν} P_{μν} (D_x)_{νμ}，D_x=<μ|x|ν>。
//!   1 a₀·e = 2.541746 D（Debye）。

use calchem_basis::{BasisSetData, MoleculeCore};
use calchem_tensor::PackedSymMatrix;
use faer::linalg::solvers::SelfAdjointEigendecomposition;
use faer::Mat;

/// 原子单位电荷到 Debye 的换算（1 e·a₀ = 2.541746473 Debye）。
pub const AU_TO_DEBYE: f64 = 2.541746473;

/// 布居分析结果。
pub struct PopulationReport {
    pub n_atoms: usize,
    /// Mulliken 部分电荷（原子核电荷 − 电子布居）。
    pub mulliken_charges: Vec<f64>,
    /// Löwdin 部分电荷。
    pub lowdin_charges: Vec<f64>,
    /// Mulliken 每原子总电子布居。
    pub mulliken_pop: Vec<f64>,
    /// Löwdin 每原子总电子布居。
    pub lowdin_pop: Vec<f64>,
    /// 偶极矩（原子单位 e·a₀）。
    pub dipole: [f64; 3],
    /// 偶极矩（Debye）。
    pub dipole_debye: [f64; 3],
    /// 偶极矩大小（Debye）。
    pub dipole_magnitude_debye: f64,
}

/// 建立 AO 索引 → 原子索引的映射。
fn ao_to_atom(basis: &BasisSetData) -> Vec<usize> {
    let nbf = basis.nbf;
    let mut map = vec![0usize; nbf];
    for sh in 0..basis.shells.len() {
        let (s, e) = basis.shell_ao_range(sh);
        let a = basis.shells[sh].atom_idx as usize;
        for k in s..e {
            map[k] = a;
        }
    }
    map
}

/// 计算 S^{1/2}（对称正交化的平方根），返回 faer 矩阵。
fn sqrt_matrix(n: usize, s: &PackedSymMatrix) -> Mat<f64> {
    let smat = Mat::from_fn(n, n, |i, j| s.get(i, j));
    let e = SelfAdjointEigendecomposition::new(smat.as_ref(), faer::Side::Lower);
    let mut out = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                let lam = *e.s().column_vector().get(k);
                acc += e.u().read(i, k) * lam.sqrt() * e.u().read(j, k);
            }
            out[(i, j)] = acc;
        }
    }
    out
}

/// 完整布居分析与偶极矩。
pub fn population_analysis(
    mol: &MoleculeCore,
    basis: &BasisSetData,
    s: &PackedSymMatrix,
    dm: &PackedSymMatrix,
) -> PopulationReport {
    let n = basis.nbf;
    let n_atoms = mol.atoms.len();
    let ao_atom = ao_to_atom(basis);

    // ---- Mulliken：(P S)_{μμ} 按原子求和。----
    let mut mulliken_pop = vec![0.0; n_atoms];
    for mu in 0..n {
        let mut ps_mu = 0.0;
        for nu in 0..n {
            ps_mu += dm.get(mu, nu) * s.get(nu, mu);
        }
        mulliken_pop[ao_atom[mu]] += ps_mu;
    }

    // ---- Löwdin：P' = S^{1/2} P S^{1/2}，取对角。----
    let s_half = sqrt_matrix(n, s);
    let pmat = Mat::from_fn(n, n, |i, j| dm.get(i, j));
    // tmp = S^{1/2} P
    let mut tmp = Mat::<f64>::zeros(n, n);
    faer::linalg::matmul::matmul(
        tmp.as_mut(),
        s_half.as_ref(),
        pmat.as_ref(),
        None,
        1.0,
        faer::Parallelism::None,
    );
    // pp = tmp S^{1/2}
    let mut pp = Mat::<f64>::zeros(n, n);
    faer::linalg::matmul::matmul(
        pp.as_mut(),
        tmp.as_ref(),
        s_half.as_ref(),
        None,
        1.0,
        faer::Parallelism::None,
    );
    let mut lowdin_pop = vec![0.0; n_atoms];
    for mu in 0..n {
        lowdin_pop[ao_atom[mu]] += pp[(mu, mu)];
    }

    // ---- 电荷 = Z_A − 布居。----
    let mut mulliken_charges = vec![0.0; n_atoms];
    let mut lowdin_charges = vec![0.0; n_atoms];
    for a in 0..n_atoms {
        let z = mol.atoms[a].atomic_number as f64;
        mulliken_charges[a] = z - mulliken_pop[a];
        lowdin_charges[a] = z - lowdin_pop[a];
    }

    // ---- 偶极矩：μ = Σ_A Z_A R_A − Σ_{μν} P_{μν} D_{νμ}。----
    let mut dipole = [0.0; 3];
    for ax in 0..3 {
        let d = calchem_int::compute_dipole_matrix(basis, ax);
        let mut nuc = 0.0;
        for a in 0..n_atoms {
            nuc += mol.atoms[a].atomic_number as f64 * mol.atoms[a].coord_bohr[ax];
        }
        let mut elec = 0.0;
        for mu in 0..n {
            for nu in 0..n {
                elec += dm.get(mu, nu) * d[nu * n + mu];
            }
        }
        dipole[ax] = nuc - elec;
    }
    let dipole_debye = [
        dipole[0] * AU_TO_DEBYE,
        dipole[1] * AU_TO_DEBYE,
        dipole[2] * AU_TO_DEBYE,
    ];
    let dipole_magnitude_debye =
        (dipole_debye[0].powi(2) + dipole_debye[1].powi(2) + dipole_debye[2].powi(2)).sqrt();

    PopulationReport {
        n_atoms,
        mulliken_charges,
        lowdin_charges,
        mulliken_pop,
        lowdin_pop,
        dipole,
        dipole_debye,
        dipole_magnitude_debye,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// H2（对称，沿 z）：电荷与偶极均应为 0。
    #[test]
    fn population_h2_symmetric() {
        let coords = [0.0, 0.0, 0.0, 0.0, 0.0, 1.4];
        let (mol, basis) = calchem_scf::build_molecule_basis(
            &[1, 1],
            &coords,
            0,
            1,
            "sto-3g",
            calchem_basis::AngularType::Cartesian,
        )
        .unwrap();
        let out =
            calchem_scf::execute_scf_pipeline(&[1, 1], &coords, 0, 1, "rhf", "sto-3g", None, false, 0)
                .unwrap();
        let rep = population_analysis(&mol, &basis, &out.overlap, &out.density);
        for a in 0..2 {
            assert!(rep.mulliken_charges[a].abs() < 1e-8, "mulliken {}", a);
            assert!(rep.lowdin_charges[a].abs() < 1e-8, "lowdin {}", a);
        }
        for ax in 0..3 {
            assert!(rep.dipole[ax].abs() < 1e-8, "dipole {}", ax);
        }
        assert!(rep.dipole_magnitude_debye < 1e-6);
    }

    /// H2O（Szabo 几何）：与 Python 参照（numpy）逐项比对。
    #[test]
    fn population_h2o_reference() {
        let coords = [
            0.0, 0.0, 0.0, // O
            0.0, 1.432, 0.0, // H1
            1.112, -0.716, 0.0, // H2
        ];
        let (mol, basis) = calchem_scf::build_molecule_basis(
            &[8, 1, 1],
            &coords,
            0,
            1,
            "sto-3g",
            calchem_basis::AngularType::Cartesian,
        )
        .unwrap();
        let out =
            calchem_scf::execute_scf_pipeline(&[8, 1, 1], &coords, 0, 1, "rhf", "sto-3g", None, false, 0)
                .unwrap();
        let rep = population_analysis(&mol, &basis, &out.overlap, &out.density);

        // Mulliken（精确积分波函数的准确值；Python 用同一密度复算一致）。
        let exp_mull = [-0.68629491, 0.32419911, 0.3620958];
        for a in 0..3 {
            assert!(
                (rep.mulliken_charges[a] - exp_mull[a]).abs() < 1e-6,
                "mulliken {} = {} expected {}",
                a, rep.mulliken_charges[a], exp_mull[a]
            );
        }
        // Löwdin（精确积分波函数的准确值）。
        let exp_low = [-0.41722924, 0.20552155, 0.2117077];
        for a in 0..3 {
            assert!(
                (rep.lowdin_charges[a] - exp_low[a]).abs() < 1e-6,
                "lowdin {} = {} expected {}",
                a, rep.lowdin_charges[a], exp_low[a]
            );
        }
        // 偶极（a.u.，精确积分波函数的准确值）。
        let exp_dip = [0.58344373, 0.32836218, 0.0];
        for ax in 0..3 {
            assert!(
                (rep.dipole[ax] - exp_dip[ax]).abs() < 1e-6,
                "dipole {} = {} expected {}",
                ax, rep.dipole[ax], exp_dip[ax]
            );
        }
        // |μ| ≈ 1.7017 Debye
        assert!(
            (rep.dipole_magnitude_debye - 1.7017).abs() < 1e-3,
            "|mu| = {}",
            rep.dipole_magnitude_debye
        );
        // 电荷总和 = 0（中性分子）。
        let sum_m: f64 = rep.mulliken_charges.iter().sum();
        let sum_l: f64 = rep.lowdin_charges.iter().sum();
        assert!(sum_m.abs() < 1e-8, "sum mulliken {}", sum_m);
        assert!(sum_l.abs() < 1e-8, "sum lowdin {}", sum_l);
    }
}

