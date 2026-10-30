//! SCF 主求解器与输出结构（真实实现）。
//!
//! 闭壳层 RHF（RKS 由 DFT 模块扩展）。流程：
//!   1) 真实一/二电子积分（原生 Rust Obara-Saika）。
//!   2) 对称正交化 X = S^-1/2。
//!   3) 初始密度（SAD 或核心猜想）。
//!   4) 自洽迭代：J/K 收缩 → Fock → DIIS → 广义特征方程 → 密度。
//!   5) 能量 E = 0.5 tr[P(H+F)] + E_nuc。

use calchem_basis::{AngularType, BasisSetData, MoleculeCore};
use calchem_tensor::PackedSymMatrix;
use faer::Mat;

use crate::diis::Diis;
use crate::guess::sad_guess_density;

/// SCF 计算输出（闭壳层 RHF/RKS）。
#[derive(Debug, Clone)]
pub struct ScfOutput {
    /// 是否收敛。
    pub converged: bool,
    /// 实际迭代次数。
    pub iterations: usize,
    /// 总能量（含核排斥），Eh。
    pub total_energy: f64,
    /// 电子能量（不含核排斥），Eh。
    pub electronic_energy: f64,
    /// 核排斥能，Eh。
    pub nuclear_repulsion_energy: f64,
    /// 交换关联能（DFT 时存在），Eh。
    pub xc_energy: Option<f64>,
    /// 占据空间轨道数。
    pub n_occ: usize,
    /// 总电子数。
    pub n_electrons: usize,
    /// α/β 电子数。
    pub n_alpha: usize,
    pub n_beta: usize,
    /// 轨道能级（升序），Eh。
    pub mo_energies: Vec<f64>,
    /// 轨道系数，行主序 n×n（第 j 列 = 第 j 个 MO）。
    pub mo_coeff: Vec<f64>,
    /// 总密度矩阵（紧凑对称）。
    pub density: PackedSymMatrix,
    /// Fock 矩阵（紧凑对称）。
    pub fock: PackedSymMatrix,
    /// 重叠矩阵（紧凑对称）。
    pub overlap: PackedSymMatrix,
    /// 核心哈密顿量 T+V（紧凑对称）。
    pub h_core: PackedSymMatrix,
}

impl ScfOutput {
    /// 取第 j 个 MO 的系数列。
    pub fn mo_column(&self, j: usize) -> Vec<f64> {
        let n = self.overlap.dim;
        (0..n).map(|i| self.mo_coeff[i * n + j]).collect()
    }
}

/// 波函数句柄：持有分子、基组与 SCF 结果。
///
/// 供 PyO3 层在一次单点计算后继续做零拷贝 Cube、梯度、性质等，
/// 而无需重新构建分子/基组或重算。
pub struct WavefunctionHandle {
    pub mol: MoleculeCore,
    pub basis: BasisSetData,
    pub scf: ScfOutput,
}

/// 由行主序稠密矩阵构造 faer Mat。
fn mat_from_rowmajor(n: usize, d: &[f64]) -> Mat<f64> {
    Mat::from_fn(n, n, |i, j| d[i * n + j])
}

/// 把 faer Mat 转为行主序 Vec。
fn rowmajor_from_mat(n: usize, m: &Mat<f64>) -> Vec<f64> {
    (0..n).flat_map(|i| (0..n).map(move |j| m[(i, j)])).collect()
}

/// 对称矩阵特征分解，返回（升序特征值, 特征向量 U）。
fn sym_eig(a: &Mat<f64>) -> (Vec<f64>, Mat<f64>) {
    let n = a.nrows();
    let eig = faer::linalg::solvers::SelfAdjointEigendecomposition::new(a.as_ref(), faer::Side::Lower);
    let mut pairs: Vec<(f64, Vec<f64>)> = (0..n)
        .map(|j| {
            let lam = *eig.s().column_vector().get(j);
            let col: Vec<f64> = (0..n).map(|i| *eig.u().get(i, j)).collect();
            (lam, col)
        })
        .collect();
    pairs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    let evals = pairs.iter().map(|p| p.0).collect();
    let mut u = Mat::<f64>::zeros(n, n);
    for (j, (_, col)) in pairs.iter().enumerate() {
        for i in 0..n {
            u[(i, j)] = col[i];
        }
    }
    (evals, u)
}

/// 稠密矩阵乘法 C = A B。
fn matmul(a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();
    assert_eq!(k, b.nrows());
    let mut c = Mat::<f64>::zeros(m, n);
    // faer 语义：acc = beta·lhs·rhs（alpha=None 时不读旧 acc）。
    faer::linalg::matmul::matmul(
        c.as_mut(),
        a.as_ref(),
        b.as_ref(),
        None,
        1.0f64,
        faer::Parallelism::None,
    );
    c
}

/// 由 MO 系数（行主序）与每轨道占据数形成密度矩阵（紧凑对称）。
fn density_from_mo(n: usize, mo_coeff: &[f64], occ: &[f64]) -> PackedSymMatrix {
    let mut p = PackedSymMatrix::zeros(n);
    for mu in 0..n {
        for nu in 0..n {
            let mut s = 0.0;
            for i in 0..n {
                s += occ[i] * mo_coeff[mu * n + i] * mo_coeff[nu * n + i];
            }
            p.set(mu, nu, s);
        }
    }
    p
}

/// 核心 SCF 迭代（通用，占据由调用方给定，可为分数——SAD 原子用）。
#[allow(clippy::too_many_arguments)]
pub(crate) fn scf_generic(
    n: usize,
    s: &PackedSymMatrix,
    h: &PackedSymMatrix,
    eri: &[f64],
    occ: &[f64],
    guess: &PackedSymMatrix,
    max_iter: usize,
    use_diis: bool,
    use_direct: bool,
    basis: &BasisSetData,
) -> (bool, usize, f64, Vec<f64>, Vec<f64>, PackedSymMatrix, PackedSymMatrix) {
    // 正交化矩阵 X = S^-1/2，Xinv = S^1/2。
    let sd = mat_from_rowmajor(n, &s.to_dense());
    let (seval, su) = sym_eig(&sd);
    let mut x = Mat::<f64>::zeros(n, n);
    let mut xinv = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut xij = 0.0;
            let mut xijinv = 0.0;
            for k in 0..n {
                xij += su[(i, k)] * (1.0 / seval[k].sqrt()) * su[(j, k)];
                xijinv += su[(i, k)] * seval[k].sqrt() * su[(j, k)];
            }
            x[(i, j)] = xij;
            xinv[(i, j)] = xijinv;
        }
    }

    let mut diis = Diis::new(n, 8);
    let mut p: PackedSymMatrix = guess.clone();
    let mut energy = 0.0;
    let mut converged = false;
    let mut iters = 0;
    let mut mo_coeff = vec![0.0; n * n];
    let mut mo_energies = vec![0.0; n];
    let mut f_packed = PackedSymMatrix::zeros(n);

    for it in 1..=max_iter {
        iters = it;
        // 构建 J、K。
        let (j, k) = if use_direct {
            let screener = calchem_int::build_schwarz_screener(basis, 0.0);
            calchem_int::build_jk_direct_parallel(basis, &screener, &p, 1.0)
        } else {
            calchem_int::build_jk_incore(n, eri, &p, 1.0)
        };
        // F = H + J - K。
        let mut f = PackedSymMatrix::zeros(n);
        // F = H + J - 0.5 K（P 含因子 2，闭壳层交换系数 1/2）。
        for i in 0..n {
            for jj in i..n {
                f.set(i, jj, h.get(i, jj) + j.get(i, jj) - 0.5 * k.get(i, jj));
            }
        }

        // 电子能量 0.5 tr[P(H+F)]。
        let mut e_elec = 0.0;
        for i in 0..n {
            for jj in i..n {
                let w = if i == jj { 1.0 } else { 2.0 };
                e_elec += w * p.get(i, jj) * (h.get(i, jj) + f.get(i, jj));
            }
        }
        e_elec *= 0.5;

        // 转到正交基做 DIIS / 特征。
        let fd = mat_from_rowmajor(n, &f.to_dense());
        let fp = matmul(&matmul(&x, &fd), &x); // F' = X F X
        let pd = mat_from_rowmajor(n, &p.to_dense());
        let pp = matmul(&matmul(&xinv, &pd), &xinv); // P' = Xinv P Xinv

        // 始终记录当前 Fock（update 内部在历史不足 2 个时直接返回当前 Fock）。
        let fp_use = if use_diis {
            let mixed = diis.update(&rowmajor_from_mat(n, &fp), &rowmajor_from_mat(n, &pp));
            mat_from_rowmajor(n, &mixed)
        } else {
            fp
        };

        // 特征 F' C' = C' ε。
        let (evals, cprime) = sym_eig(&fp_use);
        // C = X C'。
        let cmat = matmul(&x, &cprime);
        mo_coeff = rowmajor_from_mat(n, &cmat);
        mo_energies = evals;

        // 新密度。
        let pnew = density_from_mo(n, &mo_coeff, &occ);

        // 收敛判据：能量变化 + 密度 RMS。
        let mut drms = 0.0;
        for i in 0..n {
            for jj in i..n {
                let w = if i == jj { 1.0 } else { 2.0 };
                let d = pnew.get(i, jj) - p.get(i, jj);
                drms += w * d * d;
            }
        }
        drms = drms.sqrt();
        let de = if it > 1 { (e_elec - energy).abs() } else { 1.0 };
        energy = e_elec;
        p = pnew;
        f_packed = f;

        if it >= 2 && de < 1e-10 && drms < 1e-8 {
            converged = true;
            break;
        }
    }

    (converged, iters, energy, mo_energies, mo_coeff, p, f_packed)
}

/// 对 Fock 做广义特征分解 F C = S C ε，返回（ε 升序, C 行主序）。
///
/// 用对称正交化 X = S^{-1/2}：F' = X F X 正交对角化，C = X C'。
/// 这给出"自洽 Fock"的真实轨道能量与系数（DIIS 加速用的是外推 Fock，其
/// 特征值在收敛前/快速收敛时会偏离自洽 Fock，故最终量需由本函数重算）。
fn generalized_fock_eig(
    n: usize,
    s: &PackedSymMatrix,
    f: &PackedSymMatrix,
) -> (Vec<f64>, Vec<f64>) {
    let sd = mat_from_rowmajor(n, &s.to_dense());
    let (seval, su) = sym_eig(&sd);
    let mut x = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for jj in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += su[(i, k)] * (1.0 / seval[k].sqrt()) * su[(jj, k)];
            }
            x[(i, jj)] = acc;
        }
    }
    let fd = mat_from_rowmajor(n, &f.to_dense());
    let xf = matmul(&x, &fd);
    let fprime = matmul(&xf, &x);
    let (evals, cprime) = sym_eig(&fprime);
    let cmat = matmul(&x, &cprime);
    (evals, rowmajor_from_mat(n, &cmat))
}

/// SCF 求解入口：返回完整波函数句柄（含分子、基组、SCF 结果）。
#[allow(clippy::too_many_arguments)]
pub fn execute_scf_to_handle(
    atomic_numbers: &[u8],
    coords_bohr: &[f64],
    charge: i32,
    multiplicity: usize,
    method: &str,
    basis_name: &str,
    xc: Option<&str>,
    use_ri: bool,
    max_threads: usize,
) -> Result<WavefunctionHandle, String> {
    let _ = (use_ri, max_threads);
    // 仅支持闭壳层 RHF（其余方法在后续阶段扩展）。
    if method != "rhf" && method != "rks" {
        return Err(format!("当前阶段仅实现 RHF/RKS，收到 method={}", method));
    }
    if xc.is_some() {
        return Err("DFT 泛函在 Phase 2 实现，当前阶段请用 xc=None".to_string());
    }

    let mol = MoleculeCore::new(atomic_numbers, coords_bohr, charge, multiplicity);
    let basis = calchem_basis::load_builtin_basis(basis_name, &mol, AngularType::Cartesian)?;
    let n = basis.nbf;
    let n_electrons = mol.n_electrons();
    if n_electrons % 2 != 0 {
        return Err("闭壳层 RHF 要求偶数电子（奇数电子请用 UHF，后续实现）".to_string());
    }
    let n_occ = n_electrons / 2;

    // 真实积分。
    let nuclei: Vec<([f64; 3], f64)> = mol
        .atoms
        .iter()
        .map(|a| (a.coord_bohr, a.atomic_number as f64))
        .collect();
    let s_dense = calchem_int::compute_one_electron_matrix(&basis, 0, &nuclei);
    let t_dense = calchem_int::compute_one_electron_matrix(&basis, 1, &nuclei);
    let v_dense = calchem_int::compute_one_electron_matrix(&basis, 2, &nuclei);
    let s = PackedSymMatrix::from_dense(n, &s_dense);
    let mut h = PackedSymMatrix::from_dense(n, &t_dense);
    let vh = PackedSymMatrix::from_dense(n, &v_dense);
    h.axpy(1.0, &vh);

    // 完整 ERI（in-core）。
    let eri = calchem_int::compute_eri_incore(&basis);

    // SAD 初始猜想。
    let guess = sad_guess_density(&mol, &basis, &h, &s, &eri, n_occ);

    // 占据：前 n_occ 个轨道 occ=2。
    let occ: Vec<f64> = (0..n).map(|i| if i < n_occ { 2.0 } else { 0.0 }).collect();

    // 主 SCF（in-core JK）。
    let (converged, iters, e_elec, _mo_energies, _mo_coeff, density, _fock) =
        scf_generic(n, &s, &h, &eri, &occ, &guess, 100, true, false, &basis);

    // 用最终收敛密度重建自洽 Fock，再由它取真实轨道能量与系数。
    // （DIIS 外推 Fock 的特征值在快速收敛时会偏离自洽 Fock，故必须重算。）
    let (j_f, k_f) = calchem_int::build_jk_incore(n, &eri, &density, 1.0);
    let mut f_final = PackedSymMatrix::zeros(n);
    for i in 0..n {
        for jj in i..n {
            f_final.set(i, jj, h.get(i, jj) + j_f.get(i, jj) - 0.5 * k_f.get(i, jj));
        }
    }
    let (mo_energies, mo_coeff) = generalized_fock_eig(n, &s, &f_final);

    let e_nuc = mol.nuclear_repulsion_energy();
    let e_tot = e_elec + e_nuc;

    let scf = ScfOutput {
        converged,
        iterations: iters,
        total_energy: e_tot,
        electronic_energy: e_elec,
        nuclear_repulsion_energy: e_nuc,
        xc_energy: None,
        n_occ,
        n_electrons,
        n_alpha: n_occ,
        n_beta: n_occ,
        mo_energies,
        mo_coeff,
        density,
        fock: f_final,
        overlap: s,
        h_core: h,
    };
    Ok(WavefunctionHandle { mol, basis, scf })
}

/// SCF 求解入口（仅返回 SCF 结果；保留以兼容既有调用与测试）。
#[allow(clippy::too_many_arguments)]
pub fn execute_scf_pipeline(
    atomic_numbers: &[u8],
    coords_bohr: &[f64],
    charge: i32,
    multiplicity: usize,
    method: &str,
    basis_name: &str,
    xc: Option<&str>,
    use_ri: bool,
    max_threads: usize,
) -> Result<ScfOutput, String> {
    execute_scf_to_handle(
        atomic_numbers,
        coords_bohr,
        charge,
        multiplicity,
        method,
        basis_name,
        xc,
        use_ri,
        max_threads,
    )
    .map(|h| h.scf)
}

/// 便捷：构建分子与基组（供上层/测试复用）。
pub fn build_molecule_basis(
    atomic_numbers: &[u8],
    coords_bohr: &[f64],
    charge: i32,
    multiplicity: usize,
    basis_name: &str,
    angular: AngularType,
) -> Result<(MoleculeCore, BasisSetData), String> {
    let mol = MoleculeCore::new(atomic_numbers, coords_bohr, charge, multiplicity);
    let basis = calchem_basis::load_builtin_basis(basis_name, &mol, angular)?;
    Ok((mol, basis))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_case(
        name: &str,
        z: &[u8],
        coords: &[f64],
        expected: f64,
        tol: f64,
    ) {
        let out = execute_scf_pipeline(z, coords, 0, 1, "rhf", "sto-3g", None, false, 0).unwrap();
        println!(
            "{} E={:18.12} converged={} iters={}",
            name, out.total_energy, out.converged, out.iterations
        );
        assert!(out.converged, "{} 未收敛", name);
        assert!(
            (out.total_energy - expected).abs() < tol,
            "{} E={} expected={}",
            name,
            out.total_energy,
            expected
        );
    }

    /// 调试：检查位移几何 SCF 的实际驻定性（C^T F C 非对角）。
    #[test]
    fn debug_disp_stat() {
        let coords = [0.0, 0.0, 0.0, 0.0, 1.432, 0.0, 1.112, -0.716, 0.0];
        let h = 0.001;
        let mut cp = coords; cp[0]+=h;
        let (molp,bp) = build_molecule_basis(&[8,1,1],&cp,0,1,"sto-3g",AngularType::Cartesian).unwrap();
        let np = molp.atoms.iter().map(|a|(a.coord_bohr,a.atomic_number as f64)).collect::<Vec<_>>();
        let sp = PackedSymMatrix::from_dense(7,&calchem_int::compute_one_electron_matrix(&bp,0,&np));
        let mut hp = calchem_int::compute_one_electron_matrix(&bp,1,&np);
        let vp = calchem_int::compute_one_electron_matrix(&bp,2,&np);
        for i in 0..hp.len() { hp[i]+=vp[i]; }
        let hp = PackedSymMatrix::from_dense(7,&hp);
        let erip = calchem_int::compute_eri_incore(&bp);
        let occ: Vec<f64> = (0..7).map(|i| if i<4 {2.0} else {0.0}).collect();
        let gsad = sad_guess_density(&molp,&bp,&hp,&sp,&erip,4);
        let (conv,it,e,_,cm,pp,fp) = scf_generic(7,&sp,&hp,&erip,&occ,&gsad,100,true,false,&bp);
        println!("conv={} it={} e_elec={}", conv,it,e);
        // C^T F C 非对角最大值。
        let mut maxoff: f64=0.0;
        for p in 0..7 { for q in 0..7 {
            let mut s=0.0;
            for m in 0..7 { for n in 0..7 { s+=cm[m*7+p]*fp.get(m,n)*cm[n*7+q]; } }
            if p!=q { maxoff=maxoff.max(s.abs()); }
        } }
        println!("max |C^T F C off-diag| = {}", maxoff);
        // 直接构造 F P S - S P F 范数（SCF 驻定的等价条件）。
        let mut maxcomm: f64=0.0;
        for i in 0..7 { for j in 0..7 {
            let mut fps=0.0; let mut spf=0.0;
            for k in 0..7 { for l in 0..7 {
                fps += fp.get(i,k)*pp.get(k,l)*sp.get(l,j);
                spf += sp.get(i,k)*pp.get(k,l)*fp.get(l,j);
            } }
            maxcomm=maxcomm.max((fps-spf).abs());
        } }
        println!("max |FPS - SPF| = {}", maxcomm);
    }

    /// 调试：分解 A = C^T S dC 的对称/反对称部分。
    #[test]
    fn debug_A_decomp() {
        let coords = [0.0, 0.0, 0.0, 0.0, 1.432, 0.0, 1.112, -0.716, 0.0];
        let h = 0.001;
        let (mol0,b0) = build_molecule_basis(&[8,1,1],&coords,0,1,"sto-3g",AngularType::Cartesian).unwrap();
        let n0 = mol0.atoms.iter().map(|a|(a.coord_bohr,a.atomic_number as f64)).collect::<Vec<_>>();
        let s0 = PackedSymMatrix::from_dense(7,&calchem_int::compute_one_electron_matrix(&b0,0,&n0));
        let mut h0 = calchem_int::compute_one_electron_matrix(&b0,1,&n0);
        let v0 = calchem_int::compute_one_electron_matrix(&b0,2,&n0);
        for i in 0..h0.len() { h0[i]+=v0[i]; }
        let h0 = PackedSymMatrix::from_dense(7,&h0);
        let eri0 = calchem_int::compute_eri_incore(&b0);
        let g0 = sad_guess_density(&mol0,&b0,&h0,&s0,&eri0,4);
        let occ: Vec<f64> = (0..7).map(|i| if i<4 {2.0} else {0.0}).collect();
        let (_,_,_,_,c0,p0,_) = scf_generic(7,&s0,&h0,&eri0,&occ,&g0,100,true,false,&b0);
        // 位移。
        let mut cp = coords; cp[0]+=h;
        let (molp,bp) = build_molecule_basis(&[8,1,1],&cp,0,1,"sto-3g",AngularType::Cartesian).unwrap();
        let np = molp.atoms.iter().map(|a|(a.coord_bohr,a.atomic_number as f64)).collect::<Vec<_>>();
        let sp = PackedSymMatrix::from_dense(7,&calchem_int::compute_one_electron_matrix(&bp,0,&np));
        let mut hp = calchem_int::compute_one_electron_matrix(&bp,1,&np);
        let vp = calchem_int::compute_one_electron_matrix(&bp,2,&np);
        for i in 0..hp.len() { hp[i]+=vp[i]; }
        let hp = PackedSymMatrix::from_dense(7,&hp);
        let erip = calchem_int::compute_eri_incore(&bp);
        let (_,_,_,_,cp_,pp,_) = scf_generic(7,&sp,&hp,&erip,&occ,&p0,100,true,false,&bp);
        // 位移 MO 系数 cp_，参考 c0。计算 A = C0^T S0 (Cp - C0) （近似 dC）。
        // 注意：Cp 是位移几何的 MO，与 C0 基不同。用 C0^T S0 dC。
        let mut A = vec![vec![0.0;7];7];
        for p in 0..7 { for q in 0..7 {
            let mut s = 0.0;
            for m in 0..7 { for n in 0..7 {
                let dc = cp_[n*7+q]-c0[n*7+q];
                s += c0[m*7+p]*s0.get(m,n)*dc;
            } }
            A[p][q]=s/h;
        } }
        // C^T dS C。
        let mut dS = vec![vec![0.0;7];7];
        for p in 0..7 { for q in 0..7 {
            let mut s = 0.0;
            for m in 0..7 { for n in 0..7 {
                s += c0[m*7+p]*(sp.get(m,n)-s0.get(m,n))/h*c0[n*7+q];
            } }
            dS[p][q]=s;
        } }
        // 检查 A + A^T = -dS。
        let mut maxres: f64 = 0.0;
        for p in 0..7 { for q in 0..7 { maxres=maxres.max((A[p][q]+A[q][p]+dS[p][q]).abs()); } }
        println!("|A+A^T+dS| max = {}", maxres);
        // 对称/反对称密度项。
        let mut eps = vec![0.0;7];
        for p in 0..7 {
            let mut e=0.0;
            for m in 0..7 { for n in 0..7 { e+=c0[m*7+p]*h0.get(m,n)*c0[n*7+p]; } }
            // 用 Fock 更准确：这里用 h0 仅占位，下面单独算。
            eps[p]=e;
        }
        // 用 Fock（含 J/K）。
        let (j0,k0) = calchem_int::build_jk_incore(7,&eri0,&p0,1.0);
        for p in 0..7 {
            let mut e=0.0;
            for m in 0..7 { for n in 0..7 {
                let f = h0.get(m,n)+j0.get(m,n)-0.5*k0.get(m,n);
                e+=c0[m*7+p]*f*c0[n*7+p];
            } }
            eps[p]=e;
        }
        let mut tsym=0.0; let mut tanti=0.0;
        for p in 0..7 { for q in 0..7 {
            let asym = -0.5*dS[p][q];
            let aanti = A[p][q]-asym;
            tsym += 2.0*occ[p]*eps[p]*asym;
            tanti += 2.0*occ[p]*eps[p]*aanti;
        } }
        println!("density term: sym={} anti={} total={}", tsym, tanti, tsym+tanti);
    }

    /// 调试：直接计算 tr[F dP] 与能量加权密度项。
    #[test]
    fn debug_trfdp() {
        let coords = [0.0, 0.0, 0.0, 0.0, 1.432, 0.0, 1.112, -0.716, 0.0];
        let h = 0.001;
        let (mol0,b0) = build_molecule_basis(&[8,1,1],&coords,0,1,"sto-3g",AngularType::Cartesian).unwrap();
        let n0 = mol0.atoms.iter().map(|a|(a.coord_bohr,a.atomic_number as f64)).collect::<Vec<_>>();
        let s0 = PackedSymMatrix::from_dense(7,&calchem_int::compute_one_electron_matrix(&b0,0,&n0));
        let mut h0 = calchem_int::compute_one_electron_matrix(&b0,1,&n0);
        let v0 = calchem_int::compute_one_electron_matrix(&b0,2,&n0);
        for i in 0..h0.len() { h0[i]+=v0[i]; }
        let h0 = PackedSymMatrix::from_dense(7,&h0);
        let eri0 = calchem_int::compute_eri_incore(&b0);
        let g0 = sad_guess_density(&mol0,&b0,&h0,&s0,&eri0,4);
        let occ: Vec<f64> = (0..7).map(|i| if i<4 {2.0} else {0.0}).collect();
        let (_,_,_,_,c0,p0,f0) = scf_generic(7,&s0,&h0,&eri0,&occ,&g0,100,true,false,&b0);
        // 位移。
        let mut cp = coords; cp[0]+=h;
        let (molp,bp) = build_molecule_basis(&[8,1,1],&cp,0,1,"sto-3g",AngularType::Cartesian).unwrap();
        let np = molp.atoms.iter().map(|a|(a.coord_bohr,a.atomic_number as f64)).collect::<Vec<_>>();
        let sp = PackedSymMatrix::from_dense(7,&calchem_int::compute_one_electron_matrix(&bp,0,&np));
        let mut hp = calchem_int::compute_one_electron_matrix(&bp,1,&np);
        let vp = calchem_int::compute_one_electron_matrix(&bp,2,&np);
        for i in 0..hp.len() { hp[i]+=vp[i]; }
        let hp = PackedSymMatrix::from_dense(7,&hp);
        let erip = calchem_int::compute_eri_incore(&bp);
        let (_,_,_,_,_,pp,_) = scf_generic(7,&sp,&hp,&erip,&occ,&p0,100,true,false,&bp);
        // tr[F0 (Pp - P0)]。
        let mut trfdp = 0.0;
        for i in 0..7 { for j in 0..7 { trfdp += f0.get(i,j)*(pp.get(i,j)-p0.get(i,j)); } }
        println!("tr[F dP] = {} (per h: {})", trfdp, trfdp/h);
        // C^T dS C 对角（用 MO 系数 c0，行主序）。
        // dS = (Sp - S0)/h。
        let mut ctsd = vec![0.0;7];
        for p in 0..7 {
            let mut s = 0.0;
            for m in 0..7 { for n in 0..7 {
                let dsmn = (sp.get(m,n)-s0.get(m,n))/h;
                s += c0[m*7+p]*dsmn*c0[n*7+p];
            } }
            ctsd[p]=s;
        }
        let mut ew = 0.0;
        for p in 0..7 { ew += occ[p] * (0..7).map(|_| {
            // 需要 MO 能量；用 F0 在 MO 基的对角近似。
            let mut eps = 0.0;
            for m in 0..7 { for n in 0..7 { eps += c0[m*7+p]*f0.get(m,n)*c0[n*7+p]; } }
            eps
        }).next().unwrap() * ctsd[p]; }
        println!("-Σ occ ε (C^T dS C) = {}", -ew);
    }

    /// 调试：位移几何下，SAD 起点 vs 参考密度起点（SCF 应找到同一最低解）。
    #[test]
    fn debug_displaced_start() {
        let coords = [0.0, 0.0, 0.0, 0.0, 1.432, 0.0, 1.112, -0.716, 0.0];
        let h = 0.001;
        // 参考。
        let (mol0, b0) = build_molecule_basis(&[8,1,1],&coords,0,1,"sto-3g",AngularType::Cartesian).unwrap();
        let eri0 = calchem_int::compute_eri_incore(&b0);
        let s0 = calchem_int::compute_one_electron_matrix(&b0,0,&mol0.atoms.iter().map(|a|(a.coord_bohr,a.atomic_number as f64)).collect::<Vec<_>>());
        let s0 = PackedSymMatrix::from_dense(7,&s0);
        let guess0 = sad_guess_density(&mol0,&b0,&PackedSymMatrix::zeros(7),&s0,&eri0,4);
        let occ: Vec<f64> = (0..7).map(|i| if i<4 {2.0} else {0.0}).collect();
        let (_,_,e0,_,_,p0,_) = scf_generic(7,&s0,&{
            let mut hh = calchem_int::compute_one_electron_matrix(&b0,1,&mol0.atoms.iter().map(|a|(a.coord_bohr,a.atomic_number as f64)).collect::<Vec<_>>());
            let vv = calchem_int::compute_one_electron_matrix(&b0,2,&mol0.atoms.iter().map(|a|(a.coord_bohr,a.atomic_number as f64)).collect::<Vec<_>>());
            for i in 0..hh.len() { hh[i]+=vv[i]; }
            PackedSymMatrix::from_dense(7,&hh)
        },&eri0,&occ,&guess0,100,true,false,&b0);
        println!("ref E={}", e0);
        // 位移几何。
        let mut cp = coords; cp[0]+=h;
        let (molp,bp) = build_molecule_basis(&[8,1,1],&cp,0,1,"sto-3g",AngularType::Cartesian).unwrap();
        let nucp = molp.atoms.iter().map(|a|(a.coord_bohr,a.atomic_number as f64)).collect::<Vec<_>>();
        let mut hp = calchem_int::compute_one_electron_matrix(&bp,1,&nucp);
        let vp = calchem_int::compute_one_electron_matrix(&bp,2,&nucp);
        for i in 0..hp.len() { hp[i]+=vp[i]; }
        let hp = PackedSymMatrix::from_dense(7,&hp);
        let sp = calchem_int::compute_one_electron_matrix(&bp,0,&nucp);
        let sp = PackedSymMatrix::from_dense(7,&sp);
        let erip = calchem_int::compute_eri_incore(&bp);
        // (a) SAD 起点。
        let gsad = sad_guess_density(&molp,&bp,&hp,&sp,&erip,4);
        let (_,_,esa,_,_,psad,_) = scf_generic(7,&sp,&hp,&erip,&occ,&gsad,100,true,false,&bp);
        // (b) 参考密度起点。
        let (_,_,ep0,_,_,pp0,_) = scf_generic(7,&sp,&hp,&erip,&occ,&p0,100,true,false,&bp);
        // (c) 参考密度起点，无 DIIS。
        let (_,_,ep0nd,_,_,_,_) = scf_generic(7,&sp,&hp,&erip,&occ,&p0,100,false,false,&bp);
        println!("displaced: SAD-start E={} ref-start E={} ref-start-noDIIS E={}", esa, ep0, ep0nd);
        let mut dmax: f64 = 0.0;
        for i in 0..7 { for j in 0..7 { dmax=dmax.max((psad.get(i,j)-pp0.get(i,j)).abs()); } }
        println!("density diff SAD vs refstart max={}", dmax);
    }

    #[test]
    fn dump_h2o_dm() {
        let coords = [0.0, 0.0, 0.0, 0.0, 1.432, 0.0, 1.112, -0.716, 0.0];
        let out = execute_scf_pipeline(&[8, 1, 1], &coords, 0, 1, "rhf", "sto-3g", None, false, 0)
            .unwrap();
        let mut s = String::new();
        for i in 0..7 {
            for j in 0..7 {
                s.push_str(&format!("{}\n", out.density.get(i, j)));
            }
        }
        std::fs::write(r"E:\YDW\Trae_Gaussian\_build_temp\rust_h2o_dm.txt", s).unwrap();
    }

    #[test]
    fn dump_ne_eri() {
        let mol = MoleculeCore::new(&[10], &[0.0, 0.0, 0.0], 0, 1);
        let b = calchem_basis::load_builtin_basis("sto-3g", &mol, AngularType::Cartesian).unwrap();
        let eri = calchem_int::compute_eri_incore(&b);
        let mut s = String::new();
        for i in 0..5 { for j in 0..5 { for k in 0..5 { for l in 0..5 {
            let v = eri[((i * 5 + j) * 5 + k) * 5 + l];
            s.push_str(&format!("{} {} {} {} {}\n", i, j, k, l, v));
        } } } }
        std::fs::write(r"E:\YDW\Trae_Gaussian\_build_temp\rust_ne_eri.txt", s).unwrap();
    }

    #[test]
    fn debug_ne() {
        let out = execute_scf_pipeline(&[10], &[0.0, 0.0, 0.0], 0, 1, "rhf", "sto-3g", None, false, 0)
            .unwrap();
        for i in 0..5 {
            let row: Vec<f64> = (0..5).map(|j| out.density.get(i, j)).collect();
            println!("P row{}: {:?}", i, row);
        }
        let mol = MoleculeCore::new(&[10], &[0.0, 0.0, 0.0], 0, 1);
        let b = calchem_basis::load_builtin_basis("sto-3g", &mol, AngularType::Cartesian).unwrap();
        let eri = calchem_int::compute_eri_incore(&b);
        let (j, k) = calchem_int::build_jk_incore(5, &eri, &out.density, 1.0);
        for i in 0..5 {
            println!("H{}={} J{}={} K{}={} F{}={}", i, out.h_core.get(i, i), i, j.get(i, i), i, k.get(i, i), i, out.fock.get(i, i));
        }
        let ev: Vec<f64> = (0..5).map(|i| out.mo_energies[i]).collect();
        println!("ev: {:?}", ev);
    }

    #[test]
    fn debug_h2_full() {
        let out = execute_scf_pipeline(
            &[1, 1],
            &[0.0, 0.0, 0.0, 0.0, 0.0, 1.4],
            0, 1, "rhf", "sto-3g", None, false, 0,
        )
        .unwrap();
        let n = 2;
        let mol = MoleculeCore::new(&[1, 1], &[0.0, 0.0, 0.0, 0.0, 0.0, 1.4], 0, 1);
        let out_basis = calchem_basis::load_builtin_basis("sto-3g", &mol, AngularType::Cartesian).unwrap();
        println!("P00={} P01={} P11={}", out.density.get(0, 0), out.density.get(0, 1), out.density.get(1, 1));
        println!("F00={} F01={} F11={}", out.fock.get(0, 0), out.fock.get(0, 1), out.fock.get(1, 1));
        println!("H00={} H01={} H11={}", out.h_core.get(0, 0), out.h_core.get(0, 1), out.h_core.get(1, 1));
        println!("mo={:?} ev={:?}", out.mo_coeff, out.mo_energies);
        let eri = calchem_int::compute_eri_incore(&out_basis);
        let (j, k) = calchem_int::build_jk_incore(n, &eri, &out.density, 1.0);
        println!("J00={} J01={} J11={}", j.get(0, 0), j.get(0, 1), j.get(1, 1));
        println!("K00={} K01={} K11={}", k.get(0, 0), k.get(0, 1), k.get(1, 1));
        let _ = n;
    }

    #[test]
    fn check_matmul() {
        // 2x2 简单乘法。
        let a = mat_from_rowmajor(2, &[1.0, 2.0, 3.0, 4.0]);
        let b = mat_from_rowmajor(2, &[5.0, 6.0, 7.0, 8.0]);
        println!("a00={} a01={} a10={} a11={}", a[(0,0)], a[(0,1)], a[(1,0)], a[(1,1)]);
        println!("b00={} b01={} b10={} b11={}", b[(0,0)], b[(0,1)], b[(1,0)], b[(1,1)]);
        let c = matmul(&a, &b);
        println!("c00={} c01={} c10={} c11={}", c[(0, 0)], c[(0, 1)], c[(1, 0)], c[(1, 1)]);
        // 期望 [[19,22],[43,50]]
        assert!((c[(0, 0)] - 19.0).abs() < 1e-9);
        assert!((c[(0, 1)] - 22.0).abs() < 1e-9);
        assert!((c[(1, 0)] - 43.0).abs() < 1e-9);
        assert!((c[(1, 1)] - 50.0).abs() < 1e-9);
    }

    #[test]
    fn debug_h2() {
        let z = [1u8, 1];
        let coords = [0.0, 0.0, 0.0, 0.0, 0.0, 1.4];
        let mol = MoleculeCore::new(&z, &coords, 0, 1);
        let basis = calchem_basis::load_builtin_basis("sto-3g", &mol, AngularType::Cartesian).unwrap();
        let n = basis.nbf;
        let nuclei: Vec<([f64; 3], f64)> =
            mol.atoms.iter().map(|a| (a.coord_bohr, a.atomic_number as f64)).collect();
        let sd = calchem_int::compute_one_electron_matrix(&basis, 0, &nuclei);
        let td = calchem_int::compute_one_electron_matrix(&basis, 1, &nuclei);
        let vd = calchem_int::compute_one_electron_matrix(&basis, 2, &nuclei);
        let s = PackedSymMatrix::from_dense(n, &sd);
        let mut h = PackedSymMatrix::from_dense(n, &td);
        h.axpy(1.0, &PackedSymMatrix::from_dense(n, &vd));
        let eri = calchem_int::compute_eri_incore(&basis);
        let g = crate::guess::sad_guess_density(&mol, &basis, &h, &s, &eri, 1);
        println!("H00={} H01={} H11={}", h.get(0, 0), h.get(0, 1), h.get(1, 1));
        println!("GUESS P00={} P01={} P11={}", g.get(0, 0), g.get(0, 1), g.get(1, 1));
        let (j, k) = calchem_int::build_jk_incore(n, &eri, &g, 1.0);
        println!("J00={} J01={} J11={}", j.get(0, 0), j.get(0, 1), j.get(1, 1));
        println!("K00={} K01={} K11={}", k.get(0, 0), k.get(0, 1), k.get(1, 1));
    }

    #[test]
    fn baseline_h2() {
        run_case(
            "H2",
            &[1, 1],
            &[0.0, 0.0, 0.0, 0.0, 0.0, 1.4],
            -1.1167143251,
            1e-8,
        );
    }

    #[test]
    fn baseline_he() {
        run_case("He", &[2], &[0.0, 0.0, 0.0], -2.8077839575, 1e-6);
    }

    #[test]
    fn baseline_ne() {
        run_case("Ne", &[10], &[0.0, 0.0, 0.0], -126.6045291715, 1e-6);
    }

    #[test]
    fn baseline_h2o() {
        run_case(
            "H2O",
            &[8, 1, 1],
            &[
                0.0, 0.0, 0.0,
                0.0, 1.432, 0.0,
                1.112, -0.716, 0.0,
            ],
            // 注意：BASELINE 的 -74.6953191771 用的是参照实现的幂级数 Boys，
            // 该 Boys 在 T≈26 有约 3.8e-8 相消误差。本引擎 F0 始终由 erf 精确给出，
            // 更精确的真值为 -74.695319114（已用 Python 替换为 erf-Boys 后复现）。
            -74.695319114,
            1e-7,
        );
    }
}
