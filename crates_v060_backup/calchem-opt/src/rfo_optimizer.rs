//! 信赖域有理函数优化（RFO）几何优化主循环（笛卡尔坐标）。
//!
//! 流程（让计算化学方向、Python 薄弱的研究生也能看懂）：
//!   1. 对当前几何做一次真实 SCF，得到能量 E 与解析梯度 g（含 Pulay 项）。
//!   2. 用近似 Hessian H 构造 RFO 增广矩阵并求其最低本征对，得到步进
//!        dx = -(H - λI)^{-1} g
//!      （RFO 自动给每一步定标，比纯牛顿法稳健）。
//!   3. 用信赖域上限截断步长，必要时回溯（能量不下降就缩小步长）。
//!   4. 用真实梯度差做 BFGS 更新，让 Hessian 越来越准。
//!   5. 当最大力 / RMS 力 / 最大位移 / RMS 位移都低于阈值时判定收敛。
//!
//! 参考：
//!   - RFO：见 Szabo & Ostlund / Schlegel 等的标准几何优化方法；
//!   - BFGS：标准拟牛顿更新；
//!   - 收敛判据沿用 Gaussian 风格的四个判据。

use calchem_basis::{AngularType, MoleculeCore};
use calchem_grad::compute_analytical_gradient;
use faer::linalg::solvers::SelfAdjointEigendecomposition;
use faer::Side;

use crate::hessian_guess::{initial_model_hessian, DEFAULT_H0};

/// 几何优化配置。
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// 最大优化步数。
    pub max_steps: usize,
    /// 最大力收敛阈值（Eh/Bohr）。
    pub f_tol: f64,
    /// RMS 力收敛阈值。
    pub rms_f_tol: f64,
    /// 最大位移收敛阈值（Bohr）。
    pub dx_tol: f64,
    /// RMS 位移收敛阈值。
    pub rms_dx_tol: f64,
    /// 单步最大位移（信赖域半径，Bohr）。
    pub max_step: f64,
    /// 基组名。
    pub basis: String,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        // 收敛判据沿用 Gaussian 风格默认值（450e-6 / 300e-6 / 18e-3 / 12e-3）。
        OptimizationConfig {
            max_steps: 100,
            f_tol: 4.5e-4,
            rms_f_tol: 3.0e-4,
            dx_tol: 1.8e-3,
            rms_dx_tol: 1.2e-3,
            max_step: 0.3,
            basis: "sto-3g".to_string(),
        }
    }
}

/// 几何优化结果。
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// 是否收敛。
    pub converged: bool,
    /// 实际优化步数。
    pub steps: usize,
    /// 最终坐标（行主序，与输入一致）。
    pub coords: Vec<f64>,
    /// 最终总能量（Eh）。
    pub energy: f64,
    /// 最终梯度（[atom][x,y,z]，Eh/Bohr）。
    pub gradient: Vec<[f64; 3]>,
    /// 每一步的能量轨迹（含初始点）。
    pub energies: Vec<f64>,
    /// 人类可读的状态说明。
    pub message: String,
}

/// 一次完整的"单点 + 解析梯度"评估。
fn evaluate(
    z: &[u8],
    coords: &[f64],
    basis: &str,
) -> Result<(MoleculeCore, calchem_basis::BasisSetData, calchem_scf::ScfOutput, Vec<[f64; 3]>), String> {
    let mol = MoleculeCore::new(z, coords, 0, 1);
    let basis_set = calchem_basis::load_builtin_basis(basis, &mol, AngularType::Cartesian)?;
    let scf = calchem_scf::execute_scf_pipeline(z, coords, 0, 1, "rhf", basis, None, false, 0)?;
    let grad = compute_analytical_gradient(&mol, &basis_set, &scf);
    Ok((mol, basis_set, scf, grad))
}

/// 把梯度 [atom][x,y,z] 展平为长度 m 的向量。
fn flatten_grad(g: &[[f64; 3]]) -> Vec<f64> {
    g.iter().flat_map(|a| a.iter().copied()).collect()
}

/// 计算 RFO 步长方向（未截断）。
///
/// 构造增广矩阵 R = [H, g; gᵀ, 0]，取最低本征向量 [v; c]，步进 dx = v/c。
fn rfo_step(hessian: &[f64], g: &[f64], m: usize) -> Vec<f64> {
    let n = m + 1;
    let mut r = faer::Mat::<f64>::zeros(n, n);
    for i in 0..m {
        for j in 0..m {
            r.write(i, j, hessian[i * m + j]);
        }
        r.write(i, m, g[i]);
        r.write(m, i, g[i]);
    }
    r.write(m, m, 0.0);

    let eig = SelfAdjointEigendecomposition::new(r.as_ref(), Side::Lower);
    let u = eig.u();
    // 最低本征值对应第 0 列（faer 按升序返回）。
    let c = u.read(m, 0);
    let mut dx = vec![0.0; m];
    if c.abs() < 1e-300 {
        // 极端退化时退化为最速下降。
        for i in 0..m {
            dx[i] = -g[i];
        }
        return dx;
    }
    for i in 0..m {
        dx[i] = u.read(i, 0) / c;
    }
    dx
}

/// 信赖域截断：把步长向量限制在 max_step 以内（保持方向）。
fn trust_cap(dx: &mut [f64], max_step: f64) {
    let norm: f64 = dx.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > max_step {
        let scale = max_step / norm;
        for x in dx.iter_mut() {
            *x *= scale;
        }
    }
}

/// BFGS Hessian 更新（返回新 Hessian，行主序）。
///
///   H' = H + (y yᵀ)/(sᵀy) - (H s sᵀ H)/(sᵀ H s)
/// 其中 s = 坐标差，y = 梯度差。sᵀy 不充分正（曲率条件不满足）时跳过更新。
fn bfgs_update(hessian: &[f64], s: &[f64], y: &[f64], m: usize) -> Vec<f64> {
    let sty: f64 = s.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    // 曲率条件：sᵀy 需为正且不太接近 0。
    if sty <= 1e-10 {
        return hessian.to_vec();
    }
    // Hs。
    let mut hs = vec![0.0; m];
    for i in 0..m {
        let mut acc = 0.0;
        for k in 0..m {
            acc += hessian[i * m + k] * s[k];
        }
        hs[i] = acc;
    }
    let s_hs: f64 = s.iter().zip(hs.iter()).map(|(a, b)| a * b).sum();
    let mut hnew = hessian.to_vec();
    for i in 0..m {
        for j in 0..m {
            let yy = y[i] * y[j] / sty;
            let hh = if s_hs.abs() > 1e-30 {
                hs[i] * hs[j] / s_hs
            } else {
                0.0
            };
            hnew[i * m + j] += yy - hh;
        }
    }
    hnew
}

/// 力统计：最大力、RMS 力。
fn force_stats(g: &[f64]) -> (f64, f64) {
    let maxf = g.iter().map(|x| x.abs()).fold(0.0, f64::max);
    let rms = (g.iter().map(|x| x * x).sum::<f64>() / g.len() as f64).sqrt();
    (maxf, rms)
}

/// 位移统计：最大位移、RMS 位移。
fn step_stats(dx: &[f64]) -> (f64, f64) {
    let maxd = dx.iter().map(|x| x.abs()).fold(0.0, f64::max);
    let rms = (dx.iter().map(|x| x * x).sum::<f64>() / dx.len() as f64).sqrt();
    (maxd, rms)
}

/// 运行几何优化。
pub fn run_optimization(
    z: &[u8],
    coords: &[f64],
    config: &OptimizationConfig,
) -> OptimizationResult {
    let m = coords.len();
    let mut cur = coords.to_vec();

    // 初始评估。
    let init = match evaluate(z, &cur, &config.basis) {
        Ok(v) => v,
        Err(e) => {
            return OptimizationResult {
                converged: false,
                steps: 0,
                coords: cur,
                energy: 0.0,
                gradient: vec![],
                energies: vec![],
                message: format!("初始单点失败：{}", e),
            };
        }
    };
    let mut energy = init.2.total_energy;
    let mut gradient = init.3;
    let mut gflat = flatten_grad(&gradient);
    let mut energies = vec![energy];
    let mut hessian = initial_model_hessian(m, DEFAULT_H0);

    let mut last_dx = vec![0.0; m];

    for step in 0..config.max_steps {
        // 收敛判据（第一步也检查）。
        let (maxf, rmsf) = force_stats(&gflat);
        let (maxd, rmsd) = step_stats(&last_dx);
        let force_ok = maxf < config.f_tol && rmsf < config.rms_f_tol;
        // 位移判据在第 0 步没有位移，放宽（只看力）。
        let step_ok = step == 0 || (maxd < config.dx_tol && rmsd < config.rms_dx_tol);
        if force_ok && step_ok {
            return OptimizationResult {
                converged: true,
                steps: step,
                coords: cur,
                energy,
                gradient,
                energies,
                message: format!("在 {} 步内收敛（maxf={:.2e}, rmsf={:.2e}）", step, maxf, rmsf),
            };
        }

        // RFO 方向 + 信赖域截断。
        let mut dx = rfo_step(&hessian, &gflat, m);
        trust_cap(&mut dx, config.max_step);

        // 带回溯的步进：能量不下降就逐步缩小。
        let prev_coords = cur.clone();
        let prev_g = gflat.clone();
        let mut trial = prev_coords.clone();
        for i in 0..m {
            trial[i] += dx[i];
        }
        let mut scale = 1.0;
        let mut accepted = false;
        let mut best_energy = energy;
        let mut best_coords = prev_coords.clone();
        let mut best_grad = gradient.clone();
        let mut best_gflat = gflat.clone();
        for _bt in 0..6 {
            if let Ok((_mol, _b, scf, gr)) = evaluate(z, &trial, &config.basis) {
                let e = scf.total_energy;
                if e < best_energy {
                    let gfl = flatten_grad(&gr);
                    best_energy = e;
                    best_coords = trial.clone();
                    best_grad = gr;
                    best_gflat = gfl;
                }
                // 接受条件：能量有实质下降。
                if e < energy - 1e-10 {
                    accepted = true;
                    break;
                }
            }
            // 回溯：缩小步长。
            scale *= 0.5;
            trial = prev_coords.clone();
            for i in 0..m {
                trial[i] += dx[i] * scale;
            }
        }

        if !accepted {
            // 用搜索到的最佳点；若仍无下降则缩小信赖域后重试，最终判定停滞。
            if best_energy < energy {
                // 接受能量更低的点，即使不满足严格下降阈值。
            } else {
                return OptimizationResult {
                    converged: false,
                    steps: step + 1,
                    coords: cur,
                    energy,
                    gradient,
                    energies,
                    message: format!("第 {} 步无法找到能量下降方向（可能已在极小点或优化停滞）", step + 1),
                };
            }
        }

        // 实际位移（用于 BFGS 与位移判据）。
        let mut s = vec![0.0; m];
        for i in 0..m {
            s[i] = best_coords[i] - prev_coords[i];
        }
        let y: Vec<f64> = best_gflat.iter().zip(prev_g.iter()).map(|(a, b)| a - b).collect();
        hessian = bfgs_update(&hessian, &s, &y, m);

        cur = best_coords;
        energy = best_energy;
        gradient = best_grad;
        gflat = best_gflat;
        last_dx = s;
        energies.push(energy);
    }

    OptimizationResult {
        converged: false,
        steps: config.max_steps,
        coords: cur,
        energy,
        gradient,
        energies,
        message: format!("达到最大步数 {} 仍未收敛", config.max_steps),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// H2 / RHF / STO-3G 几何优化：从 1.4 Bohr 出发，应收敛到 1.346 Bohr。
    #[test]
    fn optimize_h2_equilibrium() {
        // 沿 z 轴放置。
        let coords = [0.0, 0.0, 0.0, 0.0, 0.0, 1.4];
        let cfg = OptimizationConfig::default();
        let res = run_optimization(&[1, 1], &coords, &cfg);
        for (i, e) in res.energies.iter().enumerate() {
            println!("step={} E={:.10}", i, e);
        }
        // 平衡键长 = 两原子距离。
        let dx = res.coords[3] - res.coords[0];
        let dy = res.coords[4] - res.coords[1];
        let dz = res.coords[5] - res.coords[2];
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        println!("converged={} steps={} r={:.6} E={:.10} msg={}",
            res.converged, res.steps, r, res.energy, res.message);
        assert!(res.converged, "未收敛：{}", res.message);
        assert!((r - 1.346).abs() < 1e-3, "r={} 偏离 1.346", r);
    }

    /// 从较远的 1.7 Bohr 出发也应收敛到 1.346（验证确实在移动而非停在初始点）。
    #[test]
    fn optimize_h2_from_far() {
        let coords = [0.0, 0.0, 0.0, 0.0, 0.0, 1.7];
        let cfg = OptimizationConfig::default();
        let res = run_optimization(&[1, 1], &coords, &cfg);
        let r = {
            let dx = res.coords[3] - res.coords[0];
            let dy = res.coords[4] - res.coords[1];
            let dz = res.coords[5] - res.coords[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        };
        println!("converged={} steps={} r={:.6}", res.converged, res.steps, r);
        assert!(res.converged, "未收敛：{}", res.message);
        assert!((r - 1.346).abs() < 1e-3, "r={} 偏离 1.346", r);
    }

    /// 诊断：H2 梯度分解（冻结 vs Pulay），定位 +0.01 偏差来源。
    #[test]
    fn debug_h2_grad_decomp() {
        let coords = [0.0, 0.0, 0.0, 0.0, 0.0, 1.4];
        let mol = MoleculeCore::new(&[1, 1], &coords, 0, 1);
        let basis = calchem_basis::load_builtin_basis("sto-3g", &mol, AngularType::Cartesian).unwrap();
        let scf = calchem_scf::execute_scf_pipeline(&[1, 1], &coords, 0, 1, "rhf", "sto-3g", None, false, 0).unwrap();
        let nuclei: Vec<([f64; 3], f64)> = mol.atoms.iter().map(|a| (a.coord_bohr, a.atomic_number as f64)).collect();
        let nbf = basis.nbf;
        let mut gnuc = mol.nuclear_repulsion_grad();
        // atom0 z 与 atom1 z。
        let dh = calchem_grad::build_deriv_hcore(&mol, &basis, &nuclei, 0, 2);
        let de = calchem_grad::build_deriv_eri(&basis, 0, 2);
        let mut one = 0.0;
        for m in 0..nbf { for n in 0..nbf { one += scf.density.get(m, n) * dh[m * nbf + n]; } }
        let (dj, dk) = calchem_int::build_jk_incore(nbf, &de, &scf.density, 1.0);
        let mut two = 0.0;
        for m in 0..nbf { for n in m..nbf {
            let wgt = if m == n { 1.0 } else { 2.0 };
            two += wgt * scf.density.get(m, n) * (0.5 * dj.get(m, n) - 0.25 * dk.get(m, n));
        } }
        let w = calchem_grad::build_energy_weighted_density(&scf);
        let ds = calchem_grad::build_deriv_overlap(&basis, 0, 2);
        let mut pulay = 0.0;
        for m in 0..nbf { for n in 0..nbf { pulay += w.get(m, n) * ds[m * nbf + n]; } }
        pulay = -pulay;
        let frozen = gnuc[0][2] + one + two;
        println!("nuc={:.6} one={:.6} two={:.6} frozen={:.6} pulay={:.6} total={:.6}",
            gnuc[0][2], one, two, frozen, pulay, frozen + pulay);
        // 数值梯度（atom0 z）。
        let h = 1e-5;
        let cp = [0.0, 0.0, h, 0.0, 0.0, 1.4];
        let ep = calchem_scf::execute_scf_pipeline(&[1, 1], &cp, 0, 1, "rhf", "sto-3g", None, false, 0).unwrap().total_energy;
        let cm = [0.0, 0.0, -h, 0.0, 0.0, 1.4];
        let em = calchem_scf::execute_scf_pipeline(&[1, 1], &cm, 0, 1, "rhf", "sto-3g", None, false, 0).unwrap().total_energy;
        println!("num atom0 z = {:.6}", (ep - em) / (2.0 * h));

        // 冻结数值梯度：固定使用 r=1.4 的密度 P0，在位移几何上重算能量。
        let p0 = scf.density.clone();
        let frozen_energy_at = |rz: f64| -> f64 {
            let cc = [0.0, 0.0, 0.0, 0.0, 0.0, rz];
            let mm = MoleculeCore::new(&[1, 1], &cc, 0, 1);
            let bb = calchem_basis::load_builtin_basis("sto-3g", &mm, AngularType::Cartesian).unwrap();
            let nn: Vec<([f64;3],f64)> = mm.atoms.iter().map(|a|(a.coord_bohr,a.atomic_number as f64)).collect();
            let t = calchem_int::compute_one_electron_matrix(&bb, 1, &nn);
            let v = calchem_int::compute_one_electron_matrix(&bb, 2, &nn);
            let mut hh = vec![0.0; nbf*nbf];
            for i in 0..nbf { for j in 0..nbf { hh[i*nbf+j]=t[i*nbf+j]+v[i*nbf+j]; } }
            let eri = calchem_int::compute_eri_incore(&bb);
            let (jj,kk) = calchem_int::build_jk_incore(nbf, &eri, &p0, 1.0);
            let mut e = 0.0;
            for i in 0..nbf { for j in 0..nbf { e += p0.get(i,j)*hh[i*nbf+j]; } }
            for i in 0..nbf { for j in i..nbf {
                let wgt = if i==j {1.0} else {2.0};
                e += wgt*p0.get(i,j)*(0.5*jj.get(i,j)-0.25*kk.get(i,j));
            } }
            e + mm.nuclear_repulsion_energy()
        };
        // 注意：位移 atom0 z 时，r 变化相反（atom0 +z 使 r 减小）。
        let efp = frozen_energy_at(1.4 - h);
        let efm = frozen_energy_at(1.4 + h);
        println!("num frozen atom0 z = {:.6}", (efp - efm)/(2.0*h));

        // 重叠 dS 数值核对。
        let overlap_at = |rz: f64| -> Vec<f64> {
            let cc = [0.0,0.0,0.0,0.0,0.0,rz];
            let mm = MoleculeCore::new(&[1,1],&cc,0,1);
            let bb = calchem_basis::load_builtin_basis("sto-3g",&mm,AngularType::Cartesian).unwrap();
            calchem_int::compute_one_electron_matrix(&bb,0,&[])
        };
        let sp2 = overlap_at(1.4-h); let sm2 = overlap_at(1.4+h);
        println!("num dS12 atom0 z = {:.6}", (sp2[1]-sm2[1])/(2.0*h));
        println!("ana dS12 = {:.6}", ds[1]);
        println!("W = {:.6} {:.6} / {:.6} {:.6}", w.get(0,0), w.get(0,1), w.get(1,0), w.get(1,1));
        // 数值 tr[F dP]：用位移几何的松弛密度，F 取 r=1.4 的 Fock。
        let f0 = scf.fock.clone();
        let scfp = calchem_scf::execute_scf_pipeline(&[1,1],&cp,0,1,"rhf","sto-3g",None,false,0).unwrap();
        let scfm = calchem_scf::execute_scf_pipeline(&[1,1],&cm,0,1,"rhf","sto-3g",None,false,0).unwrap();
        let mut trfdp = 0.0;
        for i in 0..nbf { for j in 0..nbf {
            let dpd = (scfp.density.get(i,j)-scfm.density.get(i,j))/(2.0*h);
            trfdp += f0.get(i,j)*dpd;
        } }
        println!("num tr[FdP] = {:.6}  ana pulay = {:.6}", trfdp, pulay);
        // 直接核对 SCF 报告的轨道能量与 Fock 特征值。
        println!("scf.mo_energies = {:?}", scf.mo_energies);
        // 对最终 Fock 做"广义"特征分解 F C = S C ε（对称正交化），得到真实 MO 能量。
        let fock_dense = scf.fock.to_dense();
        let fmat = faer::Mat::from_fn(nbf, nbf, |i, j| fock_dense[i*nbf+j]);
        let smat = faer::Mat::from_fn(nbf, nbf, |i, j| scf.overlap.get(i,j));
        let seig = faer::linalg::solvers::SelfAdjointEigendecomposition::new(smat.as_ref(), Side::Lower);
        let mut xmat = faer::Mat::<f64>::zeros(nbf, nbf);
        for i in 0..nbf { for j in 0..nbf {
            let mut acc = 0.0;
            for k in 0..nbf {
                let sv = *seig.s().column_vector().get(k);
                acc += seig.u().read(i,k)*(1.0/sv.sqrt())*seig.u().read(j,k);
            }
            xmat[(i,j)] = acc;
        } }
        // F' = X F X（两步）。
        let mut xf = faer::Mat::<f64>::zeros(nbf, nbf);
        faer::linalg::matmul::matmul(xf.as_mut(), xmat.as_ref(), fmat.as_ref(),
            None, 1.0, faer::Parallelism::None);
        let mut fprime = faer::Mat::<f64>::zeros(nbf, nbf);
        faer::linalg::matmul::matmul(fprime.as_mut(), xf.as_ref(), xmat.as_ref(),
            None, 1.0, faer::Parallelism::None);
        let feig = faer::linalg::solvers::SelfAdjointEigendecomposition::new(fprime.as_ref(), Side::Lower);
        let mut gen_evals = vec![0.0; nbf];
        for i in 0..nbf { gen_evals[i] = *feig.s().column_vector().get(i); }
        println!("generalized fock eigenvalues = {:?}", gen_evals);
        // 直接用 Fock 特征值重建 W_12。
        let c1 = scf.mo_coeff[0];
        let c2 = scf.mo_coeff[1];
        println!("C col0 = [{}, {}]", c1, c2);
        println!("P12 = {:.6}", scf.density.get(0,1));
    }
}

