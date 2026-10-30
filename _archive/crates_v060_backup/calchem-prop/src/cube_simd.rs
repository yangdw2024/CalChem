//! 三维规则网格评估器：分子轨道 / 电子密度 / 静电势（ESP），以及 Gaussian .cube 导出。
//!
//! 网格顺序（与 Gaussian .cube 一致）：x 最快，y 次之，z 最慢，
//! 即 `idx = ix + nx*iy + nx*ny*iz`。
//!
//! 性能：按"网格点"用 rayon 并行；每个原语在 α|r−R|²>36（exp<2.3e-16）时跳过。

use calchem_basis::{BasisSetData, MoleculeCore};
use calchem_int::one_electron_block;
use calchem_tensor::PackedSymMatrix;
use rayon::prelude::*;

/// 三维规则网格规格。
pub struct Grid3DSpec {
    /// 网格原点（Bohr）。
    pub origin_bohr: [f64; 3],
    /// 三条轴向量（每个元素是沿该轴的位移，Bohr）。
    pub axes_bohr: [[f64; 3]; 3],
    /// 三条轴上的格点数。
    pub dims: [usize; 3],
}

impl Grid3DSpec {
    /// 轴对齐网格（原点 + 三个方向的间距）。
    pub fn axis_aligned(
        origin: [f64; 3],
        spacing: [f64; 3],
        dims: [usize; 3],
    ) -> Grid3DSpec {
        Grid3DSpec {
            origin_bohr: origin,
            axes_bohr: [
                [spacing[0], 0.0, 0.0],
                [0.0, spacing[1], 0.0],
                [0.0, 0.0, spacing[2]],
            ],
            dims,
        }
    }

    /// 总格点数。
    pub fn total(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    /// 索引 (ix,iy,iz) 对应的空间点（Bohr）。
    pub fn point(&self, ix: usize, iy: usize, iz: usize) -> [f64; 3] {
        let mut r = self.origin_bohr;
        for k in 0..3 {
            let n = [ix, iy, iz][k];
            r[0] += n as f64 * self.axes_bohr[k][0];
            r[1] += n as f64 * self.axes_bohr[k][1];
            r[2] += n as f64 * self.axes_bohr[k][2];
        }
        r
    }
}

/// 评估点 r 处所有基函数值（长度 nbf）。
///
/// χ_μ(r) = Σ_p c_p N_p (r−R)^(l,m,n) exp(−α_p|r−R|²)。
pub fn eval_basis_functions(basis: &BasisSetData, r: [f64; 3]) -> Vec<f64> {
    let mut vals = vec![0.0; basis.nbf];
    for sh in 0..basis.shells.len() {
        let shell = &basis.shells[sh];
        let (s, _e) = basis.shell_ao_range(sh);
        let la = shell.angular_l as usize;
        let cart = calchem_basis::cartesian_angular_indices(la);
        let dx = r[0] - shell.center_bohr[0];
        let dy = r[1] - shell.center_bohr[1];
        let dz = r[2] - shell.center_bohr[2];
        let r2 = dx * dx + dy * dy + dz * dz;
        for p in 0..shell.n_prim() {
            let alpha = shell.exponents[p];
            let c = shell.coefficients[p];
            // 径向截断：α|r−R|²>36 时 exp<2.3e-16，跳过。
            if alpha * r2 > 36.0 {
                continue;
            }
            let g = (-alpha * r2).exp();
            for (k, &(al, am, an)) in cart.iter().enumerate() {
                let n = calchem_int::primitive_norm(alpha, al, am, an);
                let poly = dx.powi(al as i32) * dy.powi(am as i32) * dz.powi(an as i32);
                vals[s + k] += c * n * poly * g;
            }
        }
    }
    vals
}

/// 评估指定分子轨道（mo_vec 为该轨道的系数，长度 nbf）在网格上的值。
pub fn eval_orbital_grid(basis: &BasisSetData, mo_vec: &[f64], grid: &Grid3DSpec) -> Vec<f32> {
    let (nx, ny, nz) = (grid.dims[0], grid.dims[1], grid.dims[2]);
    let mut out = vec![0.0f32; grid.total()];
    // 按 (iy,iz) 划分 x 扫描线，并行。
    let lines: Vec<(usize, usize)> = (0..ny)
        .flat_map(|iy| (0..nz).map(move |iz| (iy, iz)))
        .collect();
    let line_results: Vec<(usize, Vec<f32>)> = lines
        .par_iter()
        .map(|&(iy, iz)| {
            let base = nx * ny * iz + nx * iy;
            let mut line = vec![0.0f32; nx];
            for ix in 0..nx {
                let r = grid.point(ix, iy, iz);
                let bfs = eval_basis_functions(basis, r);
                let mut v = 0.0;
                for m in 0..basis.nbf {
                    v += mo_vec[m] * bfs[m];
                }
                line[ix] = v as f32;
            }
            (base, line)
        })
        .collect();
    for (base, line) in line_results {
        out[base..base + nx].copy_from_slice(&line);
    }
    out
}

/// 评估电子密度 ρ(r) = Σ_μν P_μν χ_μ(r) χ_ν(r) 在网格上的值。
pub fn eval_density_grid(basis: &BasisSetData, dm: &PackedSymMatrix, grid: &Grid3DSpec) -> Vec<f32> {
    let (nx, ny, nz) = (grid.dims[0], grid.dims[1], grid.dims[2]);
    let mut out = vec![0.0f32; grid.total()];
    let lines: Vec<(usize, usize)> = (0..ny)
        .flat_map(|iy| (0..nz).map(move |iz| (iy, iz)))
        .collect();
    let line_results: Vec<(usize, Vec<f32>)> = lines
        .par_iter()
        .map(|&(iy, iz)| {
            let base = nx * ny * iz + nx * iy;
            let mut line = vec![0.0f32; nx];
            for ix in 0..nx {
                let r = grid.point(ix, iy, iz);
                let bfs = eval_basis_functions(basis, r);
                let mut v = 0.0;
                for m in 0..basis.nbf {
                    for n in 0..basis.nbf {
                        v += dm.get(m, n) * bfs[m] * bfs[n];
                    }
                }
                line[ix] = v as f32;
            }
            (base, line)
        })
        .collect();
    for (base, line) in line_results {
        out[base..base + nx].copy_from_slice(&line);
    }
    out
}

/// 评估静电势（ESP）在网格上的值：
///
///   ESP(r) = Σ_A Z_A/|r−R_A| − Σ_μν P_μν V_μν(r)，
///   V_μν(r) = ∫ χ_μ(r') χ_ν(r')/|r−r'| dr'
///           （核吸引积分，把网格点 r 当作单位点电荷）。
pub fn eval_esp_grid(
    mol: &MoleculeCore,
    basis: &BasisSetData,
    dm: &PackedSymMatrix,
    grid: &Grid3DSpec,
) -> Vec<f32> {
    let (nx, ny, nz) = (grid.dims[0], grid.dims[1], grid.dims[2]);
    let mut out = vec![0.0f32; grid.total()];
    let lines: Vec<(usize, usize)> = (0..ny)
        .flat_map(|iy| (0..nz).map(move |iz| (iy, iz)))
        .collect();
    let line_results: Vec<(usize, Vec<f32>)> = lines
        .par_iter()
        .map(|&(iy, iz)| {
            let base = nx * ny * iz + nx * iy;
            let mut line = vec![0.0f32; nx];
            for ix in 0..nx {
                let r = grid.point(ix, iy, iz);
                // 核贡献。
                let mut vnuc = 0.0;
                for a in 0..mol.atoms.len() {
                    let d = dist(r, mol.atoms[a].coord_bohr);
                    if d > 1e-10 {
                        vnuc += mol.atoms[a].atomic_number as f64 / d;
                    }
                }
                // 电子贡献：Σ_μν P_μν V_μν(r)。
                // 注意：one_electron_block(op=2) 遵循"电子-核吸引"约定，带负号
                // （返回 −Z·<1/|r−C|>）；ESP 要正的静电势，故取负。
                let mut velec = 0.0;
                let nsh = basis.shells.len();
                for sa in 0..nsh {
                    let (a0, a1) = basis.shell_ao_range(sa);
                    for sb in 0..nsh {
                        let (b0, b1) = basis.shell_ao_range(sb);
                        let block = one_electron_block(
                            &basis.shells[sa],
                            &basis.shells[sb],
                            2,
                            &[(r, 1.0)],
                        );
                        let nrow = a1 - a0;
                        let ncol = b1 - b0;
                        for ii in 0..nrow {
                            for jj in 0..ncol {
                                velec -= dm.get(a0 + ii, b0 + jj)
                                    * block[ii * ncol + jj];
                            }
                        }
                    }
                }
                line[ix] = (vnuc - velec) as f32;
            }
            (base, line)
        })
        .collect();
    for (base, line) in line_results {
        out[base..base + nx].copy_from_slice(&line);
    }
    out
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d0 = a[0] - b[0];
    let d1 = a[1] - b[1];
    let d2 = a[2] - b[2];
    (d0 * d0 + d1 * d1 + d2 * d2).sqrt()
}

/// 写出标准 Gaussian .cube 文件。
///
/// 体素顺序：x 最快（与本模块网格一致）。
pub fn write_cube_file(
    path: &std::path::Path,
    mol: &MoleculeCore,
    grid: &Grid3DSpec,
    values: &[f32],
) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    let natm = mol.atoms.len();
    writeln!(f, "Generated by CalChem")?;
    writeln!(f, "Cube export")?;
    // 原子数 + 原点。
    writeln!(
        f,
        "{:5} {:12.6} {:12.6} {:12.6}",
        natm, grid.origin_bohr[0], grid.origin_bohr[1], grid.origin_bohr[2]
    )?;
    // 三条轴：点数 + 轴向量。
    for k in 0..3 {
        writeln!(
            f,
            "{:5} {:12.6} {:12.6} {:12.6}",
            grid.dims[k], grid.axes_bohr[k][0], grid.axes_bohr[k][1], grid.axes_bohr[k][2]
        )?;
    }
    // 原子：Z + 电荷（与 Z 同）+ 坐标。
    for a in 0..natm {
        let z = mol.atoms[a].atomic_number as f64;
        writeln!(
            f,
            "{:5} {:12.6} {:12.6} {:12.6} {:12.6}",
            mol.atoms[a].atomic_number,
            z,
            mol.atoms[a].coord_bohr[0],
            mol.atoms[a].coord_bohr[1],
            mol.atoms[a].coord_bohr[2]
        )?;
    }
    // 数据：每行 6 个值。
    for chunk in values.chunks(6) {
        for v in chunk {
            write!(f, " {:13.5e}", v)?;
        }
        writeln!(f)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h2() -> (MoleculeCore, BasisSetData, calchem_scf::ScfOutput) {
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
        (mol, basis, out)
    }

    /// 网格点值与逐点直接评估一致（轨道 & 密度）。
    #[test]
    fn grid_matches_point_eval() {
        let (_mol, basis, out) = h2();
        let nbf = basis.nbf;
        // 占据轨道系数（第 0 列）。
        let mo0: Vec<f64> = (0..nbf).map(|m| out.mo_coeff[m * nbf]).collect();
        let grid = Grid3DSpec::axis_aligned(
            [-2.0, -2.0, -2.0],
            [0.4, 0.4, 0.4],
            [11, 11, 14],
        );
        let mo_grid = eval_orbital_grid(&basis, &mo0, &grid);
        let rho_grid = eval_density_grid(&basis, &out.density, &grid);
        // 抽 5 个点，逐点直接算并比对。
        for &(ix, iy, iz) in &[(0, 0, 0), (3, 5, 2), (7, 2, 9), (10, 10, 13), (5, 8, 6)] {
            let r = grid.point(ix, iy, iz);
            let bfs = eval_basis_functions(&basis, r);
            let mo_v: f64 = (0..nbf).map(|m| mo0[m] * bfs[m]).sum();
            let mut rho_v = 0.0;
            for m in 0..nbf {
                for n in 0..nbf {
                    rho_v += out.density.get(m, n) * bfs[m] * bfs[n];
                }
            }
            let idx = ix + grid.dims[0] * iy + grid.dims[0] * grid.dims[1] * iz;
            assert!(
                (mo_grid[idx] as f64 - mo_v).abs() < 1e-6,
                "mo idx={} grid={} point={}",
                idx, mo_grid[idx], mo_v
            );
            assert!(
                (rho_grid[idx] as f64 - rho_v).abs() < 1e-6,
                "rho idx={} grid={} point={}",
                idx, rho_grid[idx], rho_v
            );
        }
    }

    /// 密度在覆盖分子的盒子上积分 ≈ 电子数（2）。
    #[test]
    fn density_integral_is_nelectrons() {
        let (_mol, basis, out) = h2();
        let h = 0.2;
        // 盒子覆盖 H2（沿 z，0 与 1.4）：x,y ∈ [-3,3)，z ∈ [-3, 4.4)。
        let nx = 30;
        let ny = 30;
        let nz = 37;
        let grid = Grid3DSpec::axis_aligned([-3.0, -3.0, -3.0], [h, h, h], [nx, ny, nz]);
        let rho = eval_density_grid(&basis, &out.density, &grid);
        let sum: f64 = rho.iter().map(|v| *v as f64).sum::<f64>() * h * h * h;
        // 矩形积分在盒子边界有少量泄漏，要求 1% 以内。
        assert!(
            (sum - 2.0).abs() < 0.02,
            "∫ρ dr = {} expected ~2",
            sum
        );
    }

    /// 写出 .cube 文件并能读回（检查文件头与数据长度）。
    #[test]
    fn cube_roundtrip() {
        let (mol, basis, out) = h2();
        let nbf = basis.nbf;
        let mo0: Vec<f64> = (0..nbf).map(|m| out.mo_coeff[m * nbf]).collect();
        let grid = Grid3DSpec::axis_aligned([-2.0, -2.0, -2.0], [0.5, 0.5, 0.5], [9, 9, 11]);
        let values = eval_orbital_grid(&basis, &mo0, &grid);
        let path = std::path::Path::new(r"E:\YDW\Trae_Gaussian\_build_temp\test_h2_mo.cube");
        write_cube_file(path, &mol, &grid, &values).unwrap();
        // 读回：检查文件头含原子数与数据总数。
        let text = std::fs::read_to_string(path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        // 第 3 行（index 2）含 natm=2。
        assert!(lines[2].trim().starts_with('2'), "natm header: {}", lines[2]);
        // 数据值总数 = grid.total()。
        // 头部：2 标题 + 1（原子数/原点）+ 3 轴 + natm 原子行。
        let header = 2 + 1 + 3 + mol.atoms.len();
        let mut nvals = 0usize;
        for line in lines.iter().skip(header) {
            nvals += line.split_whitespace().count();
        }
        assert_eq!(nvals, grid.total(), "cube data count");
    }

    /// 远离分子处 ESP ≈ 总电荷/r（中性分子 → 快速衰减）。
    #[test]
    fn esp_far_field() {
        let (mol, basis, out) = h2();
        // 在 z 方向远处取点。
        let r = [0.0, 0.0, 10.0];
        let mut vnuc = 0.0;
        for a in 0..mol.atoms.len() {
            let d = dist(r, mol.atoms[a].coord_bohr);
            vnuc += mol.atoms[a].atomic_number as f64 / d;
        }
        let mut velec = 0.0;
        for sa in 0..basis.shells.len() {
            let (a0, a1) = basis.shell_ao_range(sa);
            for sb in 0..basis.shells.len() {
                let (b0, b1) = basis.shell_ao_range(sb);
                let block = one_electron_block(
                    &basis.shells[sa],
                    &basis.shells[sb],
                    2,
                    &[(r, 1.0)],
                );
                let nrow = a1 - a0;
                let ncol = b1 - b0;
                for ii in 0..nrow {
                    for jj in 0..ncol {
                        // 取负：op=2 带电子-核吸引的负号。
                        velec -= out.density.get(a0 + ii, b0 + jj) * block[ii * ncol + jj];
                    }
                }
            }
        }
        let esp = vnuc - velec;
        println!("ESP debug: vnuc={} velec={} esp={}", vnuc, velec, esp);
        // 中性分子在 r=10 处 ESP 应很小（<0.01）。
        assert!(esp.abs() < 0.01, "ESP far = {}", esp);
    }
}

