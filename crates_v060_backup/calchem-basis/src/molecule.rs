//! 分子拓扑：原子、分子核心、核排斥能。

/// 一个原子。
#[derive(Debug, Clone)]
pub struct Atom {
    /// 原子序数（核电荷数）。
    pub atomic_number: u8,
    /// 元素符号。
    pub symbol: String,
    /// 坐标（Bohr）。
    pub coord_bohr: [f64; 3],
}

/// 分子核心：原子列表 + 电荷 + 自旋多重度。
#[derive(Debug, Clone)]
pub struct MoleculeCore {
    pub atoms: Vec<Atom>,
    pub charge: i32,
    pub multiplicity: usize,
}

impl MoleculeCore {
    /// 由原子序数（每个原子一个）与展平坐标（每原子 3 个，Bohr）构造。
    pub fn new(atomic_numbers: &[u8], coords_bohr: &[f64], charge: i32, multiplicity: usize) -> Self {
        let n = atomic_numbers.len();
        assert_eq!(coords_bohr.len(), 3 * n);
        let mut atoms = Vec::with_capacity(n);
        for i in 0..n {
            let z = atomic_numbers[i];
            atoms.push(Atom {
                atomic_number: z,
                symbol: element_symbol(z).to_string(),
                coord_bohr: [
                    coords_bohr[i * 3],
                    coords_bohr[i * 3 + 1],
                    coords_bohr[i * 3 + 2],
                ],
            });
        }
        Self {
            atoms,
            charge,
            multiplicity,
        }
    }

    /// 原子数。
    pub fn natom(&self) -> usize {
        self.atoms.len()
    }

    /// 总电子数（中性时为核电荷之和，减去净电荷）。
    pub fn n_electrons(&self) -> usize {
        let zsum: i64 = self.atoms.iter().map(|a| a.atomic_number as i64).sum();
        (zsum - self.charge as i64) as usize
    }

    /// 核排斥能 E_nuc = Σ_{A<B} Z_A Z_B / R_AB。
    pub fn nuclear_repulsion_energy(&self) -> f64 {
        let mut e = 0.0;
        let n = self.atoms.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let r = self.inter_distance(i, j);
                e += (self.atoms[i].atomic_number as f64) * (self.atoms[j].atomic_number as f64) / r;
            }
        }
        e
    }

    /// 两原子间距离（Bohr）。
    pub fn inter_distance(&self, i: usize, j: usize) -> f64 {
        let a = &self.atoms[i].coord_bohr;
        let b = &self.atoms[j].coord_bohr;
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// 核排斥能对原子坐标的解析梯度 [n_atom, 3]，单位 Eh/Bohr。
    pub fn nuclear_repulsion_grad(&self) -> Vec<[f64; 3]> {
        let n = self.atoms.len();
        let mut g = vec![[0.0; 3]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let a = self.atoms[i].coord_bohr;
                let b = self.atoms[j].coord_bohr;
                let dx = a[0] - b[0];
                let dy = a[1] - b[1];
                let dz = a[2] - b[2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                let zi = self.atoms[i].atomic_number as f64;
                let zj = self.atoms[j].atomic_number as f64;
                // d(1/r)/dR_i = -(R_i - R_j)/r^3
                let f = -zi * zj / (r * r * r);
                for k in 0..3 {
                    let d = a[k] - b[k];
                    g[i][k] += f * d;
                    g[j][k] -= f * d;
                }
            }
        }
        g
    }
}

/// 元素符号表（原子序数 => 符号），覆盖前 54 号元素及常见重金属。
pub fn element_symbol(z: u8) -> &'static str {
    match z {
        1 => "H", 2 => "He",
        3 => "Li", 4 => "Be", 5 => "B", 6 => "C", 7 => "N", 8 => "O", 9 => "F", 10 => "Ne",
        11 => "Na", 12 => "Mg", 13 => "Al", 14 => "Si", 15 => "P", 16 => "S", 17 => "Cl", 18 => "Ar",
        19 => "K", 20 => "Ca", 21 => "Sc", 22 => "Ti", 23 => "V", 24 => "Cr", 25 => "Mn",
        26 => "Fe", 27 => "Co", 28 => "Ni", 29 => "Cu", 30 => "Zn", 31 => "Ga", 32 => "Ge",
        33 => "As", 34 => "Se", 35 => "Br", 36 => "Kr",
        37 => "Rb", 38 => "Sr", 39 => "Y", 40 => "Zr", 41 => "Nb", 42 => "Mo", 43 => "Tc",
        44 => "Ru", 45 => "Rh", 46 => "Pd", 47 => "Ag", 48 => "Cd", 49 => "In", 50 => "Sn",
        51 => "Sb", 52 => "Te", 53 => "I", 54 => "Xe",
        78 => "Pt", 79 => "Au", 80 => "Hg", 82 => "Pb",
        _ => "X",
    }
}

/// 元素符号 => 原子序数（找不到返回 0）。
pub fn atomic_number_from_symbol(sym: &str) -> u8 {
    let s = sym.trim();
    for z in 1u8..=82 {
        if element_symbol(z).eq_ignore_ascii_case(s) {
            return z;
        }
    }
    0
}
