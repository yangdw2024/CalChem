//! 基组层：从 JSON 加载基组数据，为分子构建收缩壳层。

use serde::Deserialize;
use std::collections::HashMap;
use std::f64::consts::PI;

/// 单个 GTO 原语
#[derive(Debug, Clone)]
pub struct Primitive {
    pub alpha: f64,
    pub norm: f64,
}

/// 收缩壳层
#[derive(Debug, Clone)]
pub struct Shell {
    pub origin: [f64; 3],
    pub shell_type: ShellType,
    pub coeffs: Vec<f64>,
    pub exponents: Vec<f64>,
    pub primitives: Vec<Primitive>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShellType {
    S,
    P,
    D,
}

impl ShellType {
    pub fn angular_momentum(&self) -> u32 {
        match self {
            ShellType::S => 0,
            ShellType::P => 1,
            ShellType::D => 2,
        }
    }

    pub fn n_functions(&self) -> usize {
        match self {
            ShellType::S => 1,
            ShellType::P => 3,
            ShellType::D => 6, // Cartesian
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "s" => ShellType::S,
            "p" => ShellType::P,
            "d" => ShellType::D,
            _ => ShellType::S,
        }
    }
}

/// 笛卡尔角动量组合
pub fn cartesian_indices(L: u32) -> Vec<(i32, i32, i32)> {
    let mut result = Vec::new();
    for l in (0..=L).rev() {
        for m in (0..=L - l).rev() {
            let n = L - l - m;
            result.push((l as i32, m as i32, n as i32));
        }
    }
    result
}

fn double_factorial(n: i32) -> f64 {
    if n <= 0 {
        return 1.0;
    }
    let mut r = 1.0;
    let mut v = n;
    while v > 0 {
        r *= v as f64;
        v -= 2;
    }
    r
}

/// 计算单个原语的归一化常数（s 型）
fn norm_s(alpha: f64) -> f64 {
    (2.0 * alpha / PI).powf(0.75)
}

/// 计算 p 型原语归一化
fn norm_p(alpha: f64) -> f64 {
    (2.0 * alpha / PI).powf(0.75) * (4.0 * alpha).sqrt()
}

/// 计算 d 型原语归一化（笛卡尔）
fn norm_d(alpha: f64) -> f64 {
    (2.0 * alpha / PI).powf(0.75) * (4.0 * alpha) / (2.0_f64.sqrt())
}

#[derive(Debug, Deserialize)]
struct BasisFile {
    elements: HashMap<String, Vec<ShellDef>>,
}

#[derive(Debug, Deserialize)]
struct ShellDef {
    coefficients: Vec<f64>,
    exponents: Vec<f64>,
    shell_type: String,
}

/// 完整基组
pub struct BasisSet {
    pub name: String,
    shells: Vec<Shell>,
    shell_ranges: Vec<(usize, usize)>,
    pub nbf: usize,
}

impl BasisSet {
    /// 从 JSON 数据加载基组并为分子构建壳层
    pub fn build(
        name: &str,
        json_data: &str,
        atoms: &[(u8, [f64; 3])], // (atomic_number, position_bohr)
    ) -> Result<Self, String> {
        let bf: BasisFile = serde_json::from_str(json_data)
            .map_err(|e| format!("基组 JSON 解析失败: {e}"))?;

        let mut shells = Vec::new();

        for &(z, pos) in atoms {
            let symbol = atomic_number_to_symbol(z)?;
            let shell_defs = bf.elements.get(symbol)
                .ok_or_else(|| format!("基组中找不到元素 {symbol}"))?;

            for def in shell_defs {
                let stype = ShellType::from_str(&def.shell_type);
                let mut primitives = Vec::new();

                for &alpha in &def.exponents {
                    let norm = match stype {
                        ShellType::S => norm_s(alpha),
                        ShellType::P => norm_p(alpha),
                        ShellType::D => norm_d(alpha),
                    };
                    primitives.push(Primitive { alpha, norm });
                }

                shells.push(Shell {
                    origin: pos,
                    shell_type: stype,
                    coeffs: def.coefficients.clone(),
                    exponents: def.exponents.clone(),
                    primitives,
                });
            }
        }

        // 计算基函数范围
        let mut shell_ranges = Vec::new();
        let mut idx = 0;
        for s in &shells {
            let nf = s.shell_type.n_functions();
            shell_ranges.push((idx, idx + nf));
            idx += nf;
        }

        Ok(Self {
            name: name.to_string(),
            shells,
            shell_ranges,
            nbf: idx,
        })
    }

    pub fn shells(&self) -> &[Shell] {
        &self.shells
    }

    pub fn shell_ranges(&self) -> &[(usize, usize)] {
        &self.shell_ranges
    }

    /// 获取指定基函数对应的 (shell_index, function_index_in_shell)
    pub fn bf_to_shell_func(&self, bf_idx: usize) -> (usize, usize) {
        for (i, &(start, end)) in self.shell_ranges.iter().enumerate() {
            if bf_idx >= start && bf_idx < end {
                return (i, bf_idx - start);
            }
        }
        panic!("基函数索引越界: {bf_idx}");
    }
}

fn atomic_number_to_symbol(z: u8) -> Result<&'static str, String> {
    match z {
        1 => Ok("H"),
        2 => Ok("He"),
        3 => Ok("Li"),
        4 => Ok("Be"),
        5 => Ok("B"),
        6 => Ok("C"),
        7 => Ok("N"),
        8 => Ok("O"),
        9 => Ok("F"),
        10 => Ok("Ne"),
        11 => Ok("Na"),
        12 => Ok("Mg"),
        13 => Ok("Al"),
        14 => Ok("Si"),
        15 => Ok("P"),
        16 => Ok("S"),
        17 => Ok("Cl"),
        18 => Ok("Ar"),
        _ => Err(format!("不支持的原子序数: {z}")),
    }
}
