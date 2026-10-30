//! 内置基组加载器：从内置数据库读取原始壳层记录，按原子排布并计算 AO 偏移。

use crate::basis_database;
use crate::{cartesian_angular_indices, AngularType, BasisSetData, MoleculeCore, Shell};

/// 原语归一化 N(α,l,m,n)，与 Python `gto.py` 一致：
///   N = (2α/π)^0.75 · (4α)^((l+m+n)/2) / √[(2l-1)!!(2m-1)!!(2n-1)!!]
fn primitive_norm(alpha: f64, l: usize, m: usize, n: usize) -> f64 {
    let odd_df = |k: usize| -> f64 {
        // (2k-1)!!，k=0 时为 1。
        let mut p = 1.0;
        for i in 1..=k {
            p *= (2 * i - 1) as f64;
        }
        p
    };
    let num = (2.0 * alpha / std::f64::consts::PI).powf(0.75)
        * (4.0 * alpha).powf((l + m + n) as f64 / 2.0);
    let den = (odd_df(l) * odd_df(m) * odd_df(n)).sqrt();
    num / den
}

/// 收缩归一化：把标准库存系数（通常为 8 位有效数字，自重叠 ≈ 0.99999999）
/// 缩放为自重叠恰好 1。与 Python `GTOShell._normalize_contraction` 完全一致。
///
/// 返回缩放后的收缩系数。
fn renormalize_contraction(angular_l: u8, exponents: &[f64], coeffs: &[f64]) -> Vec<f64> {
    let l = angular_l as usize;
    // 取第一个笛卡尔分量（径向归一化对所有分量相同）。
    let (cl, cm, cn) = cartesian_angular_indices(l)[0];
    let np = exponents.len();
    let mut s = vec![0.0; np * np];
    let odd_df = |k: usize| -> f64 {
        let mut p = 1.0;
        for i in 1..=k {
            p *= (2 * i - 1) as f64;
        }
        p
    };
    for i in 0..np {
        for j in 0..np {
            let ap = exponents[i] + exponents[j];
            let mut ang = 1.0;
            if cl > 0 {
                ang *= odd_df(cl) / (2.0 * ap).powf(cl as f64);
            }
            if cm > 0 {
                ang *= odd_df(cm) / (2.0 * ap).powf(cm as f64);
            }
            if cn > 0 {
                ang *= odd_df(cn) / (2.0 * ap).powf(cn as f64);
            }
            let integral = (std::f64::consts::PI / ap).powf(1.5) * ang;
            let ni = primitive_norm(exponents[i], cl, cm, cn);
            let nj = primitive_norm(exponents[j], cl, cm, cn);
            s[i * np + j] = ni * nj * integral;
        }
    }
    // self_overlap = c @ S @ c。
    let mut self_overlap = 0.0;
    for i in 0..np {
        let mut row = 0.0;
        for j in 0..np {
            row += s[i * np + j] * coeffs[j];
        }
        self_overlap += coeffs[i] * row;
    }
    let f = 1.0 / self_overlap.sqrt();
    coeffs.iter().map(|c| c * f).collect()
}

/// 原始壳层记录（与具体原子无关的模板）。
#[derive(Debug, Clone)]
pub struct RawShell {
    /// 角动量 l。
    pub angular_l: u8,
    /// 原语指数。
    pub exponents: Vec<f64>,
    /// 归一化原语的收缩系数。
    pub coefficients: Vec<f64>,
}

/// 基组名称归一化（小写、去空格）。
pub fn normalize_basis_name(name: &str) -> String {
    name.trim().to_lowercase()
}

/// 加载内置基组并为分子构建 BasisSetData。
///
/// 不做任何写死/伪造：壳层中心、指数、系数全部来自内置数据库（源自标准 BSE 数据）。
pub fn load_builtin_basis(
    name: &str,
    mol: &MoleculeCore,
    angular: AngularType,
) -> Result<BasisSetData, String> {
    let key = normalize_basis_name(name);
    let mut shells: Vec<Shell> = Vec::new();
    let mut offset: usize = 0;

    for (atom_idx, atom) in mol.atoms.iter().enumerate() {
        let recs = basis_database::element_records(&key, atom.atomic_number).ok_or_else(|| {
            format!(
                "基组 '{}' 不包含元素 {} (Z={}) 的数据",
                name, atom.symbol, atom.atomic_number
            )
        })?;

        for raw in recs {
            let nf = shell_n_functions(raw.angular_l, angular);
            // 收缩归一化（与 Python 参照一致），使每个收缩基函数自重叠恰好 1。
            let coeffs = renormalize_contraction(raw.angular_l, &raw.exponents, &raw.coefficients);
            shells.push(Shell {
                atom_idx: atom_idx as u32,
                angular_l: raw.angular_l,
                center_bohr: atom.coord_bohr,
                exponents: raw.exponents,
                coefficients: coeffs,
                ao_offset: offset as u32,
            });
            offset += nf;
        }
    }

    Ok(BasisSetData {
        name: key,
        angular_type: angular,
        shells,
        nbf: offset,
    })
}

/// 给定角动量与类型，返回该壳层基函数数。
fn shell_n_functions(l: u8, at: AngularType) -> usize {
    let l = l as usize;
    match at {
        AngularType::Spherical => 2 * l + 1,
        AngularType::Cartesian => (l + 1) * (l + 2) / 2,
    }
}
