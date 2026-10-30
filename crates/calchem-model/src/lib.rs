//! CalChem 共享值类型。

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CalcError {
    #[error("无效输入: {0}")]
    InvalidInput(String),
    #[error("不支持: {0}")]
    Unsupported(String),
    #[error("数值错误: {0}")]
    Numerical(String),
    #[error("SCF 未收敛")]
    NotConverged,
    #[error("资源限制: {0}")]
    ResourceLimit(String),
}

#[derive(Debug, Clone)]
pub struct Atom {
    pub z: u8,
    pub position_bohr: [f64; 3],
}

#[derive(Debug, Clone)]
pub struct Molecule {
    pub atoms: Vec<Atom>,
    pub charge: i32,
    pub multiplicity: u32,
}

impl Molecule {
    pub fn try_new(atoms: Vec<Atom>, charge: i32, multiplicity: u32) -> Result<Self, CalcError> {
        if atoms.is_empty() {
            return Err(CalcError::InvalidInput("分子为空".into()));
        }
        for a in &atoms {
            if a.z == 0 {
                return Err(CalcError::InvalidInput("原子序数为 0".into()));
            }
        }
        Ok(Self { atoms, charge, multiplicity })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Reference {
    Rhf, Uhf, Rks, Uks,
}
