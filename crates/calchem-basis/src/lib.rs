pub mod normalization;

use calchem_model::{CalcError, Molecule};
pub use normalization::{
    double_factorial_odd, normalized_contraction, normalized_primitive_self_overlap, primitive_norm,
};
use serde::Deserialize;
use std::collections::HashMap;

const STO_3G_JSON: &str = include_str!("../data/sto-3g.json");
const BASIS_631G_JSON: &str = include_str!("../data/6-31g.json");

#[derive(Debug, Clone, Deserialize)]
struct RawShellData {
    shell_type: String,
    exponents: Vec<f64>,
    coefficients: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawBasisFile {
    name: String,
    #[allow(dead_code)]
    source: Option<String>,
    elements: HashMap<String, Vec<RawShellData>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellType {
    S,
    P,
    D,
}

impl ShellType {
    pub fn angular_momentum(&self) -> u8 {
        match self {
            ShellType::S => 0,
            ShellType::P => 1,
            ShellType::D => 2,
        }
    }

    pub fn n_ao(&self) -> usize {
        match self {
            ShellType::S => 1,
            ShellType::P => 3,
            ShellType::D => 6,
        }
    }

    #[allow(non_snake_case)]
    pub fn n_AO(&self) -> usize {
        self.n_ao()
    }

    pub fn from_str(s: &str) -> Result<Self, CalcError> {
        match s.trim().to_ascii_lowercase().as_str() {
            "s" => Ok(ShellType::S),
            "p" => Ok(ShellType::P),
            "d" => Ok(ShellType::D),
            other => Err(CalcError::UnsupportedBasis(format!(
                "Unsupported shell type: '{other}'"
            ))),
        }
    }

    pub fn components(&self) -> &'static [[u8; 3]] {
        match self {
            ShellType::S => &[[0, 0, 0]],
            ShellType::P => &[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            ShellType::D => &[
                [2, 0, 0],
                [1, 1, 0],
                [1, 0, 1],
                [0, 2, 0],
                [0, 1, 1],
                [0, 0, 2],
            ],
        }
    }
}

#[derive(Debug, Clone)]
pub struct Primitive {
    pub exponent: f64,
    /// Contraction-normalized coefficient `\tilde{c}_p` such that `<phi | phi> = 1`.
    pub coeff: f64,
    /// Original unnormalized contraction coefficient from the basis set table.
    pub raw_coeff: f64,
    /// Primitive normalization constant for the leading Cartesian component `(L, 0, 0)`.
    /// For Cartesian `d` shells where axial `(2,0,0)` and off-diagonal `(1,1,0)` norms differ,
    /// use `Primitive::norm_for(lmn)` or `Shell::primitive_norm_for(prim_idx, lmn)`.
    pub norm: f64,
}

impl Primitive {
    pub fn norm_for(&self, lmn: [u8; 3]) -> f64 {
        primitive_norm(self.exponent, lmn)
    }
}

#[derive(Debug, Clone)]
pub struct Shell {
    pub atom_index: usize,
    pub center: [f64; 3],
    pub shell_type: ShellType,
    pub primitives: Vec<Primitive>,
    pub ao_offset: usize,
}

impl Shell {
    pub fn n_functions(&self) -> usize {
        self.shell_type.n_ao()
    }

    pub fn components(&self) -> &'static [[u8; 3]] {
        self.shell_type.components()
    }

    pub fn primitive_norm_for(&self, prim_idx: usize, lmn: [u8; 3]) -> f64 {
        primitive_norm(self.primitives[prim_idx].exponent, lmn)
    }
}

pub fn norm_s(alpha: f64) -> f64 {
    primitive_norm(alpha, [0, 0, 0])
}

pub fn norm_p(alpha: f64) -> f64 {
    primitive_norm(alpha, [1, 0, 0])
}

/// Returns the axial Cartesian d normalization constant `N(alpha; 2, 0, 0)`.
/// For general Cartesian d components `(l, m, n)` with `l+m+n=2`, use `primitive_norm(alpha, lmn)`.
pub fn norm_d(alpha: f64) -> f64 {
    primitive_norm(alpha, [2, 0, 0])
}

#[derive(Debug, Clone)]
pub struct BasisSet {
    pub name: String,
    pub shells: Vec<Shell>,
    pub n_ao: usize,
}

impl BasisSet {
    pub fn from_name(name: &str, mol: &Molecule) -> Result<Self, CalcError> {
        let key = name.trim().to_ascii_lowercase();
        match key.as_str() {
            "sto-3g" | "sto3g" => Self::sto_3g(mol),
            "6-31g" | "631g" => Self::basis_6_31g(mol),
            other => Err(CalcError::UnsupportedBasis(format!(
                "Unsupported basis set: '{other}'"
            ))),
        }
    }

    pub fn sto_3g(mol: &Molecule) -> Result<Self, CalcError> {
        Self::from_json_str(STO_3G_JSON, mol)
    }

    pub fn basis_6_31g(mol: &Molecule) -> Result<Self, CalcError> {
        Self::from_json_str(BASIS_631G_JSON, mol)
    }

    pub fn from_json_str(json_str: &str, mol: &Molecule) -> Result<Self, CalcError> {
        let raw: RawBasisFile = serde_json::from_str(json_str)
            .map_err(|e| CalcError::Internal(format!("Basis JSON parse error: {e}")))?;
        Self::build(raw, mol)
    }

    fn build(raw: RawBasisFile, mol: &Molecule) -> Result<Self, CalcError> {
        if mol.atoms.is_empty() {
            return Err(CalcError::InvalidMolecule(
                "Cannot build basis set for empty molecule".into(),
            ));
        }

        let mut shells = Vec::new();
        let mut ao_offset = 0;

        for (atom_index, atom) in mol.atoms.iter().enumerate() {
            let elem_shells = raw.elements.get(&atom.symbol).ok_or_else(|| {
                CalcError::UnsupportedBasis(format!(
                    "Element '{}' not found in basis '{}'",
                    atom.symbol, raw.name
                ))
            })?;

            if elem_shells.is_empty() {
                return Err(CalcError::UnsupportedBasis(format!(
                    "Element '{}' has no shells in basis '{}'",
                    atom.symbol, raw.name
                )));
            }

            for rs in elem_shells {
                let stype = ShellType::from_str(&rs.shell_type)?;
                let rep_lmn = stype.components()[0];
                let norm_coeffs =
                    normalized_contraction(&rs.exponents, &rs.coefficients, rep_lmn)?;

                let mut prims = Vec::with_capacity(rs.exponents.len());
                for ((&exp, &raw_coeff), &coeff) in rs
                    .exponents
                    .iter()
                    .zip(rs.coefficients.iter())
                    .zip(norm_coeffs.iter())
                {
                    let norm = primitive_norm(exp, rep_lmn);
                    prims.push(Primitive {
                        exponent: exp,
                        coeff,
                        raw_coeff,
                        norm,
                    });
                }

                shells.push(Shell {
                    atom_index,
                    center: atom.position_bohr,
                    shell_type: stype,
                    primitives: prims,
                    ao_offset,
                });
                ao_offset += stype.n_ao();
            }
        }

        Ok(Self {
            name: raw.name,
            shells,
            n_ao: ao_offset,
        })
    }
}
