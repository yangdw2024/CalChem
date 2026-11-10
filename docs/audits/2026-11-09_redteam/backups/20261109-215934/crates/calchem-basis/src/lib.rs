pub mod normalization;

use calchem_model::{atomic_number_from_symbol, CalcError, Molecule};
pub use normalization::{
    double_factorial_odd, normalized_contraction, normalized_primitive_self_overlap, primitive_norm,
};
use serde::Deserialize;
use std::collections::HashMap;

const STO_3G_JSON: &str = include_str!("../data/sto-3g.json");
const BASIS_631G_JSON: &str = include_str!("../data/6-31g.json");
const BASIS_631GD_JSON: &str = include_str!("../data/6-31gd.json");
const BASIS_631GDP_JSON: &str = include_str!("../data/6-31gdp.json");
const LANL2DZ_JSON: &str = include_str!("../data/lanl2dz.json");
const LANL2DZDP_JSON: &str = include_str!("../data/lanl2dzdp.json");

#[derive(Debug, Clone, Deserialize)]
struct RawShellData {
    shell_type: String,
    exponents: Vec<f64>,
    coefficients: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawEcpTerm {
    r_power: u8,
    exponent: f64,
    coeff: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct RawEcpChannel {
    angular_momentum: u8,
    terms: Vec<RawEcpTerm>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawElementEcp {
    core_electrons: u8,
    l_max: u8,
    local: Vec<RawEcpTerm>,
    #[serde(default)]
    semilocal: Vec<RawEcpChannel>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawBasisFile {
    #[serde(default)]
    name: String,
    #[allow(dead_code)]
    source: Option<String>,
    elements: HashMap<String, Vec<RawShellData>>,
    #[serde(default)]
    ecp: HashMap<String, RawElementEcp>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellType {
    S,
    P,
    D,
    F,
}

impl ShellType {
    pub fn angular_momentum(&self) -> u8 {
        match self {
            ShellType::S => 0,
            ShellType::P => 1,
            ShellType::D => 2,
            ShellType::F => 3,
        }
    }

    pub fn n_ao(&self) -> usize {
        match self {
            ShellType::S => 1,
            ShellType::P => 3,
            ShellType::D => 6,
            ShellType::F => 10,
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
            "f" => Ok(ShellType::F),
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
            ShellType::F => &[
                [3, 0, 0],
                [2, 1, 0],
                [2, 0, 1],
                [1, 2, 0],
                [1, 1, 1],
                [1, 0, 2],
                [0, 3, 0],
                [0, 2, 1],
                [0, 1, 2],
                [0, 0, 3],
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
    /// For Cartesian `d`/`f` shells where axial and off-diagonal norms differ,
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

/// A single radial Gaussian term `coeff * r^{r_power - 2} * exp(-exponent * r^2)` in an ECP channel.
/// Note that when integrated in spherical coordinates with `r^2 dr`, the radial integrand has
/// non-negative power `r^{r_power} * exp(-exponent * r^2) dr`.
#[derive(Debug, Clone, Copy)]
pub struct EcpTerm {
    pub r_power: u8,
    pub exponent: f64,
    pub coeff: f64,
}

/// A semilocal angular-momentum projector channel `\Delta U_l(r) \hat{P}_l` (`l < l_max`).
#[derive(Debug, Clone)]
pub struct EcpChannel {
    pub angular_momentum: u8,
    pub terms: Vec<EcpTerm>,
}

/// Scalar Effective Core Potential (ECP) centered on a specific atom in `Molecule`.
#[derive(Debug, Clone)]
pub struct AtomEcp {
    pub atom_index: usize,
    pub center: [f64; 3],
    pub core_electrons: u8,
    pub l_max: u8,
    pub local_terms: Vec<EcpTerm>,
    pub semilocal_channels: Vec<EcpChannel>,
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
    /// Number of core electrons replaced by ECP on each atom (`0` for all-electron atoms).
    pub core_electrons: Vec<u8>,
    /// Active scalar ECP operators on atoms carrying an effective core potential.
    pub ecp_potentials: Vec<AtomEcp>,
}

impl BasisSet {
    pub const SUPPORTED_NAMES: &'static [&'static str] = &[
        "sto-3g",
        "6-31g",
        "6-31g(d)",
        "6-31g(d,p)",
        "lanl2dz",
        "lanl2dzdp",
    ];

    pub fn canonical_name(name: &str) -> Option<&'static str> {
        match name.trim().to_ascii_lowercase().as_str() {
            "sto-3g" | "sto3g" => Some("sto-3g"),
            "6-31g" | "631g" => Some("6-31g"),
            "6-31g(d)" | "6-31g*" | "631gd" | "631g*" | "6-31gd" => Some("6-31g(d)"),
            "6-31g(d,p)" | "6-31g**" | "631gdp" | "631g**" | "6-31gdp" => Some("6-31g(d,p)"),
            "lanl2dz" | "lanl-2dz" => Some("lanl2dz"),
            "lanl2dzdp" | "lanl2dz(d,p)" | "lanl2dz*" => Some("lanl2dzdp"),
            _ => None,
        }
    }

    pub fn from_name(name: &str, mol: &Molecule) -> Result<Self, CalcError> {
        match Self::canonical_name(name) {
            Some("sto-3g") => Self::sto_3g(mol),
            Some("6-31g") => Self::basis_6_31g(mol),
            Some("6-31g(d)") => Self::basis_6_31gd(mol),
            Some("6-31g(d,p)") => Self::basis_6_31gdp(mol),
            Some("lanl2dz") => Self::lanl2dz(mol),
            Some("lanl2dzdp") => Self::lanl2dzdp(mol),
            _ => Err(CalcError::UnsupportedBasis(format!(
                "Unsupported basis set: '{}'",
                name.trim()
            ))),
        }
    }

    pub fn sto_3g(mol: &Molecule) -> Result<Self, CalcError> {
        let mut bs = Self::from_json_str(STO_3G_JSON, mol)?;
        if bs.name.is_empty() {
            bs.name = "sto-3g".to_string();
        }
        Ok(bs)
    }

    pub fn basis_6_31g(mol: &Molecule) -> Result<Self, CalcError> {
        Self::from_json_str(BASIS_631G_JSON, mol)
    }

    pub fn basis_6_31gd(mol: &Molecule) -> Result<Self, CalcError> {
        Self::from_json_str(BASIS_631GD_JSON, mol)
    }

    pub fn basis_6_31gdp(mol: &Molecule) -> Result<Self, CalcError> {
        Self::from_json_str(BASIS_631GDP_JSON, mol)
    }

    pub fn lanl2dz(mol: &Molecule) -> Result<Self, CalcError> {
        Self::from_json_str(LANL2DZ_JSON, mol)
    }

    pub fn lanl2dzdp(mol: &Molecule) -> Result<Self, CalcError> {
        Self::from_json_str(LANL2DZDP_JSON, mol)
    }

    pub fn from_json_str(json_str: &str, mol: &Molecule) -> Result<Self, CalcError> {
        let raw: RawBasisFile = serde_json::from_str(json_str)
            .map_err(|e| CalcError::Internal(format!("Basis JSON parse error: {e}")))?;
        Self::build(raw, mol)
    }

    /// Returns `true` if at least one atom in this basis carries an ECP.
    #[inline]
    pub fn has_ecp(&self) -> bool {
        !self.ecp_potentials.is_empty()
    }

    /// Effective nuclear charge `Z_eff = Z - N_core` for `atom_idx` in `mol`.
    #[inline]
    pub fn effective_nuclear_charge(&self, mol: &Molecule, atom_idx: usize) -> f64 {
        let z = mol.atoms[atom_idx].z as i32;
        let n_core = self
            .core_electrons
            .get(atom_idx)
            .copied()
            .unwrap_or(0) as i32;
        (z - n_core) as f64
    }

    /// Nuclear-nuclear repulsion energy using `Z_eff(A) * Z_eff(B) / R_AB`
    /// (identical to `mol.nuclear_repulsion_energy()` for all-electron basis sets).
    pub fn nuclear_repulsion_energy(&self, mol: &Molecule) -> f64 {
        let mut e_nuc = 0.0;
        for i in 0..mol.atoms.len() {
            let z_i = self.effective_nuclear_charge(mol, i);
            for j in (i + 1)..mol.atoms.len() {
                let z_j = self.effective_nuclear_charge(mol, j);
                let dx = mol.atoms[i].position_bohr[0] - mol.atoms[j].position_bohr[0];
                let dy = mol.atoms[i].position_bohr[1] - mol.atoms[j].position_bohr[1];
                let dz = mol.atoms[i].position_bohr[2] - mol.atoms[j].position_bohr[2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                if r > 1e-12 {
                    e_nuc += z_i * z_j / r;
                }
            }
        }
        e_nuc
    }

    fn validate_geometry_and_valence(&self, mol: &Molecule) -> Result<(usize, usize), CalcError> {
        if mol.atoms.is_empty() {
            return Err(CalcError::InvalidMolecule(
                "Molecule must contain at least one atom".into(),
            ));
        }

        for (idx, atom) in mol.atoms.iter().enumerate() {
            if atom.z == 0 || atom.symbol.trim().is_empty() {
                return Err(CalcError::InvalidMolecule(format!(
                    "Atom at index {idx} has invalid symbol '{}' or atomic number {}",
                    atom.symbol, atom.z
                )));
            }
            let expected_z = atomic_number_from_symbol(&atom.symbol)?;
            if atom.z != expected_z {
                return Err(CalcError::InvalidMolecule(format!(
                    "Atom at index {idx} has mismatched symbol '{}' (expected Z={expected_z}, got Z={})",
                    atom.symbol, atom.z
                )));
            }
            if !atom.position_bohr.iter().all(|c| c.is_finite()) {
                return Err(CalcError::InvalidMolecule(format!(
                    "Atom {idx} ('{}') has non-finite coordinates",
                    atom.symbol
                )));
            }
        }

        for i in 0..mol.atoms.len() {
            for j in (i + 1)..mol.atoms.len() {
                let dx = mol.atoms[i].position_bohr[0] - mol.atoms[j].position_bohr[0];
                let dy = mol.atoms[i].position_bohr[1] - mol.atoms[j].position_bohr[1];
                let dz = mol.atoms[i].position_bohr[2] - mol.atoms[j].position_bohr[2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                if !r.is_finite() || r < 1e-3 {
                    return Err(CalcError::InvalidMolecule(format!(
                        "Atoms {i} and {j} are too close (r = {r:.3e} Bohr < 1e-3 Bohr)"
                    )));
                }
            }
        }

        if mol.multiplicity < 1 {
            return Err(CalcError::InvalidMolecule(
                "Multiplicity must be >= 1".into(),
            ));
        }

        let total_z_eff: i32 = (0..mol.atoms.len())
            .map(|i| self.effective_nuclear_charge(mol, i) as i32)
            .sum();
        let ne = total_z_eff - mol.charge;
        if ne <= 0 {
            return Err(CalcError::InvalidMolecule(format!(
                "Non-positive valence electron count: {ne}"
            )));
        }
        let ne_usize = ne as usize;
        let unpaired = mol.multiplicity - 1;
        if unpaired > ne_usize || (ne_usize - unpaired) % 2 != 0 {
            return Err(CalcError::InvalidMolecule(format!(
                "Incompatible valence electron count ({ne_usize}) and multiplicity ({})",
                mol.multiplicity
            )));
        }
        let n_beta = (ne_usize - unpaired) / 2;
        let n_alpha = n_beta + unpaired;
        Ok((n_alpha, n_beta))
    }

    /// Validates closed-shell RHF compatibility accounting for ECP core electrons.
    pub fn validate_rhf_valence(&self, mol: &Molecule) -> Result<(usize, usize), CalcError> {
        if mol.multiplicity != 1 {
            return Err(CalcError::UnsupportedMethod(format!(
                "RHF requires singlet multiplicity=1, got multiplicity={}",
                mol.multiplicity
            )));
        }
        let (n_alpha, n_beta) = self.validate_geometry_and_valence(mol)?;
        if n_alpha != n_beta || n_alpha == 0 {
            return Err(CalcError::UnsupportedMethod(
                "RHF requires a closed-shell molecule with positive even electron count".into(),
            ));
        }
        Ok((n_alpha, n_beta))
    }

    /// Validates open-shell UHF compatibility accounting for ECP core electrons.
    pub fn validate_uhf_valence(&self, mol: &Molecule) -> Result<(usize, usize), CalcError> {
        let (n_alpha, n_beta) = self.validate_geometry_and_valence(mol)?;
        if n_alpha == 0 {
            return Err(CalcError::InvalidMolecule("Molecule has no valence electrons".into()));
        }
        Ok((n_alpha, n_beta))
    }

    fn build(raw: RawBasisFile, mol: &Molecule) -> Result<Self, CalcError> {
        if mol.atoms.is_empty() {
            return Err(CalcError::InvalidMolecule(
                "Cannot build basis set for empty molecule".into(),
            ));
        }

        let mut shells = Vec::new();
        let mut core_electrons = Vec::with_capacity(mol.atoms.len());
        let mut ecp_potentials = Vec::new();
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

            if let Some(raw_ecp) = raw.ecp.get(&atom.symbol) {
                if raw_ecp.core_electrons >= atom.z {
                    return Err(CalcError::InvalidMolecule(format!(
                        "ECP core_electrons ({}) >= atomic number ({}) on '{}'",
                        raw_ecp.core_electrons, atom.z, atom.symbol
                    )));
                }
                core_electrons.push(raw_ecp.core_electrons);
                let local_terms = raw_ecp
                    .local
                    .iter()
                    .map(|t| EcpTerm {
                        r_power: t.r_power,
                        exponent: t.exponent,
                        coeff: t.coeff,
                    })
                    .collect();
                let semilocal_channels = raw_ecp
                    .semilocal
                    .iter()
                    .map(|ch| EcpChannel {
                        angular_momentum: ch.angular_momentum,
                        terms: ch
                            .terms
                            .iter()
                            .map(|t| EcpTerm {
                                r_power: t.r_power,
                                exponent: t.exponent,
                                coeff: t.coeff,
                            })
                            .collect(),
                    })
                    .collect();
                ecp_potentials.push(AtomEcp {
                    atom_index,
                    center: atom.position_bohr,
                    core_electrons: raw_ecp.core_electrons,
                    l_max: raw_ecp.l_max,
                    local_terms,
                    semilocal_channels,
                });
            } else {
                core_electrons.push(0);
            }
        }

        Ok(Self {
            name: raw.name,
            shells,
            n_ao: ao_offset,
            core_electrons,
            ecp_potentials,
        })
    }
}
