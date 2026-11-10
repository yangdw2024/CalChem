use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const ANGSTROM_TO_BOHR: f64 = 1.8897261245650618;
pub const BOHR_TO_ANGSTROM: f64 = 1.0 / ANGSTROM_TO_BOHR;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum CalcError {
    #[error("Invalid molecule specification: {0}")]
    InvalidMolecule(String),
    #[error("Unsupported basis set or element: {0}")]
    UnsupportedBasis(String),
    #[error("Unsupported method or spin state: {0}")]
    UnsupportedMethod(String),
    #[error("Linear algebra failure: {0}")]
    LinearAlgebra(String),
    #[error("SCF failed to converge in {iterations} iterations (dE={delta_e:.3e}, rmsD={rms_d:.3e})")]
    NotConverged {
        iterations: usize,
        delta_e: f64,
        rms_d: f64,
    },
    #[error("Internal error: {0}")]
    Internal(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Atom {
    pub symbol: String,
    pub z: u8,
    pub position_bohr: [f64; 3],
}

impl Atom {
    pub fn from_angstrom(symbol: &str, pos_ang: [f64; 3]) -> Result<Self, CalcError> {
        let z = atomic_number_from_symbol(symbol)?;
        if !pos_ang.iter().all(|v| v.is_finite()) {
            return Err(CalcError::InvalidMolecule(format!(
                "Atom '{}' has non-finite coordinates",
                symbol
            )));
        }
        Ok(Self {
            symbol: symbol.trim().to_string(),
            z,
            position_bohr: [
                pos_ang[0] * ANGSTROM_TO_BOHR,
                pos_ang[1] * ANGSTROM_TO_BOHR,
                pos_ang[2] * ANGSTROM_TO_BOHR,
            ],
        })
    }

    pub fn from_bohr(symbol: &str, pos_bohr: [f64; 3]) -> Result<Self, CalcError> {
        let z = atomic_number_from_symbol(symbol)?;
        if !pos_bohr.iter().all(|v| v.is_finite()) {
            return Err(CalcError::InvalidMolecule(format!(
                "Atom '{}' has non-finite coordinates",
                symbol
            )));
        }
        Ok(Self {
            symbol: symbol.trim().to_string(),
            z,
            position_bohr: pos_bohr,
        })
    }
}

const PERIODIC_TABLE_SYMBOLS: [&str; 86] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
];

pub fn atomic_number_from_symbol(sym: &str) -> Result<u8, CalcError> {
    let clean = sym.trim();
    if clean.is_empty() {
        return Err(CalcError::InvalidMolecule("Empty element symbol".into()));
    }
    let mut chars = clean.chars();
    let first = chars.next().unwrap().to_ascii_uppercase();
    let rest: String = chars.map(|c| c.to_ascii_lowercase()).collect();
    let norm = format!("{}{}", first, rest);
    for (idx, &known) in PERIODIC_TABLE_SYMBOLS.iter().enumerate() {
        if known == norm {
            return Ok((idx + 1) as u8);
        }
    }
    Err(CalcError::InvalidMolecule(format!(
        "Unsupported element symbol: {}",
        sym
    )))
}

pub fn symbol_from_atomic_number(z: u8) -> Result<String, CalcError> {
    if z == 0 || (z as usize) > PERIODIC_TABLE_SYMBOLS.len() {
        return Err(CalcError::InvalidMolecule(format!(
            "Unsupported atomic number: {z}"
        )));
    }
    Ok(PERIODIC_TABLE_SYMBOLS[(z - 1) as usize].to_string())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Molecule {
    pub atoms: Vec<Atom>,
    pub charge: i32,
    pub multiplicity: usize,
}

impl Molecule {
    pub fn new(atoms: Vec<Atom>, charge: i32, multiplicity: usize) -> Self {
        Self {
            atoms,
            charge,
            multiplicity,
        }
    }

    pub fn nelectron(&self) -> Result<usize, CalcError> {
        let total_z: i32 = self.atoms.iter().map(|a| a.z as i32).sum();
        let ne = total_z - self.charge;
        if ne <= 0 {
            return Err(CalcError::InvalidMolecule(format!(
                "Non-positive electron count: {}",
                ne
            )));
        }
        Ok(ne as usize)
    }

    pub fn n_alpha_beta(&self) -> Result<(usize, usize), CalcError> {
        let ne = self.nelectron()?;
        if self.multiplicity < 1 {
            return Err(CalcError::InvalidMolecule("Multiplicity must be >= 1".into()));
        }
        let unpaired = self.multiplicity - 1;
        if unpaired > ne || (ne - unpaired) % 2 != 0 {
            return Err(CalcError::InvalidMolecule(format!(
                "Incompatible electron count ({}) and multiplicity ({})",
                ne, self.multiplicity
            )));
        }
        let n_beta = (ne - unpaired) / 2;
        let n_alpha = n_beta + unpaired;
        Ok((n_alpha, n_beta))
    }

    /// Validates that the molecule is well-formed and compatible with closed-shell RHF.
    /// Returns `(n_alpha, n_beta)` where `n_alpha == n_beta > 0`.
    pub fn validate_rhf(&self) -> Result<(usize, usize), CalcError> {
        if self.atoms.is_empty() {
            return Err(CalcError::InvalidMolecule(
                "Molecule must contain at least one atom".into(),
            ));
        }

        for (idx, atom) in self.atoms.iter().enumerate() {
            if atom.z == 0 || atom.symbol.trim().is_empty() {
                return Err(CalcError::InvalidMolecule(format!(
                    "Atom at index {} has invalid symbol '{}' or atomic number {}",
                    idx, atom.symbol, atom.z
                )));
            }
            let expected_z = atomic_number_from_symbol(&atom.symbol)?;
            if atom.z != expected_z {
                return Err(CalcError::InvalidMolecule(format!(
                    "Atom at index {} has mismatched symbol '{}' (expected Z={}, got Z={})",
                    idx, atom.symbol, expected_z, atom.z
                )));
            }
            if !atom.position_bohr.iter().all(|c| c.is_finite()) {
                return Err(CalcError::InvalidMolecule(format!(
                    "Atom {} ('{}') has non-finite coordinates",
                    idx, atom.symbol
                )));
            }
        }

        for i in 0..self.atoms.len() {
            for j in (i + 1)..self.atoms.len() {
                let dx = self.atoms[i].position_bohr[0] - self.atoms[j].position_bohr[0];
                let dy = self.atoms[i].position_bohr[1] - self.atoms[j].position_bohr[1];
                let dz = self.atoms[i].position_bohr[2] - self.atoms[j].position_bohr[2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                if !r.is_finite() || r < 1e-3 {
                    return Err(CalcError::InvalidMolecule(format!(
                        "Atoms {} and {} are too close (r = {:.3e} Bohr < 1e-3 Bohr)",
                        i, j, r
                    )));
                }
            }
        }

        if self.multiplicity != 1 {
            return Err(CalcError::UnsupportedMethod(format!(
                "RHF requires singlet multiplicity=1, got multiplicity={}",
                self.multiplicity
            )));
        }

        let (n_alpha, n_beta) = self.n_alpha_beta()?;
        if n_alpha != n_beta || n_alpha == 0 {
            return Err(CalcError::UnsupportedMethod(
                "RHF requires a closed-shell molecule with positive even electron count".into(),
            ));
        }

        Ok((n_alpha, n_beta))
    }

    pub fn nuclear_repulsion_energy(&self) -> f64 {
        let mut e_nuc = 0.0;
        for i in 0..self.atoms.len() {
            for j in (i + 1)..self.atoms.len() {
                let dx = self.atoms[i].position_bohr[0] - self.atoms[j].position_bohr[0];
                let dy = self.atoms[i].position_bohr[1] - self.atoms[j].position_bohr[1];
                let dz = self.atoms[i].position_bohr[2] - self.atoms[j].position_bohr[2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                if r > 1e-12 {
                    e_nuc += (self.atoms[i].z as f64) * (self.atoms[j].z as f64) / r;
                }
            }
        }
        e_nuc
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasisFunctionMeta {
    pub atom_index: usize,
    pub shell_index: usize,
    pub angular_lmn: [u8; 3],
    pub center_bohr: [f64; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellMeta {
    pub atom_index: usize,
    pub angular_momentum: u8,
    pub is_spherical: bool,
    pub exponents: Vec<f64>,
    pub coefficients: Vec<f64>,
    pub center_bohr: [f64; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScfOptions {
    pub max_iter: usize,
    pub conv_energy: f64,
    pub conv_density: f64,
    pub use_diis: bool,
    pub diis_max_vecs: usize,
    pub level_shift: f64,
    pub damping: f64,
}

impl Default for ScfOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            conv_energy: 1e-10,
            conv_density: 1e-8,
            use_diis: true,
            diis_max_vecs: 6,
            level_shift: 0.0,
            damping: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScfDiagnostics {
    pub iterations: usize,
    pub energy_history: Vec<f64>,
    pub rms_density_history: Vec<f64>,
    pub diis_error_history: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScfWavefunction {
    pub converged: bool,
    pub iterations: usize,
    pub total_energy_hartree: f64,
    pub electronic_energy_hartree: f64,
    pub nuclear_repulsion_hartree: f64,
    pub n_ao: usize,
    pub n_mo: usize,
    pub mo_energies: Vec<f64>,
    pub mo_occupations: Vec<f64>,
    /// Molecular orbital coefficient matrix (flattened row-major, index `mu * n_mo + i`).
    pub mo_coefficients: Vec<f64>,
    /// Alpha spin density matrix (flattened row-major).
    /// NOTE: In RHF (closed-shell), this field stores the *spin-summed* total density matrix:
    /// `P = 2 * C_occ * C_occ^T`. Do not multiply by 2 again.
    pub density_alpha: Vec<f64>,
    /// Beta spin density matrix (flattened row-major). None for RHF.
    pub density_beta: Option<Vec<f64>>,
    pub fock_alpha: Vec<f64>,
    pub overlap_matrix: Vec<f64>,
    pub h_core_matrix: Vec<f64>,
    pub diagnostics: ScfDiagnostics,
}

/// Unrestricted Hartree-Fock wavefunction (separate alpha / beta spin orbitals).
///
/// All matrices are flattened row-major `n_ao x n_ao`. Densities are *per-spin*
/// (`P^a = C^a_occ C^a_occ^T`, `P^b = C^b_occ C^b_occ^T`), never spin-summed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UhfWavefunction {
    pub converged: bool,
    pub iterations: usize,
    pub total_energy_hartree: f64,
    pub electronic_energy_hartree: f64,
    pub nuclear_repulsion_hartree: f64,
    pub n_ao: usize,
    pub n_alpha: usize,
    pub n_beta: usize,
    pub mo_energies_alpha: Vec<f64>,
    pub mo_energies_beta: Vec<f64>,
    /// Alpha MO coefficients, index `mu * n_ao + i`.
    pub mo_coefficients_alpha: Vec<f64>,
    /// Beta MO coefficients, index `mu * n_ao + i`.
    pub mo_coefficients_beta: Vec<f64>,
    pub density_alpha: Vec<f64>,
    pub density_beta: Vec<f64>,
    /// Expectation value `<S^2>`; exact value is `S(S+1)`, excess is spin contamination.
    pub s_squared: f64,
    pub overlap_matrix: Vec<f64>,
    pub diagnostics: ScfDiagnostics,
}

impl UhfWavefunction {
    /// Spin-summed density `P = P^a + P^b`.
    pub fn total_density(&self) -> Vec<f64> {
        self.density_alpha
            .iter()
            .zip(&self.density_beta)
            .map(|(a, b)| a + b)
            .collect()
    }

    /// Highest occupied spin-orbital energy across both spins (SOMO for doublets).
    pub fn homo_energy(&self) -> Option<f64> {
        let a = self.mo_energies_alpha.iter().take(self.n_alpha).cloned();
        let b = self.mo_energies_beta.iter().take(self.n_beta).cloned();
        a.chain(b).fold(None, |acc: Option<f64>, e| Some(acc.map_or(e, |m| m.max(e))))
    }

    /// Lowest unoccupied spin-orbital energy across both spins.
    pub fn lumo_energy(&self) -> Option<f64> {
        let a = self.mo_energies_alpha.iter().skip(self.n_alpha).cloned();
        let b = self.mo_energies_beta.iter().skip(self.n_beta).cloned();
        a.chain(b).fold(None, |acc: Option<f64>, e| Some(acc.map_or(e, |m| m.min(e))))
    }
}

impl Molecule {
    /// Validates the molecule for an unrestricted (UHF) calculation.
    /// Returns `(n_alpha, n_beta)`; any multiplicity consistent with the electron count is allowed.
    pub fn validate_uhf(&self) -> Result<(usize, usize), CalcError> {
        if self.atoms.is_empty() {
            return Err(CalcError::InvalidMolecule("Molecule has no atoms".into()));
        }
        for i in 0..self.atoms.len() {
            for j in (i + 1)..self.atoms.len() {
                let a = self.atoms[i].position_bohr;
                let b = self.atoms[j].position_bohr;
                let d2 = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
                if d2.sqrt() < 1e-3 {
                    return Err(CalcError::InvalidMolecule(format!(
                        "Atoms {i} and {j} are coincident (distance {:.3e} Bohr)",
                        d2.sqrt()
                    )));
                }
            }
        }
        let (n_alpha, n_beta) = self.n_alpha_beta()?;
        if n_alpha == 0 {
            return Err(CalcError::InvalidMolecule("Molecule has no electrons".into()));
        }
        Ok((n_alpha, n_beta))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nuclear_repulsion_h2_he_h2o() {
        let h2 = Molecule::new(
            vec![
                Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("H", [0.0, 0.0, 1.4]).unwrap(),
            ],
            0,
            1,
        );
        assert!((h2.nuclear_repulsion_energy() - (1.0 / 1.4)).abs() < 1e-15);

        let he = Molecule::new(vec![Atom::from_bohr("He", [0.0, 0.0, 0.0]).unwrap()], 0, 1);
        assert_eq!(he.nuclear_repulsion_energy(), 0.0);

        let h2o = Molecule::new(
            vec![
                Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
                Atom::from_bohr("H", [1.112, -0.716, 0.0]).unwrap(),
            ],
            0,
            1,
        );
        let r_oh1 = 1.432_f64;
        let r_oh2 = (1.112_f64 * 1.112 + 0.716 * 0.716).sqrt();
        let r_hh = (1.112_f64 * 1.112 + (1.432 + 0.716) * (1.432 + 0.716)).sqrt();
        let expected = 8.0 / r_oh1 + 8.0 / r_oh2 + 1.0 / r_hh;
        assert!((h2o.nuclear_repulsion_energy() - expected).abs() < 1e-14);
    }

    #[test]
    fn test_validate_rhf_checks() {
        // Empty molecule
        let empty = Molecule::new(vec![], 0, 1);
        assert!(matches!(
            empty.validate_rhf(),
            Err(CalcError::InvalidMolecule(_))
        ));

        // Coincident atoms
        let coincident = Molecule::new(
            vec![
                Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("H", [0.0, 0.0, 1e-5]).unwrap(),
            ],
            0,
            1,
        );
        assert!(matches!(
            coincident.validate_rhf(),
            Err(CalcError::InvalidMolecule(_))
        ));

        // Odd electron count with singlet multiplicity
        let h_atom = Molecule::new(vec![Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap()], 0, 1);
        assert!(matches!(
            h_atom.validate_rhf(),
            Err(CalcError::InvalidMolecule(_))
        ));

        // Triplet multiplicity
        let h2_triplet = Molecule::new(
            vec![
                Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("H", [0.0, 0.0, 1.4]).unwrap(),
            ],
            0,
            3,
        );
        assert!(matches!(
            h2_triplet.validate_rhf(),
            Err(CalcError::UnsupportedMethod(_))
        ));

        // Valid closed-shell H2O
        let h2o = Molecule::new(
            vec![
                Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
                Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
                Atom::from_bohr("H", [1.112, -0.716, 0.0]).unwrap(),
            ],
            0,
            1,
        );
        assert_eq!(h2o.validate_rhf().unwrap(), (5, 5));
    }
}
