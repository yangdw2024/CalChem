//! Continuum dielectric solvent models, solvent database, and dielectric response functions.

use serde::{Deserialize, Serialize};

/// Supported continuum solvation formalisms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolventModel {
    /// Conductor-like Polarizable Continuum Model (C-PCM): `f(eps) = (eps - 1) / eps`.
    Cpcm,
    /// Conductor-like Screening Model (COSMO): `f(eps) = (eps - 1) / (eps + 0.5)`.
    Cosmo,
    /// Integral Equation Formalism PCM in conductor limit: `f(eps) = (eps - 1) / eps`.
    IefPcm,
    /// Dipolar spherical reaction field (Onsager model): `f(eps) = 2*(eps - 1) / (2*eps + 1)`.
    Onsager,
}

impl Default for SolventModel {
    fn default() -> Self {
        Self::Cpcm
    }
}

impl SolventModel {
    /// Evaluates the dielectric screening prefactor `f(epsilon)`.
    pub fn dielectric_factor(self, epsilon: f64) -> f64 {
        if (epsilon - 1.0).abs() < 1e-12 {
            return 0.0;
        }
        match self {
            Self::Cpcm | Self::IefPcm => (epsilon - 1.0) / epsilon,
            Self::Cosmo => (epsilon - 1.0) / (epsilon + 0.5),
            Self::Onsager => 2.0 * (epsilon - 1.0) / (2.0 * epsilon + 1.0),
        }
    }

    /// Evaluates the optical (fast) dielectric response factor `f(eps_infty)` where `eps_infty = n^2`.
    pub fn optical_dielectric_factor(self, refractive_index: f64) -> f64 {
        let eps_inf = refractive_index * refractive_index;
        self.dielectric_factor(eps_inf)
    }
}

/// Physical parameters of a continuum solvent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Solvent {
    /// Canonical solvent identifier (e.g. `"water"`, `"chlorobenzene"`).
    pub name: String,
    /// Static dielectric constant `epsilon` (relative permittivity).
    pub epsilon: f64,
    /// Optical refractive index `n` (defines high-frequency permittivity `eps_inf = n^2`).
    pub refractive_index: f64,
    /// Macroscopic surface tension `gamma` (mN/m = dyn/cm).
    pub surface_tension_m_n_per_m: f64,
    /// Effective solvent probe radius (Angstrom).
    pub probe_radius_angstrom: f64,
}

impl Default for Solvent {
    fn default() -> Self {
        Self::water()
    }
}

impl Solvent {
    /// Conversion constant from mN/m (mJ/m^2) to kcal / (mol * Angstrom^2).
    pub const MN_M_TO_KCAL_MOL_ANGSTROM2: f64 = 0.001439326;

    /// Creates a custom solvent with specified properties.
    pub fn custom(
        name: impl Into<String>,
        epsilon: f64,
        refractive_index: f64,
        surface_tension_m_n_per_m: f64,
        probe_radius_angstrom: f64,
    ) -> Self {
        Self {
            name: name.into(),
            epsilon,
            refractive_index,
            surface_tension_m_n_per_m,
            probe_radius_angstrom,
        }
    }

    /// Water (H2O, standard aqueous medium).
    pub fn water() -> Self {
        Self::custom("water", 78.3553, 1.3330, 72.8, 1.385)
    }

    /// Acetonitrile (CH3CN, polar aprotic solvent).
    pub fn acetonitrile() -> Self {
        Self::custom("acetonitrile", 37.5, 1.3441, 29.3, 2.19)
    }

    /// Methanol (CH3OH, protic solvent).
    pub fn methanol() -> Self {
        Self::custom("methanol", 32.61, 1.3288, 22.5, 2.05)
    }

    /// Ethanol (CH3CH2OH).
    pub fn ethanol() -> Self {
        Self::custom("ethanol", 24.55, 1.3611, 22.4, 2.22)
    }

    /// Acetone (CH3COCH3).
    pub fn acetone() -> Self {
        Self::custom("acetone", 20.7, 1.3586, 23.5, 2.38)
    }

    /// Dichloromethane (DCM, CH2Cl2).
    pub fn dichloromethane() -> Self {
        Self::custom("dichloromethane", 8.93, 1.4244, 28.1, 2.33)
    }

    /// Tetrahydrofuran (THF, C4H8O).
    pub fn tetrahydrofuran() -> Self {
        Self::custom("tetrahydrofuran", 7.58, 1.4072, 26.4, 2.52)
    }

    /// Chlorobenzene (C6H5Cl, standard OPV thin-film processing solvent).
    pub fn chlorobenzene() -> Self {
        Self::custom("chlorobenzene", 5.62, 1.5248, 33.6, 2.72)
    }

    /// Chloroform (CHCl3).
    pub fn chloroform() -> Self {
        Self::custom("chloroform", 4.806, 1.4459, 27.2, 2.52)
    }

    /// Toluene (C7H8, aromatic solvent).
    pub fn toluene() -> Self {
        Self::custom("toluene", 2.379, 1.4969, 28.5, 2.82)
    }

    /// Benzene (C6H6, nonpolar aromatic solvent).
    pub fn benzene() -> Self {
        Self::custom("benzene", 2.2706, 1.5011, 28.9, 2.63)
    }

    /// Hexane (C6H14, aliphatic alkane solvent).
    pub fn hexane() -> Self {
        Self::custom("hexane", 1.88, 1.3749, 18.4, 3.00)
    }

    /// Cyclohexane (C6H12).
    pub fn cyclohexane() -> Self {
        Self::custom("cyclohexane", 2.02, 1.4262, 24.9, 2.81)
    }

    /// Gas phase / vacuum (epsilon = 1.0, zero screening).
    pub fn gas() -> Self {
        Self::custom("gas", 1.0, 1.0, 0.0, 0.0)
    }

    /// Resolves a solvent by case-insensitive name or alias.
    pub fn from_name(name: &str) -> Option<Self> {
        let n = name.trim().to_ascii_lowercase();
        let key = n.replace(['-', '_', ' '], "");
        match key.as_str() {
            "water" | "h2o" | "aqueous" => Some(Self::water()),
            "acetonitrile" | "mecn" | "ch3cn" => Some(Self::acetonitrile()),
            "methanol" | "meoh" | "ch3oh" => Some(Self::methanol()),
            "ethanol" | "etoh" | "ch3ch2oh" => Some(Self::ethanol()),
            "acetone" | "ch3coch3" => Some(Self::acetone()),
            "dichloromethane" | "dcm" | "ch2cl2" => Some(Self::dichloromethane()),
            "tetrahydrofuran" | "thf" => Some(Self::tetrahydrofuran()),
            "chlorobenzene" | "cb" | "phcl" => Some(Self::chlorobenzene()),
            "chloroform" | "chcl3" => Some(Self::chloroform()),
            "toluene" | "phme" => Some(Self::toluene()),
            "benzene" | "phh" => Some(Self::benzene()),
            "hexane" | "nhexane" => Some(Self::hexane()),
            "cyclohexane" => Some(Self::cyclohexane()),
            "gas" | "vacuum" | "none" => Some(Self::gas()),
            _ => None,
        }
    }

    /// Macroscopic surface tension converted to kcal / (mol * Angstrom^2).
    pub fn surface_tension_kcal_mol_angstrom2(&self) -> f64 {
        self.surface_tension_m_n_per_m * Self::MN_M_TO_KCAL_MOL_ANGSTROM2
    }
}
