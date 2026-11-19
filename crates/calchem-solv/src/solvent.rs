//! Continuum dielectric solvent models, solvent database, and dielectric response functions.

use serde::{Deserialize, Serialize};

/// Supported continuum solvation formalisms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolventModel {
    /// Conductor-like Polarizable Continuum Model (C-PCM): `f(eps) = (eps - 1) / eps`.
    Cpcm,
    /// Conductor-like Screening Model (COSMO): `f(eps) = (eps - 1) / (eps + 0.5)`.
    Cosmo,
    /// Integral Equation Formalism PCM (`K q = R V`).
    IefPcm,
    /// Dipolar spherical reaction field (Onsager model): `f(eps) = 2*(eps - 1) / (2*eps + 1)`.
    Onsager,
    /// Universal Solvation Model based on Density (SMD, Marenich-Cramer-Truhlar 2009):
    /// IEF-PCM electrostatics on SMD intrinsic radii + CDS non-electrostatic term.
    Smd,
}

impl Default for SolventModel {
    fn default() -> Self {
        Self::Cpcm
    }
}

impl SolventModel {
    pub fn validate_native(self) -> Result<(), calchem_model::CalcError> {
        // All five continuum solvation models (C-PCM, COSMO, IEF-PCM, Onsager, SMD) are natively supported.
        let _ = self;
        Ok(())
    }
    /// Evaluates the dielectric screening prefactor `f(epsilon)`.
    pub fn dielectric_factor(self, epsilon: f64) -> f64 {
        if (epsilon - 1.0).abs() < 1e-12 {
            return 0.0;
        }
        match self {
            Self::Cpcm | Self::IefPcm | Self::Smd => (epsilon - 1.0) / epsilon,
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

/// Minnesota Solvent Descriptor Database parameters for SMD CDS non-electrostatic energy.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct SmdDescriptors {
    /// Refractive index `n` at 293 K.
    pub n: f64,
    /// Abraham hydrogen-bond acidity `alpha`.
    pub alpha: f64,
    /// Abraham hydrogen-bond basicity `beta`.
    pub beta: f64,
    /// Macroscopic surface tension `gamma` in `cal / (mol * Angstrom^2)`.
    pub gamma_cal: f64,
    /// Aromaticity fraction `phi` (aromatic carbons / non-H atoms).
    pub phi: f64,
    /// Electronegative halogenicity fraction `psi` (F, Cl, Br atoms / non-H atoms).
    pub psi: f64,
    /// True if aqueous SMD parameterization (`icds = 1`) should be used.
    pub is_water: bool,
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
    /// Minnesota SMD descriptors (`n, alpha, beta, gamma_cal, phi, psi`).
    #[serde(default)]
    pub smd: SmdDescriptors,
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
            smd: SmdDescriptors {
                n: refractive_index,
                gamma_cal: surface_tension_m_n_per_m * Self::MN_M_TO_KCAL_MOL_ANGSTROM2 * 1000.0,
                ..Default::default()
            },
        }
    }

    fn with_smd(mut self, smd: SmdDescriptors) -> Self {
        self.smd = smd;
        self
    }

    /// Water (H2O, standard aqueous medium).
    pub fn water() -> Self {
        Self::custom("water", 78.3553, 1.3328, 72.8, 1.385).with_smd(SmdDescriptors {
            n: 1.3328,
            alpha: 0.82,
            beta: 0.35,
            gamma_cal: 103.62,
            phi: 0.0,
            psi: 0.0,
            is_water: true,
        })
    }

    /// Acetonitrile (CH3CN, polar aprotic solvent).
    pub fn acetonitrile() -> Self {
        Self::custom("acetonitrile", 35.688, 1.3442, 29.3, 2.19).with_smd(SmdDescriptors {
            n: 1.3442,
            alpha: 0.07,
            beta: 0.32,
            gamma_cal: 41.25,
            phi: 0.0,
            psi: 0.0,
            is_water: false,
        })
    }

    /// Methanol (CH3OH, protic solvent).
    pub fn methanol() -> Self {
        Self::custom("methanol", 32.613, 1.3288, 22.5, 2.05).with_smd(SmdDescriptors {
            n: 1.3288,
            alpha: 0.43,
            beta: 0.47,
            gamma_cal: 31.77,
            phi: 0.0,
            psi: 0.0,
            is_water: false,
        })
    }

    /// Ethanol (CH3CH2OH).
    pub fn ethanol() -> Self {
        Self::custom("ethanol", 24.852, 1.3611, 22.4, 2.22).with_smd(SmdDescriptors {
            n: 1.3611,
            alpha: 0.37,
            beta: 0.48,
            gamma_cal: 31.62,
            phi: 0.0,
            psi: 0.0,
            is_water: false,
        })
    }

    /// Acetone (CH3COCH3).
    pub fn acetone() -> Self {
        Self::custom("acetone", 20.493, 1.3588, 23.5, 2.38).with_smd(SmdDescriptors {
            n: 1.3588,
            alpha: 0.04,
            beta: 0.49,
            gamma_cal: 33.77,
            phi: 0.0,
            psi: 0.0,
            is_water: false,
        })
    }

    /// Dichloromethane (DCM, CH2Cl2).
    pub fn dichloromethane() -> Self {
        Self::custom("dichloromethane", 8.93, 1.4242, 28.1, 2.33).with_smd(SmdDescriptors {
            n: 1.4242,
            alpha: 0.10,
            beta: 0.05,
            gamma_cal: 39.15,
            phi: 0.0,
            psi: 0.667,
            is_water: false,
        })
    }

    /// Tetrahydrofuran (THF, C4H8O).
    pub fn tetrahydrofuran() -> Self {
        Self::custom("tetrahydrofuran", 7.4257, 1.4050, 26.4, 2.52).with_smd(SmdDescriptors {
            n: 1.4050,
            alpha: 0.0,
            beta: 0.48,
            gamma_cal: 39.44,
            phi: 0.0,
            psi: 0.0,
            is_water: false,
        })
    }

    /// Chlorobenzene (C6H5Cl, standard OPV thin-film processing solvent).
    pub fn chlorobenzene() -> Self {
        Self::custom("chlorobenzene", 5.6968, 1.5241, 33.6, 2.72).with_smd(SmdDescriptors {
            n: 1.5241,
            alpha: 0.0,
            beta: 0.07,
            gamma_cal: 47.48,
            phi: 0.857,
            psi: 0.143,
            is_water: false,
        })
    }

    /// Chloroform (CHCl3).
    pub fn chloroform() -> Self {
        Self::custom("chloroform", 4.7113, 1.4459, 27.2, 2.52).with_smd(SmdDescriptors {
            n: 1.4459,
            alpha: 0.15,
            beta: 0.02,
            gamma_cal: 38.39,
            phi: 0.0,
            psi: 0.75,
            is_water: false,
        })
    }

    /// Toluene (C7H8, aromatic solvent).
    pub fn toluene() -> Self {
        Self::custom("toluene", 2.3741, 1.4961, 28.5, 2.82).with_smd(SmdDescriptors {
            n: 1.4961,
            alpha: 0.0,
            beta: 0.14,
            gamma_cal: 40.20,
            phi: 0.857,
            psi: 0.0,
            is_water: false,
        })
    }

    /// Benzene (C6H6, nonpolar aromatic solvent).
    pub fn benzene() -> Self {
        Self::custom("benzene", 2.2706, 1.5011, 28.9, 2.63).with_smd(SmdDescriptors {
            n: 1.5011,
            alpha: 0.0,
            beta: 0.14,
            gamma_cal: 40.62,
            phi: 1.0,
            psi: 0.0,
            is_water: false,
        })
    }

    /// Hexane (C6H14, aliphatic alkane solvent).
    pub fn hexane() -> Self {
        Self::custom("hexane", 1.8819, 1.3749, 18.4, 3.00).with_smd(SmdDescriptors {
            n: 1.3749,
            alpha: 0.0,
            beta: 0.0,
            gamma_cal: 25.75,
            phi: 0.0,
            psi: 0.0,
            is_water: false,
        })
    }

    /// Cyclohexane (C6H12).
    pub fn cyclohexane() -> Self {
        Self::custom("cyclohexane", 2.0165, 1.4266, 24.9, 2.81).with_smd(SmdDescriptors {
            n: 1.4266,
            alpha: 0.0,
            beta: 0.0,
            gamma_cal: 35.48,
            phi: 0.0,
            psi: 0.0,
            is_water: false,
        })
    }

    /// Dimethyl sulfoxide (DMSO, (CH3)2SO).
    pub fn dmso() -> Self {
        Self::custom("dimethylsulfoxide", 46.826, 1.4783, 42.9, 2.41).with_smd(SmdDescriptors {
            n: 1.4783,
            alpha: 0.0,
            beta: 0.88,
            gamma_cal: 61.78,
            phi: 0.0,
            psi: 0.0,
            is_water: false,
        })
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
            "dmso" | "dimethylsulfoxide" => Some(Self::dmso()),
            "gas" | "vacuum" | "none" => Some(Self::gas()),
            _ => None,
        }
    }

    /// Macroscopic surface tension converted to kcal / (mol * Angstrom^2).
    pub fn surface_tension_kcal_mol_angstrom2(&self) -> f64 {
        self.surface_tension_m_n_per_m * Self::MN_M_TO_KCAL_MOL_ANGSTROM2
    }
}
