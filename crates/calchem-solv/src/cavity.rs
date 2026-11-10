//! Molecular cavity construction using interlocking Bondi van der Waals spheres and
//! Lebedev-Laikov spherical tessellation.

use calchem_dft::lebedev::LebedevDegree;
use calchem_model::{Molecule, BOHR_TO_ANGSTROM};
use serde::{Deserialize, Serialize};

/// Supported Lebedev angular grid orders for cavity boundary discretization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CavityLebedevOrder {
    /// 26 points per atomic sphere.
    N26,
    /// 50 points per atomic sphere (standard default).
    N50,
    /// 110 points per atomic sphere (high angular precision).
    N110,
    /// 194 points per atomic sphere.
    N194,
    /// 302 points per atomic sphere.
    N302,
}

impl Default for CavityLebedevOrder {
    fn default() -> Self {
        Self::N50
    }
}

impl From<CavityLebedevOrder> for LebedevDegree {
    fn from(order: CavityLebedevOrder) -> Self {
        match order {
            CavityLebedevOrder::N26 => LebedevDegree::N26,
            CavityLebedevOrder::N50 => LebedevDegree::N50,
            CavityLebedevOrder::N110 => LebedevDegree::N110,
            CavityLebedevOrder::N194 => LebedevDegree::N194,
            CavityLebedevOrder::N302 => LebedevDegree::N302,
        }
    }
}

/// Bondi (1964) van der Waals radii in Angstroms.
pub fn bondi_vdw_radius(atomic_number: u8) -> f64 {
    match atomic_number {
        1 => 1.20,  // H
        2 => 1.40,  // He
        3 => 1.82,  // Li
        4 => 1.53,  // Be
        5 => 1.92,  // B
        6 => 1.70,  // C
        7 => 1.55,  // N
        8 => 1.52,  // O
        9 => 1.47,  // F
        10 => 1.54, // Ne
        11 => 2.27, // Na
        12 => 1.73, // Mg
        13 => 1.84, // Al
        14 => 2.10, // Si
        15 => 1.80, // P
        16 => 1.80, // S
        17 => 1.75, // Cl
        18 => 1.88, // Ar
        19 => 2.75, // K
        20 => 2.31, // Ca
        35 => 1.85, // Br
        53 => 1.98, // I
        _ => 2.00,  // General heavy default
    }
}

/// Options controlling cavity geometry and tessellation density.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CavityOptions {
    /// Scaling multiplier applied to atomic vdW radii (standard COSMO/C-PCM default = 1.20).
    pub scaling_factor: f64,
    /// Lebedev quadrature rule applied on each atomic sphere.
    pub lebedev_order: CavityLebedevOrder,
    /// Solvent probe radius (Angstrom). Default 0.0 for vdW/solvent cavity; >0 for SAS.
    pub probe_radius_angstrom: f64,
    /// Overlap exclusion tolerance in Bohr (default 1e-5).
    pub exclusion_tol_bohr: f64,
}

impl Default for CavityOptions {
    fn default() -> Self {
        Self {
            scaling_factor: 1.20,
            lebedev_order: CavityLebedevOrder::N50,
            probe_radius_angstrom: 0.0,
            exclusion_tol_bohr: 1e-5,
        }
    }
}

/// Single surface element (tessera) on the molecular cavity boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tessera {
    /// Cartesian position of the tessera center (Bohr).
    pub position_bohr: [f64; 3],
    /// Outward unit normal vector.
    pub normal: [f64; 3],
    /// Surface area of this tessera (Bohr^2).
    pub area_bohr2: f64,
    /// Index of the parent atom whose sphere generated this tessera.
    pub atom_index: usize,
}

/// Discretized molecular boundary cavity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MolecularCavity {
    /// Collection of exposed boundary tesserae.
    pub tesserae: Vec<Tessera>,
    /// Total cavity surface area in atomic units (Bohr^2).
    pub total_area_bohr2: f64,
    /// Total cavity surface area in Angstrom^2.
    pub total_area_angstrom2: f64,
    /// Solvent-accessible surface area per atom (Angstrom^2).
    pub atomic_areas_angstrom2: Vec<f64>,
}

impl MolecularCavity {
    /// Builds the molecular cavity by placing Lebedev-tessellated spheres on each atom
    /// and filtering out points buried inside intersecting spheres of neighboring atoms.
    pub fn build(mol: &Molecule, opts: &CavityOptions) -> Self {
        let n_atoms = mol.atoms.len();
        let angstrom_to_bohr = 1.0 / BOHR_TO_ANGSTROM;

        // Compute effective sphere radii in Bohr for all atoms
        let radii_bohr: Vec<f64> = mol
            .atoms
            .iter()
            .map(|a| {
                let r_vdw = bondi_vdw_radius(a.z);
                let r_ang = opts.scaling_factor * r_vdw + opts.probe_radius_angstrom;
                r_ang * angstrom_to_bohr
            })
            .collect();

        let deg: LebedevDegree = opts.lebedev_order.into();
        let lebedev_points = deg.points();
        let mut tesserae = Vec::new();
        let mut atomic_areas_bohr2 = vec![0.0f64; n_atoms];

        for (i, atom) in mol.atoms.iter().enumerate() {
            let r_i = radii_bohr[i];
            let c_i = atom.position_bohr;
            let r_i2 = r_i * r_i;

            for &(u, w) in lebedev_points {
                // Point on sphere surface
                let s = [c_i[0] + r_i * u[0], c_i[1] + r_i * u[1], c_i[2] + r_i * u[2]];
                let area = w * r_i2;

                // Check whether this point lies inside any other atomic sphere
                let mut buried = false;
                for (j, other_atom) in mol.atoms.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    let c_j = other_atom.position_bohr;
                    let r_j = radii_bohr[j];
                    let dist2 = (s[0] - c_j[0]).powi(2)
                        + (s[1] - c_j[1]).powi(2)
                        + (s[2] - c_j[2]).powi(2);

                    if dist2 < (r_j - opts.exclusion_tol_bohr).powi(2) {
                        buried = true;
                        break;
                    }
                }

                if !buried {
                    tesserae.push(Tessera {
                        position_bohr: s,
                        normal: u,
                        area_bohr2: area,
                        atom_index: i,
                    });
                    atomic_areas_bohr2[i] += area;
                }
            }
        }

        let total_area_bohr2: f64 = tesserae.iter().map(|t| t.area_bohr2).sum();
        let bohr2_to_angstrom2 = BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM;
        let total_area_angstrom2 = total_area_bohr2 * bohr2_to_angstrom2;
        let atomic_areas_angstrom2: Vec<f64> = atomic_areas_bohr2
            .iter()
            .map(|&a| a * bohr2_to_angstrom2)
            .collect();

        Self {
            tesserae,
            total_area_bohr2,
            total_area_angstrom2,
            atomic_areas_angstrom2,
        }
    }

    /// Number of exposed surface tesserae.
    pub fn n_tesserae(&self) -> usize {
        self.tesserae.len()
    }
}
