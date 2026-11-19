//! SMD (Solvation Model based on Density; Marenich, Cramer, Truhlar,
//! *J. Phys. Chem. B* 2009, 113, 6378-6396) CDS non-electrostatic energy.

use calchem_model::{Molecule, BOHR_TO_ANGSTROM};

use crate::cavity::{CavityLebedevOrder, CavityOptions, CavityRadiiMode, MolecularCavity};
use crate::solvent::Solvent;

/// Hartree to kcal/mol conversion matching Minnesota SMD (`mnsol.F`).
pub const HARTREE_TO_KCAL_SMD: f64 = 627.509451;

/// Maps atomic number `Z` to MNSOL type index `1..=11` (`0` if unparameterized).
#[inline]
fn mnsol_type(z: u8) -> usize {
    match z {
        1 => 1,  // H
        6 => 2,  // C
        7 => 3,  // N
        8 => 4,  // O
        9 => 5,  // F
        16 => 6, // S
        17 => 7, // Cl
        35 => 8, // Br
        15 => 9, // P
        53 => 10, // I
        14 => 11, // Si
        _ => 0,
    }
}

/// Pairwise cutoff distances `RKKVAL` (Angstrom) from `mnsol.F` (`SMXCDS`).
fn rkk_threshold(t1: usize, t2: usize) -> f64 {
    if t1 == 0 || t2 == 0 || t1 > 11 || t2 > 11 {
        return 0.0;
    }
    const TABLE: [[f64; 11]; 11] = [
        // H,   C,    N,    O,    F,    S,    Cl,   Br,   P,    I,    Si
        [0.00, 1.55, 1.55, 1.55, 0.00, 2.14, 0.00, 0.00, 0.00, 0.00, 0.00], // H
        [1.55, 1.84, 1.84, 1.84, 1.84, 2.20, 2.10, 2.30, 2.20, 2.60, 0.00], // C
        [1.55, 1.84, 1.85, 1.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], // N
        [1.55, 1.84, 1.50, 2.75, 0.00, 1.71, 0.00, 0.00, 2.10, 0.00, 2.10], // O
        [0.00, 1.84, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], // F
        [2.14, 2.20, 0.00, 1.71, 0.00, 2.75, 0.00, 0.00, 2.50, 0.00, 0.00], // S
        [0.00, 2.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], // Cl
        [0.00, 2.30, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], // Br
        [0.00, 2.20, 0.00, 2.10, 0.00, 2.50, 0.00, 0.00, 0.00, 0.00, 0.00], // P
        [0.00, 2.60, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], // I
        [0.00, 0.00, 0.00, 2.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], // Si
    ];
    TABLE[t1 - 1][t2 - 1]
}

struct SmdTensionTables {
    /// 1-indexed (len 120): 1..=100 atomic, 101..=118 pair modifications.
    sigma: [f64; 120],
    /// 1-indexed (len 100): H bonded to atom Z.
    hsigma: [f64; 100],
    /// Molecular macroscopic surface tension term `CSSIGM` in `cal / (mol * Angstrom^2)`.
    cssigm: f64,
}

impl SmdTensionTables {
    fn for_solvent(solvent: &Solvent) -> Self {
        let mut sigma = [0.0f64; 120];
        let mut hsigma = [0.0f64; 100];

        if solvent.smd.is_water || solvent.name.eq_ignore_ascii_case("water") {
            sigma[1] = 48.69;
            sigma[6] = 129.74;
            sigma[9] = 38.18;
            sigma[16] = -9.10;
            sigma[17] = 9.82;
            sigma[35] = -8.72;
            sigma[101] = -72.95; // C-C
            sigma[103] = 68.69;  // O-C
            sigma[105] = -48.22; // N-C
            sigma[106] = 121.98; // O-N
            sigma[114] = 68.85;  // O-P
            sigma[116] = 84.10;  // N-C(3)
            hsigma[6] = -60.77;  // H-C
            return Self {
                sigma,
                hsigma,
                cssigm: 0.0,
            };
        }

        let d = &solvent.smd;
        let sn = d.n;
        let sa = d.alpha;
        let sb = d.beta;

        // Zero-order atomic terms (cal mol^-1 A^-2)
        sigma[6] = 58.10 * sn + 48.10 * sa + 32.87 * sb;
        sigma[7] = 32.62 * sn;
        sigma[8] = -17.56 * sn + 193.06 * sa - 43.79 * sb;
        sigma[14] = -18.04 * sn;
        sigma[16] = -33.17 * sn;
        sigma[17] = -24.31 * sn;
        sigma[35] = -35.42 * sn;

        // Pair / environment modifications (101..=116)
        sigma[101] = -62.05 * sn; // C-C
        sigma[103] = -15.70 * sn + 95.99 * sa; // O-C
        sigma[104] = -128.16 * sb; // O-O
        sigma[105] = -41.00 * sa; // N-C
        sigma[106] = 79.13 * sb; // O-N
        sigma[110] = -99.76 * sn + 152.20 * sa; // C-N

        // H-X modifications
        hsigma[6] = -36.37 * sn;
        hsigma[8] = -19.39 * sn;

        let cssigm = 0.35 * d.gamma_cal - 4.19 * d.phi * d.phi - 6.68 * d.psi * d.psi;
        Self {
            sigma,
            hsigma,
            cssigm,
        }
    }
}

#[inline]
fn switching_t(r_ang: f64, r_hold: f64, delta_r: f64) -> f64 {
    if r_hold <= 0.0 {
        return 0.0;
    }
    let cutoff = r_hold + delta_r;
    if r_ang >= cutoff {
        0.0
    } else {
        (delta_r / (r_ang - cutoff)).exp()
    }
}

/// Computes `(g_cds_hartree, sasa_angstrom2)` using Marenich-Cramer-Truhlar 2009 SMD CDS rules.
pub fn compute_smd_cds_energy(mol: &Molecule, solvent: &Solvent) -> (f64, f64) {
    let nat = mol.atoms.len();
    if nat == 0 || (solvent.epsilon - 1.0).abs() < 1e-12 {
        return (0.0, 0.0);
    }

    let sasa_cav = MolecularCavity::build(
        mol,
        &CavityOptions {
            scaling_factor: 1.0,
            lebedev_order: CavityLebedevOrder::N302,
            probe_radius_angstrom: 0.40,
            exclusion_tol_bohr: 1e-5,
            radii_mode: CavityRadiiMode::SmdSasa,
        },
    );
    let areas = &sasa_cav.atomic_areas_angstrom2;
    let total_area_a2 = sasa_cav.total_area_angstrom2;

    let mut r_ang = vec![0.0f64; nat * nat];
    for i in 0..nat {
        for j in (i + 1)..nat {
            let a = mol.atoms[i].position_bohr;
            let b = mol.atoms[j].position_bohr;
            let dx = (a[0] - b[0]) * BOHR_TO_ANGSTROM;
            let dy = (a[1] - b[1]) * BOHR_TO_ANGSTROM;
            let dz = (a[2] - b[2]) * BOHR_TO_ANGSTROM;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            r_ang[i * nat + j] = dist;
            r_ang[j * nat + i] = dist;
        }
    }

    // Switching matrix COT(i, j)
    let mut cot = vec![0.0f64; nat * nat];
    for i in 0..nat {
        let zi = mol.atoms[i].z as u8;
        let ti = mnsol_type(zi);
        for j in 0..nat {
            if i == j {
                continue;
            }
            let zj = mol.atoms[j].z as u8;
            let tj = mnsol_type(zj);
            if ti == 0 || tj == 0 {
                continue;
            }
            let (r_hold, delta_r) = if (zi == 8 && zj == 6) || (zi == 6 && zj == 8) {
                (1.330, 0.10)
            } else if zi == 8 && zj == 8 {
                (1.80, 0.30)
            } else {
                (rkk_threshold(ti, tj), 0.30)
            };
            cot[i * nat + j] = switching_t(r_ang[i * nat + j], r_hold, delta_r);
        }
    }

    let tables = SmdTensionTables::for_solvent(solvent);
    let mut energy_cal = 0.0;

    for i in 0..nat {
        let zi = mol.atoms[i].z as usize;
        let mut sts = if zi < tables.sigma.len() {
            tables.sigma[zi]
        } else {
            0.0
        };

        if zi == 1 {
            for j in 0..nat {
                if i == j {
                    continue;
                }
                let zj = mol.atoms[j].z as usize;
                let cij = cot[i * nat + j];
                if zj < tables.hsigma.len() {
                    sts += cij * tables.hsigma[zj];
                }
                if zj == 7 {
                    for k in 0..nat {
                        if k != j && mol.atoms[k].z == 7 {
                            sts += cij * cot[j * nat + k] * tables.sigma[112];
                        }
                    }
                } else if zj == 8 {
                    for k in 0..nat {
                        if k != i && mol.atoms[k].z == 1 {
                            sts += cij * cot[j * nat + k] * tables.sigma[113];
                        }
                    }
                }
            }
        } else if zi == 8 {
            for j in 0..nat {
                if i == j {
                    continue;
                }
                let zj = mol.atoms[j].z;
                let cij = cot[i * nat + j];
                match zj {
                    6 => sts += cij * tables.sigma[103],
                    7 => sts += cij * tables.sigma[106],
                    16 => sts += cij * tables.sigma[118],
                    14 => sts += cij * tables.sigma[117],
                    15 => sts += cij * tables.sigma[114],
                    8 => sts += cij * tables.sigma[104],
                    _ => {}
                }
            }
        } else if zi == 7 {
            let mut rtkks = 0.0;
            let mut rtkks2 = 0.0;
            for j in 0..nat {
                if i == j || mol.atoms[j].z != 6 {
                    continue;
                }
                let mut rtkk3 = 0.0;
                let mut rtkk5 = 0.0;
                let tj = mnsol_type(6);
                for k in 0..nat {
                    if k == i || k == j {
                        continue;
                    }
                    let zk = mol.atoms[k].z as u8;
                    let tk = mnsol_type(zk);
                    if tk == 0 {
                        continue;
                    }
                    let r_hold2 = rkk_threshold(tk, tj);
                    let t_jk = switching_t(r_ang[j * nat + k], r_hold2, 0.30);
                    if zk == 8 {
                        rtkk5 += t_jk;
                    }
                    rtkk3 += t_jk;
                }
                let cij = cot[i * nat + j];
                rtkks += cij * rtkk3 * rtkk3;
                rtkks2 += cij * rtkk5;
            }
            if rtkks > 0.0 {
                sts += rtkks.powf(1.3) * tables.sigma[105];
            }
            sts += rtkks2 * tables.sigma[111];

            // N-C triple bond term
            let mut rtkk_nc3 = 0.0;
            for j in 0..nat {
                if j != i && mol.atoms[j].z == 6 {
                    rtkk_nc3 += switching_t(r_ang[i * nat + j], 1.225, 0.065);
                }
            }
            sts += rtkk_nc3 * tables.sigma[116];
        } else if zi == 6 {
            let tc = mnsol_type(6);
            let mut rtkk1 = 0.0;
            let mut rtkk2 = 0.0;
            for j in 0..nat {
                if j != i && mol.atoms[j].z == 6 {
                    rtkk1 += switching_t(r_ang[i * nat + j], rkk_threshold(tc, tc), 0.30);
                    rtkk2 += switching_t(r_ang[i * nat + j], 1.27, 0.07);
                }
            }
            sts += rtkk1 * tables.sigma[101] + rtkk2 * tables.sigma[102];

            let mut rtkk_cn = 0.0;
            for j in 0..nat {
                if j != i && mol.atoms[j].z == 7 {
                    rtkk_cn += cot[i * nat + j];
                }
            }
            sts += rtkk_cn * rtkk_cn * tables.sigma[110];
        } else if zi == 16 {
            for j in 0..nat {
                if j != i && mol.atoms[j].z == 16 {
                    sts += cot[i * nat + j] * tables.sigma[107];
                } else if j != i && mol.atoms[j].z == 15 {
                    sts += cot[i * nat + j] * tables.sigma[115];
                }
            }
        }

        energy_cal += (sts + tables.cssigm) * areas[i];
    }

    let g_cds_eh = (energy_cal / 1000.0) / HARTREE_TO_KCAL_SMD;
    (g_cds_eh, total_area_a2)
}
