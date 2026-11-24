//! Standard two-body D3(BJ)/D3(0), in atomic units; ATM is not included.
//! CN interpolation uses the original pair reference tables, not atomic mixing rules.
//! Data attribution and corresponding source: ../data/d3/NOTICE.md.
use calchem_model::{Atom, CalcError};
use serde::{Deserialize, Serialize};
#[path = "d3_data.rs"]
mod data;
const A0_ANGSTROM: f64 = 0.529177210903;
const C6_BYTES: &[u8] = include_bytes!("../data/d3/c6.bin");

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DampingScheme {
    BeckeJohnson {
        s6: f64,
        s8: f64,
        a1: f64,
        a2: f64,
    },
    Zero {
        s6: f64,
        s8: f64,
        sr6: f64,
        sr8: f64,
        alpha6: f64,
        alpha8: f64,
    },
}
impl Default for DampingScheme {
    fn default() -> Self {
        Self::BeckeJohnson {
            s6: 1.0,
            s8: 1.9889,
            a1: 0.3981,
            a2: 4.4211,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct D3Model {
    pub damping: DampingScheme,
    pub functional_name: String,
}

fn pair_index(a: usize, b: usize) -> usize {
    let hi = a.max(b);
    let lo = a.min(b);
    hi * (hi - 1) / 2 + lo - 1
}
pub fn covalent_radius_bohr(z: u8) -> f64 {
    data::COVALENT_ANGSTROM[z as usize - 1] * 4.0 / (3.0 * A0_ANGSTROM)
}
pub fn vdw_radius_bohr(z: u8) -> f64 {
    data::VDW_PAIR_ANGSTROM[pair_index(z as usize, z as usize)] / (2.0 * A0_ANGSTROM)
}
fn r4r2(z: u8) -> f64 {
    (0.5 * (z as f64).sqrt() * data::R4_OVER_R2[z as usize - 1]).sqrt()
}
fn reference_c6(a: u8, b: u8, i: usize, j: usize) -> f64 {
    let (r, c) = if a > b { (i, j) } else { (j, i) };
    let off = (pair_index(a as usize, b as usize) * 49 + c * 7 + r) * 8;
    f64::from_le_bytes(
        C6_BYTES[off..off + 8]
            .try_into()
            .expect("fixed D3 table layout"),
    )
}
fn weights(z: u8, cn: f64) -> ([f64; 7], [f64; 7]) {
    let refs = &data::REFERENCE_CN[(z as usize - 1) * 7..z as usize * 7];
    let mut w = [0.0; 7];
    let mut dw = [0.0; 7];
    let mut nearest = f64::INFINITY;
    for &r in refs {
        if r >= 0.0 {
            nearest = nearest.min((cn - r).powi(2));
        }
    }
    let mut sum = 0.0;
    for i in 0..7 {
        if refs[i] >= 0.0 {
            w[i] = (-4.0 * ((cn - refs[i]).powi(2) - nearest)).exp();
            sum += w[i];
        }
    }
    let mut mean = 0.0;
    for i in 0..7 {
        w[i] /= sum;
        dw[i] = -8.0 * (cn - refs[i]);
        mean += w[i] * dw[i];
    }
    for i in 0..7 {
        dw[i] = w[i] * (dw[i] - mean);
    }
    (w, dw)
}
fn pair_c6(a: u8, b: u8, wa: &([f64; 7], [f64; 7]), wb: &([f64; 7], [f64; 7])) -> (f64, f64, f64) {
    let (mut v, mut da, mut db) = (0.0, 0.0, 0.0);
    for i in 0..7 {
        for j in 0..7 {
            let c = reference_c6(a, b, i, j);
            v += wa.0[i] * wb.0[j] * c;
            da += wa.1[i] * wb.0[j] * c;
            db += wa.0[i] * wb.1[j] * c;
        }
    }
    (v, da, db)
}
impl D3Model {
    pub fn from_scheme(scheme: &str, functional: &str) -> Result<Self, CalcError> {
        match scheme.trim().to_ascii_lowercase().as_str() {
            "bj" | "d3bj" | "d3(bj)" => Self::d3bj(functional),
            "zero" | "d3zero" | "d3(0)" => Self::d3zero(functional),
            other => Err(CalcError::UnsupportedMethod(format!("Unknown D3 damping scheme '{other}'"))),
        }
    }

    pub fn add_gradient(&self, atoms: &[Atom], gradient: &mut [[f64; 3]]) -> Result<f64, CalcError> {
        if atoms.len() != gradient.len() {
            return Err(CalcError::Internal("D3 gradient dimension mismatch".into()));
        }
        let (energy, correction) = self.energy_and_gradient(atoms, true)?;
        for (g, d) in gradient.iter_mut().zip(correction) {
            for axis in 0..3 { g[axis] += d[axis]; }
        }
        Ok(energy)
    }

    pub fn d3bj(functional: &str) -> Result<Self, CalcError> {
        let f = functional.trim().to_ascii_lowercase();
        let damping = match f.as_str() {
            "b3lyp" | "b3lyp5" => DampingScheme::BeckeJohnson {
                s6: 1.0,
                s8: 1.9889,
                a1: 0.3981,
                a2: 4.4211,
            },
            "pbe" => DampingScheme::BeckeJohnson {
                s6: 1.0,
                s8: 0.7875,
                a1: 0.4289,
                a2: 4.4407,
            },
            "pbe0" => DampingScheme::BeckeJohnson {
                s6: 1.0,
                s8: 1.2177,
                a1: 0.4145,
                a2: 4.8593,
            },
            "blyp" => DampingScheme::BeckeJohnson {
                s6: 1.0,
                s8: 2.6996,
                a1: 0.4298,
                a2: 4.2359,
            },
            "bp86" => DampingScheme::BeckeJohnson {
                s6: 1.0,
                s8: 3.2822,
                a1: 0.3946,
                a2: 4.8516,
            },
            "tpss" => DampingScheme::BeckeJohnson {
                s6: 1.0,
                s8: 1.9435,
                a1: 0.4535,
                a2: 4.4752,
            },
            "b97" | "b97-d" => DampingScheme::BeckeJohnson {
                s6: 1.0,
                s8: 2.2609,
                a1: 0.5545,
                a2: 3.2297,
            },
            "cam-b3lyp" | "camb3lyp" => DampingScheme::BeckeJohnson {
                s6: 1.0,
                s8: 2.0674,
                a1: 0.3708,
                a2: 5.4743,
            },
            "hf" => DampingScheme::BeckeJohnson {
                s6: 1.0,
                s8: 0.9171,
                a1: 0.3385,
                a2: 2.883,
            },
            _ => {
                return Err(CalcError::UnsupportedMethod(format!(
                    "Unsupported D3 functional: {functional}"
                )))
            }
        };
        Ok(Self {
            damping,
            functional_name: f,
        })
    }
    pub fn d3zero(functional: &str) -> Result<Self, CalcError> {
        let f = functional.trim().to_ascii_lowercase();
        let damping = match f.as_str() {
            "b3lyp" | "b3lyp5" => DampingScheme::Zero {
                s6: 1.0,
                s8: 1.703,
                sr6: 1.261,
                sr8: 1.0,
                alpha6: 14.0,
                alpha8: 16.0,
            },
            "pbe" => DampingScheme::Zero {
                s6: 1.0,
                s8: 0.722,
                sr6: 1.217,
                sr8: 1.0,
                alpha6: 14.0,
                alpha8: 16.0,
            },
            "pbe0" => DampingScheme::Zero {
                s6: 1.0,
                s8: 0.928,
                sr6: 1.287,
                sr8: 1.0,
                alpha6: 14.0,
                alpha8: 16.0,
            },
            "blyp" => DampingScheme::Zero {
                s6: 1.0,
                s8: 1.682,
                sr6: 1.094,
                sr8: 1.0,
                alpha6: 14.0,
                alpha8: 16.0,
            },
            "bp86" => DampingScheme::Zero {
                s6: 1.0,
                s8: 1.683,
                sr6: 1.139,
                sr8: 1.0,
                alpha6: 14.0,
                alpha8: 16.0,
            },
            "tpss" => DampingScheme::Zero {
                s6: 1.0,
                s8: 1.105,
                sr6: 1.166,
                sr8: 1.0,
                alpha6: 14.0,
                alpha8: 16.0,
            },
            "b97" | "b97-d" => DampingScheme::Zero {
                s6: 1.0,
                s8: 0.909,
                sr6: 0.892,
                sr8: 1.0,
                alpha6: 14.0,
                alpha8: 16.0,
            },
            "cam-b3lyp" | "camb3lyp" => DampingScheme::Zero {
                s6: 1.0,
                s8: 1.217,
                sr6: 1.378,
                sr8: 1.0,
                alpha6: 14.0,
                alpha8: 16.0,
            },
            "hf" => DampingScheme::Zero {
                s6: 1.0,
                s8: 1.746,
                sr6: 1.158,
                sr8: 1.0,
                alpha6: 14.0,
                alpha8: 16.0,
            },
            _ => {
                return Err(CalcError::UnsupportedMethod(format!(
                    "Unsupported D3 functional: {functional}"
                )))
            }
        };
        Ok(Self {
            damping,
            functional_name: f,
        })
    }

    pub fn energy(&self, atoms: &[Atom]) -> Result<f64, CalcError> {
        Ok(self.energy_and_gradient(atoms, false)?.0)
    }
    pub fn gradient(&self, atoms: &[Atom]) -> Result<Vec<[f64; 3]>, CalcError> {
        Ok(self.energy_and_gradient(atoms, true)?.1)
    }
    pub fn energy_and_gradient(
        &self,
        atoms: &[Atom],
        compute_gradient: bool,
    ) -> Result<(f64, Vec<[f64; 3]>), CalcError> {
        for a in atoms {
            if !(1..=103).contains(&a.z) || !a.position_bohr.iter().all(|x| x.is_finite()) {
                return Err(CalcError::InvalidMolecule(
                    "D3 requires finite coordinates and elements Z=1..103".into(),
                ));
            }
        }
        let valid = match self.damping {
            DampingScheme::BeckeJohnson { s6, s8, a1, a2 } => {
                [s6, s8, a1, a2].iter().all(|v| v.is_finite() && *v >= 0.0) && a2 > 0.0
            }
            DampingScheme::Zero {
                s6,
                s8,
                sr6,
                sr8,
                alpha6,
                alpha8,
            } => {
                [s6, s8, sr6, sr8, alpha6, alpha8]
                    .iter()
                    .all(|v| v.is_finite() && *v >= 0.0)
                    && sr6 > 0.0
                    && sr8 > 0.0
                    && alpha6 > 6.0
                    && alpha8 > 8.0
            }
        };
        if !valid {
            return Err(CalcError::InvalidMolecule(
                "Invalid D3 damping parameters".into(),
            ));
        }
        let n = atoms.len();
        let mut cn = vec![0.0; n];
        // O(N^2) geometry only; no AO tensor. 40 Bohr is the standard CN cutoff.
        for i in 0..n {
            for j in 0..i {
                let r = distance(atoms, i, j).1;
                if !r.is_finite() || r < 1e-8 {
                    return Err(CalcError::InvalidMolecule(
                        "D3 coincident atoms or non-finite distance".into(),
                    ));
                }
                if r <= 40.0 {
                    let rc = covalent_radius_bohr(atoms[i].z) + covalent_radius_bohr(atoms[j].z);
                    let f = 1.0 / (1.0 + (-16.0 * (rc / r - 1.0)).exp());
                    cn[i] += f;
                    cn[j] += f;
                }
            }
        }
        let ws: Vec<_> = atoms
            .iter()
            .zip(&cn)
            .map(|(a, &c)| weights(a.z, c))
            .collect();
        let mut grad = vec![[0.0; 3]; n];
        let mut dedcn = vec![0.0; n];
        let mut energy = 0.0;
        for i in 0..n {
            for j in 0..i {
                let (d, r) = distance(atoms, i, j);
                let (c6, dc_i, dc_j) = pair_c6(atoms[i].z, atoms[j].z, &ws[i], &ws[j]);
                let c8_ratio = 3.0 * r4r2(atoms[i].z) * r4r2(atoms[j].z);
                let mut kernel = 0.0;
                let mut dkernel = 0.0;
                match self.damping {
                    DampingScheme::BeckeJohnson { s6, s8, a1, a2 } => {
                        let cutoff = a1 * c8_ratio.sqrt() + a2;
                        for (power, scale) in [(6, s6), (8, s8 * c8_ratio)] {
                            let denom = r.powi(power) + cutoff.powi(power);
                            kernel -= scale / denom;
                            dkernel += scale * power as f64 * r.powi(power - 1) / (denom * denom);
                        }
                    }
                    DampingScheme::Zero {
                        s6,
                        s8,
                        sr6,
                        sr8,
                        alpha6,
                        alpha8,
                    } => {
                        let r0 = data::VDW_PAIR_ANGSTROM
                            [pair_index(atoms[i].z as usize, atoms[j].z as usize)]
                            / A0_ANGSTROM;
                        for (power, scale, sr, alpha) in
                            [(6, s6, sr6, alpha6), (8, s8 * c8_ratio, sr8, alpha8)]
                        {
                            let damp = 1.0 / (1.0 + 6.0 * (sr * r0 / r).powf(alpha));
                            let factor = scale * damp / r.powi(power);
                            kernel -= factor;
                            dkernel += factor / r * (power as f64 - alpha * (1.0 - damp));
                        }
                    }
                }
                energy += c6 * kernel;
                if compute_gradient {
                    dedcn[i] += dc_i * kernel;
                    dedcn[j] += dc_j * kernel;
                    for q in 0..3 {
                        let f = c6 * dkernel * d[q] / r;
                        grad[i][q] += f;
                        grad[j][q] -= f;
                    }
                }
            }
        }
        if compute_gradient {
            for i in 0..n {
                for j in 0..i {
                    let (d, r) = distance(atoms, i, j);
                    if r > 40.0 {
                        continue;
                    }
                    let rc = covalent_radius_bohr(atoms[i].z) + covalent_radius_bohr(atoms[j].z);
                    let f = 1.0 / (1.0 + (-16.0 * (rc / r - 1.0)).exp());
                    let factor = -(dedcn[i] + dedcn[j]) * 16.0 * rc / (r * r) * f * (1.0 - f);
                    for q in 0..3 {
                        let v = factor * d[q] / r;
                        grad[i][q] += v;
                        grad[j][q] -= v;
                    }
                }
            }
        }
        if !energy.is_finite() || grad.iter().flatten().any(|x| !x.is_finite()) {
            return Err(CalcError::InvalidMolecule(
                "D3 result exceeds floating-point range".into(),
            ));
        }
        Ok((energy, grad))
    }
}
fn distance(atoms: &[Atom], i: usize, j: usize) -> ([f64; 3], f64) {
    let d = std::array::from_fn(|q| atoms[i].position_bohr[q] - atoms[j].position_bohr[q]);
    let r = d[0].hypot(d[1]).hypot(d[2]);
    (d, r)
}

/// Grimme (2006) D2 C6 parameters (J nm^6 / mol) for Z = 1..=54.
const D2_C6_J_NM6_MOL: [f64; 54] = [
    0.14, 0.08, 1.61, 1.61, 3.13, 1.75, 1.23, 0.70, 0.75, 0.63,
    5.71, 5.71, 10.79, 9.23, 7.84, 5.57, 5.07, 4.61,
    10.80, 10.80, 10.80, 10.80, 10.80, 10.80, 10.80, 10.80, 10.80, 10.80, 10.80, 10.80,
    16.99, 17.10, 16.37, 12.64, 12.47, 12.01,
    24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67,
    37.32, 38.71, 38.44, 31.74, 31.50, 29.99,
];

/// Grimme (2006) D2 atomic vdW radii R0 (Angstrom) for Z = 1..=54.
const D2_R0_ANGSTROM: [f64; 54] = [
    1.001, 1.012, 0.825, 1.408, 1.485, 1.452, 1.397, 1.342, 1.287, 1.243,
    1.144, 1.364, 1.639, 1.716, 1.705, 1.683, 1.639, 1.595,
    1.485, 1.474, 1.562, 1.562, 1.562, 1.562, 1.562, 1.562, 1.562, 1.562, 1.562, 1.562,
    1.650, 1.727, 1.760, 1.771, 1.749, 1.727,
    1.628, 1.606, 1.639, 1.639, 1.639, 1.639, 1.639, 1.639, 1.639, 1.639, 1.639, 1.639,
    1.672, 1.804, 1.881, 1.892, 1.892, 1.881,
];

/// Chai & Head-Gordon (2008) wB97X-D empirical D2 dispersion energy (Eh) and nuclear gradient (Eh/Bohr).
pub fn wb97xd_chg_d2_energy_and_gradient(atoms: &[Atom]) -> Result<(f64, Vec<[f64; 3]>), CalcError> {
    const J_NM6_MOL_TO_AU: f64 = 17.309582628876695;
    let n = atoms.len();
    let mut grad = vec![[0.0; 3]; n];
    let mut energy = 0.0;
    for i in 0..n {
        let zi = atoms[i].z as usize;
        if zi == 0 || zi > 54 {
            return Err(CalcError::UnsupportedMethod(format!(
                "wB97X-D D2 dispersion only supports Z = 1..=54, got Z={zi}"
            )));
        }
        for j in 0..i {
            let zj = atoms[j].z as usize;
            let (d, r) = distance(atoms, i, j);
            if !r.is_finite() || r < 1e-8 {
                return Err(CalcError::InvalidMolecule(
                    "wB97X-D D2 coincident atoms or non-finite distance".into(),
                ));
            }
            let c6_ij = (D2_C6_J_NM6_MOL[zi - 1] * D2_C6_J_NM6_MOL[zj - 1]).sqrt() * J_NM6_MOL_TO_AU;
            let r0_ij = (D2_R0_ANGSTROM[zi - 1] + D2_R0_ANGSTROM[zj - 1]) / A0_ANGSTROM;
            let ratio = r0_ij / r;
            let damp = 1.0 / (1.0 + 6.0 * ratio.powi(12));
            let r6 = r.powi(6);
            let e_ij = -c6_ij * damp / r6;
            energy += e_ij;

            // d(e_ij)/dR = c6_ij * damp / R^7 * (6 - 12 * (1 - damp))
            let de_dr = c6_ij * damp / (r6 * r) * (6.0 - 12.0 * (1.0 - damp));
            for q in 0..3 {
                let f = de_dr * d[q] / r;
                grad[i][q] += f;
                grad[j][q] -= f;
            }
        }
    }
    Ok((energy, grad))
}
