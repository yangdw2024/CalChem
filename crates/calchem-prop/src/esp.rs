//! Exact quantum-mechanical molecular electrostatic potential (ESP), Mulliken / Löwdin
//! population analysis, and ESP-mapped isodensity surfaces with the `CCESP\x01` wire format.
//!
//! ESP at a point `r` (atomic units, Hartree / e):
//! `V(r) = sum_A Z_eff(A) / |r - R_A| - sum_{mu,nu} P_{mu nu} <mu | 1/|r' - r| | nu>`
//! where `P` is the spin-summed (total) AO density matrix and `Z_eff` already removes ECP
//! core electrons, so valence density and screened nuclei stay consistent.

use calchem_basis::BasisSet;
use calchem_int::boys::boys_batch;
use calchem_int::one_electron::compute_overlap;
use calchem_int::primitive::{nuclear_vrr, GaussianPair};
use calchem_model::{CalcError, Molecule, BOHR_TO_ANGSTROM};
use calchem_tensor::{eigh_sym, Matrix2D};
use rayon::prelude::*;

use crate::marching_cubes::{extract_isosurface, IsosurfaceMesh};
use crate::{auto_grid_for_molecule, compute_density_grid, GridSpec};

/// Wire magic for ESP-mapped single-surface packets (`"CCESP"` + v1 + 2 pad bytes).
pub const CCESP_MAGIC: [u8; 8] = *b"CCESP\x01\x00\x00";
/// Fixed 64-byte cache-line-aligned header size for `CCESP\x01` packets.
pub const CCESP_HEADER_BYTES: usize = 64;

/// Minimum allowed distance (Bohr) between an ESP evaluation point and any nucleus.
const NUCLEUS_EXCLUSION_BOHR: f64 = 1e-8;
/// Primitive-pair prescreening threshold on `|c| * K_AB * 2 pi / gamma`.
const PRIM_PAIR_SCREEN: f64 = 1e-15;

/// One contracted primitive pair with density-weighted component coefficients.
struct EspPrimPair {
    pair: GaussianPair,
    /// `(ia, ib, weight)` with `weight = c_a c_b N_a N_b * P_sym(ia, ib)` (zero entries removed).
    terms: Vec<(usize, usize, f64)>,
}

struct EspShellPair {
    comps_a: &'static [[u8; 3]],
    comps_b: &'static [[u8; 3]],
    max_m: usize,
    prims: Vec<EspPrimPair>,
}

/// Precomputed, density-contracted electronic ESP operator (reusable across many points).
pub struct EspEvaluator {
    nuclei: Vec<([f64; 3], f64)>,
    shell_pairs: Vec<EspShellPair>,
    max_m: usize,
}

impl EspEvaluator {
    /// Builds the evaluator from a spin-summed AO density matrix (row-major, `n_ao * n_ao`).
    pub fn new(mol: &Molecule, basis: &BasisSet, density_total: &[f64]) -> Result<Self, CalcError> {
        let n = basis.n_ao;
        if density_total.len() != n * n {
            return Err(CalcError::InvalidMolecule(format!(
                "Density matrix length {} does not match n_ao^2 ({n}^2)",
                density_total.len()
            )));
        }
        let nuclei = mol
            .atoms
            .iter()
            .enumerate()
            .map(|(i, a)| (a.position_bohr, basis.effective_nuclear_charge(mol, i)))
            .collect();

        let mut shell_pairs = Vec::new();
        let mut global_max_m = 0usize;
        for (ia_sh, sh_a) in basis.shells.iter().enumerate() {
            for sh_b in basis.shells.iter().skip(ia_sh) {
                let same = std::ptr::eq(sh_a, sh_b);
                let comps_a = sh_a.components();
                let comps_b = sh_b.components();

                // Symmetrized density block: off-diagonal shell pairs count both (mu,nu) and (nu,mu).
                let mut p_blk = vec![0.0f64; comps_a.len() * comps_b.len()];
                let mut p_max = 0.0f64;
                for ia in 0..comps_a.len() {
                    let mu = sh_a.ao_offset + ia;
                    for ib in 0..comps_b.len() {
                        let nu = sh_b.ao_offset + ib;
                        let v = if same {
                            density_total[mu * n + nu]
                        } else {
                            density_total[mu * n + nu] + density_total[nu * n + mu]
                        };
                        p_blk[ia * comps_b.len() + ib] = v;
                        p_max = p_max.max(v.abs());
                    }
                }
                if p_max < 1e-14 {
                    continue;
                }

                let max_m = (sh_a.shell_type.angular_momentum()
                    + sh_b.shell_type.angular_momentum()) as usize;
                let mut prims = Vec::new();
                for (pa_idx, pa) in sh_a.primitives.iter().enumerate() {
                    for (pb_idx, pb) in sh_b.primitives.iter().enumerate() {
                        let pair =
                            GaussianPair::new(pa.exponent, sh_a.center, pb.exponent, sh_b.center);
                        let mut terms = Vec::new();
                        let mut w_max = 0.0f64;
                        for (ia, &ca) in comps_a.iter().enumerate() {
                            let na = sh_a.primitive_norm_for(pa_idx, ca);
                            for (ib, &cb) in comps_b.iter().enumerate() {
                                let p = p_blk[ia * comps_b.len() + ib];
                                if p.abs() < 1e-14 {
                                    continue;
                                }
                                let nb = sh_b.primitive_norm_for(pb_idx, cb);
                                let w = pa.coeff * pb.coeff * na * nb * p;
                                w_max = w_max.max(w.abs());
                                terms.push((ia, ib, w));
                            }
                        }
                        let bound = w_max * pair.kab * 2.0 * std::f64::consts::PI / pair.gamma;
                        if terms.is_empty() || bound < PRIM_PAIR_SCREEN {
                            continue;
                        }
                        prims.push(EspPrimPair { pair, terms });
                    }
                }
                if !prims.is_empty() {
                    global_max_m = global_max_m.max(max_m);
                    shell_pairs.push(EspShellPair {
                        comps_a,
                        comps_b,
                        max_m,
                        prims,
                    });
                }
            }
        }

        Ok(Self {
            nuclei,
            shell_pairs,
            max_m: global_max_m,
        })
    }

    /// Nuclear (screened by ECP) contribution `sum_A Z_eff / |r - R_A|`.
    pub fn nuclear_potential(&self, r: [f64; 3]) -> Result<f64, CalcError> {
        let mut v = 0.0;
        for &(c, z) in &self.nuclei {
            let d = ((r[0] - c[0]).powi(2) + (r[1] - c[1]).powi(2) + (r[2] - c[2]).powi(2)).sqrt();
            if d < NUCLEUS_EXCLUSION_BOHR {
                return Err(CalcError::InvalidMolecule(format!(
                    "ESP evaluation point {r:?} coincides with a nucleus (singular potential)"
                )));
            }
            v += z / d;
        }
        Ok(v)
    }

    /// Electronic contribution `sum P_{mu nu} <mu | 1/|r' - r| | nu>` (positive number).
    pub fn electronic_potential(&self, r: [f64; 3], boys_buf: &mut [f64]) -> f64 {
        let mut v = 0.0;
        for sp in &self.shell_pairs {
            for pp in &sp.prims {
                let pair = &pp.pair;
                let pc2 = (pair.p[0] - r[0]).powi(2)
                    + (pair.p[1] - r[1]).powi(2)
                    + (pair.p[2] - r[2]).powi(2);
                boys_batch(sp.max_m, pair.gamma * pc2, boys_buf);
                for &(ia, ib, w) in &pp.terms {
                    v += w * nuclear_vrr(sp.comps_a[ia], sp.comps_b[ib], 0, pair, r, boys_buf);
                }
            }
        }
        v
    }

    /// Total ESP `V_nuc(r) - V_el(r)` (Hartree / e).
    pub fn potential(&self, r: [f64; 3], boys_buf: &mut [f64]) -> Result<f64, CalcError> {
        Ok(self.nuclear_potential(r)? - self.electronic_potential(r, boys_buf))
    }

    /// Core-softened ESP for regular 3D voxel grids (`cubegen` convention: clamps nuclear distance
    /// at `1e-6` Bohr so symmetric grids passing through an atomic center remain finite).
    #[inline]
    pub fn potential_grid_clamped(&self, r: [f64; 3], boys_buf: &mut [f64]) -> f64 {
        let mut v_nuc = 0.0;
        for &(c, z) in &self.nuclei {
            let d = ((r[0] - c[0]).powi(2) + (r[1] - c[1]).powi(2) + (r[2] - c[2]).powi(2))
                .sqrt()
                .max(1e-6);
            v_nuc += z / d;
        }
        v_nuc - self.electronic_potential(r, boys_buf)
    }

    /// Rayon-parallel ESP at an arbitrary list of Bohr points.
    pub fn potential_at_points(&self, points_bohr: &[[f64; 3]]) -> Result<Vec<f64>, CalcError> {
        let m = self.max_m + 1;
        points_bohr
            .par_iter()
            .map_init(|| vec![0.0f64; m + 1], |buf, &r| self.potential(r, buf))
            .collect()
    }
}

/// Convenience wrapper: ESP at points (Bohr) from a total density matrix.
pub fn compute_esp_at_points(
    mol: &Molecule,
    basis: &BasisSet,
    density_total: &[f64],
    points_bohr: &[[f64; 3]],
) -> Result<Vec<f64>, CalcError> {
    EspEvaluator::new(mol, basis, density_total)?.potential_at_points(points_bohr)
}

/// ESP on a uniform `(nz, ny, nx)` grid (`float32`, Hartree / e), evaluated in parallel across
/// Z-slices without intermediate point-list allocation.
pub fn compute_esp_grid(
    mol: &Molecule,
    basis: &BasisSet,
    density_total: &[f64],
    spec: &GridSpec,
) -> Result<Vec<f32>, CalcError> {
    let [nx, ny, nz] = spec.counts;
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(CalcError::InvalidMolecule(format!(
            "Grid counts must be >= 1 along every axis, got {:?}",
            spec.counts
        )));
    }
    if !spec.origin_bohr.iter().all(|x| x.is_finite())
        || !spec.spacing_bohr.iter().all(|&s| s.is_finite() && s > 0.0)
    {
        return Err(CalcError::InvalidMolecule(
            "Grid origin must be finite and spacing must be strictly positive".into(),
        ));
    }
    let total = nx
        .checked_mul(ny)
        .and_then(|v| v.checked_mul(nz))
        .ok_or_else(|| CalcError::InvalidMolecule("Grid size overflow".into()))?;
    let ev = EspEvaluator::new(mol, basis, density_total)?;
    let boys_len = ev.max_m + 2;
    let slice_len = ny * nx;
    let mut grid = vec![0.0f32; total];
    grid.par_chunks_mut(slice_len)
        .enumerate()
        .for_each_init(
            || vec![0.0f64; boys_len],
            |boys_buf, (iz, z_slice)| {
                let z = spec.origin_bohr[2] + (iz as f64) * spec.spacing_bohr[2];
                for iy in 0..ny {
                    let y = spec.origin_bohr[1] + (iy as f64) * spec.spacing_bohr[1];
                    let row_off = iy * nx;
                    for ix in 0..nx {
                        let x = spec.origin_bohr[0] + (ix as f64) * spec.spacing_bohr[0];
                        z_slice[row_off + ix] = ev.potential_grid_clamped([x, y, z], boys_buf) as f32;
                    }
                }
            },
        );
    Ok(grid)
}

/// Mulliken and Löwdin atomic charges (and optional spin populations).
#[derive(Debug, Clone, PartialEq)]
pub struct PopulationAnalysis {
    pub mulliken_charges: Vec<f64>,
    pub lowdin_charges: Vec<f64>,
    pub mulliken_spin: Option<Vec<f64>>,
    pub lowdin_spin: Option<Vec<f64>>,
}

fn ao_atom_map(basis: &BasisSet) -> Vec<usize> {
    let mut map = vec![0usize; basis.n_ao];
    for sh in &basis.shells {
        for ic in 0..sh.components().len() {
            map[sh.ao_offset + ic] = sh.atom_index;
        }
    }
    map
}

/// Symmetric square root `S^{1/2} = U s^{1/2} U^T` of the (positive definite) overlap matrix.
fn overlap_sqrt(s: &Matrix2D) -> Result<Matrix2D, CalcError> {
    let n = s.rows;
    let (vals, vecs) = eigh_sym(s, 100, 1e-14)?;
    let mut out = Matrix2D::zeros(n, n);
    for (k, &lam) in vals.iter().enumerate() {
        if lam <= 0.0 {
            return Err(CalcError::LinearAlgebra(format!(
                "Overlap matrix is not positive definite (eigenvalue {lam:.3e})"
            )));
        }
        let sq = lam.sqrt();
        for i in 0..n {
            let vik = vecs.get(i, k) * sq;
            if vik == 0.0 {
                continue;
            }
            for j in 0..n {
                out.add_assign_at(i, j, vik * vecs.get(j, k));
            }
        }
    }
    Ok(out)
}

/// Per-atom gross populations: Mulliken `sum_{mu in A} (P S)_{mu mu}` and Löwdin
/// `sum_{mu in A} (S^{1/2} P S^{1/2})_{mu mu}`.
fn gross_populations(
    p: &[f64],
    s: &Matrix2D,
    s_half: &Matrix2D,
    atom_of: &[usize],
    n_atoms: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = s.rows;
    let mut mull = vec![0.0; n_atoms];
    let mut low = vec![0.0; n_atoms];
    // Mulliken
    for mu in 0..n {
        let mut ps = 0.0;
        for nu in 0..n {
            ps += p[mu * n + nu] * s.get(nu, mu);
        }
        mull[atom_of[mu]] += ps;
    }
    // Löwdin: (X P X)_{mu mu} = sum_{a,b} X_{mu a} P_{a b} X_{b mu}
    let mut px = vec![0.0; n * n]; // P X
    for a in 0..n {
        for b in 0..n {
            let pab = p[a * n + b];
            if pab == 0.0 {
                continue;
            }
            for mu in 0..n {
                px[a * n + mu] += pab * s_half.get(b, mu);
            }
        }
    }
    for mu in 0..n {
        let mut d = 0.0;
        for a in 0..n {
            d += s_half.get(mu, a) * px[a * n + mu];
        }
        low[atom_of[mu]] += d;
    }
    (mull, low)
}

/// Mulliken / Löwdin atomic charges `q_A = Z_eff(A) - N_A` (ECP-consistent) and, when a spin
/// density matrix `P^alpha - P^beta` is supplied, atomic spin populations.
pub fn population_analysis(
    mol: &Molecule,
    basis: &BasisSet,
    density_total: &[f64],
    spin_density: Option<&[f64]>,
) -> Result<PopulationAnalysis, CalcError> {
    let n = basis.n_ao;
    if density_total.len() != n * n {
        return Err(CalcError::InvalidMolecule(format!(
            "Density matrix length {} does not match n_ao^2 ({n}^2)",
            density_total.len()
        )));
    }
    if let Some(sd) = spin_density {
        if sd.len() != n * n {
            return Err(CalcError::InvalidMolecule(
                "Spin density matrix length does not match n_ao^2".into(),
            ));
        }
    }
    let s = compute_overlap(basis);
    let s_half = overlap_sqrt(&s)?;
    let atom_of = ao_atom_map(basis);
    let n_atoms = mol.atoms.len();

    let (mull_pop, low_pop) = gross_populations(density_total, &s, &s_half, &atom_of, n_atoms);
    let z: Vec<f64> = (0..n_atoms)
        .map(|i| basis.effective_nuclear_charge(mol, i))
        .collect();
    let mulliken_charges = (0..n_atoms).map(|i| z[i] - mull_pop[i]).collect();
    let lowdin_charges = (0..n_atoms).map(|i| z[i] - low_pop[i]).collect();

    let (mulliken_spin, lowdin_spin) = match spin_density {
        Some(sd) => {
            let (ms, ls) = gross_populations(sd, &s, &s_half, &atom_of, n_atoms);
            (Some(ms), Some(ls))
        }
        None => (None, None),
    };

    Ok(PopulationAnalysis {
        mulliken_charges,
        lowdin_charges,
        mulliken_spin,
        lowdin_spin,
    })
}

/// Isodensity surface with per-vertex ESP scalars (vertex coloring for MolForge / VTK).
#[derive(Debug, Clone, PartialEq)]
pub struct EspSurface {
    pub density_isovalue: f32,
    pub mesh: IsosurfaceMesh,
    /// Per-vertex ESP in Hartree / e (`n_vertices`).
    pub esp: Vec<f32>,
    pub esp_min: f32,
    pub esp_max: f32,
    pub units_angstrom: bool,
}

/// Builds the `rho = density_isovalue` isodensity surface (default convention `0.001` a.u.)
/// and maps the exact ESP onto every vertex.
pub fn compute_esp_surface(
    mol: &Molecule,
    basis: &BasisSet,
    density_total: &[f64],
    density_isovalue: f32,
    spacing_bohr: f64,
    margin_bohr: f64,
    output_angstrom: bool,
) -> Result<EspSurface, CalcError> {
    if !density_isovalue.is_finite() || density_isovalue <= 0.0 {
        return Err(CalcError::InvalidMolecule(format!(
            "Density isovalue must be finite and > 0, got {density_isovalue}"
        )));
    }
    let spec = auto_grid_for_molecule(mol, margin_bohr, spacing_bohr)?;
    let rho = compute_density_grid(
        basis,
        density_total,
        spec.origin_bohr,
        spec.spacing_bohr,
        spec.counts,
    )?;
    let mut mesh = extract_isosurface(
        &rho,
        spec.origin_bohr,
        spec.spacing_bohr,
        spec.counts,
        density_isovalue,
        1.0,
    )?;
    let pts: Vec<[f64; 3]> = mesh
        .vertices
        .chunks_exact(3)
        .map(|v| [v[0] as f64, v[1] as f64, v[2] as f64])
        .collect();
    let esp64 = compute_esp_at_points(mol, basis, density_total, &pts)?;
    let esp: Vec<f32> = esp64.iter().map(|&x| x as f32).collect();
    let esp_min = esp.iter().copied().fold(f32::INFINITY, f32::min);
    let esp_max = esp.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if output_angstrom {
        for v in mesh.vertices.iter_mut() {
            *v = ((*v as f64) * BOHR_TO_ANGSTROM) as f32;
        }
    }
    Ok(EspSurface {
        density_isovalue,
        mesh,
        esp_min: if esp.is_empty() { 0.0 } else { esp_min },
        esp_max: if esp.is_empty() { 0.0 } else { esp_max },
        esp,
        units_angstrom: output_angstrom,
    })
}

/// Encodes an `EspSurface` into the 64-byte-aligned little-endian `CCESP\x01` wire format.
///
/// Binary layout:
/// - `[0..8]`   : `b"CCESP\x01\x00\x00"`
/// - `[8..12]`  : `flags: u32` (bit 0 = `units_angstrom`)
/// - `[12..16]` : `density_isovalue: f32`
/// - `[16..20]` : `n_vertices: u32`
/// - `[20..24]` : `n_triangles: u32`
/// - `[24..28]` : `esp_min: f32` (Hartree / e)
/// - `[28..32]` : `esp_max: f32` (Hartree / e)
/// - `[32..64]` : reserved zero bytes
/// - `[64..]`   : `vertices (3V f32)`, `normals (3V f32)`, `esp (V f32)`, `indices (3F u32)`.
pub fn encode_esp_surface_binary(surf: &EspSurface) -> Vec<u8> {
    let nv = surf.mesh.n_vertices();
    let nt = surf.mesh.n_triangles();
    let mut buf = Vec::with_capacity(CCESP_HEADER_BYTES + (nv * 7 + nt * 3) * 4);
    buf.extend_from_slice(&CCESP_MAGIC);
    buf.extend_from_slice(&u32::from(surf.units_angstrom).to_le_bytes());
    buf.extend_from_slice(&surf.density_isovalue.to_le_bytes());
    buf.extend_from_slice(&(nv as u32).to_le_bytes());
    buf.extend_from_slice(&(nt as u32).to_le_bytes());
    buf.extend_from_slice(&surf.esp_min.to_le_bytes());
    buf.extend_from_slice(&surf.esp_max.to_le_bytes());
    buf.extend_from_slice(&[0u8; 32]);
    debug_assert_eq!(buf.len(), CCESP_HEADER_BYTES);
    for &v in surf
        .mesh
        .vertices
        .iter()
        .chain(&surf.mesh.normals)
        .chain(&surf.esp)
    {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for &i in &surf.mesh.indices {
        buf.extend_from_slice(&i.to_le_bytes());
    }
    buf
}

/// Decodes a `CCESP\x01` packet back into an `EspSurface`.
pub fn decode_esp_surface_binary(bytes: &[u8]) -> Result<EspSurface, CalcError> {
    if bytes.len() < CCESP_HEADER_BYTES {
        return Err(CalcError::InvalidMolecule(format!(
            "CCESP packet too short ({} bytes < {CCESP_HEADER_BYTES})",
            bytes.len()
        )));
    }
    if bytes[0..8] != CCESP_MAGIC {
        return Err(CalcError::InvalidMolecule(
            "Invalid CCESP packet magic or unsupported version".into(),
        ));
    }
    let w = |off: usize| [bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]];
    let flags = u32::from_le_bytes(w(8));
    let iso = f32::from_le_bytes(w(12));
    let nv = u32::from_le_bytes(w(16)) as usize;
    let nt = u32::from_le_bytes(w(20)) as usize;
    let esp_min = f32::from_le_bytes(w(24));
    let esp_max = f32::from_le_bytes(w(28));
    let expected = (nv * 7 + nt * 3)
        .checked_mul(4)
        .and_then(|p| p.checked_add(CCESP_HEADER_BYTES))
        .ok_or_else(|| CalcError::InvalidMolecule("CCESP payload size overflow".into()))?;
    if bytes.len() != expected {
        return Err(CalcError::InvalidMolecule(format!(
            "CCESP packet length mismatch: got {}, expected {expected}",
            bytes.len()
        )));
    }
    let mut cur = CCESP_HEADER_BYTES;
    let read_f32 = |count: usize, cur: &mut usize| -> Vec<f32> {
        let out = (0..count)
            .map(|k| f32::from_le_bytes(w(*cur + 4 * k)))
            .collect();
        *cur += 4 * count;
        out
    };
    let vertices = read_f32(nv * 3, &mut cur);
    let normals = read_f32(nv * 3, &mut cur);
    let esp = read_f32(nv, &mut cur);
    let indices = (0..nt * 3)
        .map(|k| u32::from_le_bytes(w(cur + 4 * k)))
        .collect();
    Ok(EspSurface {
        density_isovalue: iso,
        mesh: IsosurfaceMesh {
            isovalue: iso,
            vertices,
            normals,
            indices,
        },
        esp,
        esp_min,
        esp_max,
        units_angstrom: (flags & 1) != 0,
    })
}
