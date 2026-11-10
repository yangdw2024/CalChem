use std::collections::HashMap;
use std::sync::OnceLock;

use calchem_model::{CalcError, BOHR_TO_ANGSTROM};

/// Wire magic for binary dual-lobe orbital mesh packets (`"CCMSH"` + v1 + 2 pad bytes).
pub const CCMSH_MAGIC: [u8; 8] = *b"CCMSH\x01\x00\x00";
/// Fixed 64-byte cache-line-aligned binary packet header size.
pub const CCMSH_HEADER_BYTES: usize = 64;

/// Single-lobe watertight triangle mesh extracted by Consistent Marching Cubes.
#[derive(Debug, Clone, PartialEq)]
pub struct IsosurfaceMesh {
    pub isovalue: f32,
    /// Flat `[x0, y0, z0, x1, y1, z1, ...]` vertex buffer (`3 * n_vertices`).
    pub vertices: Vec<f32>,
    /// Flat `[nx0, ny0, nz0, ...]` unit outward normal buffer (`3 * n_vertices`).
    pub normals: Vec<f32>,
    /// Flat `[i0, i1, i2, ...]` CCW outward-wound triangle index buffer (`3 * n_triangles`).
    pub indices: Vec<u32>,
}

impl IsosurfaceMesh {
    #[inline]
    pub fn n_vertices(&self) -> usize {
        self.vertices.len() / 3
    }

    #[inline]
    pub fn n_triangles(&self) -> usize {
        self.indices.len() / 3
    }
}

/// Dual-phase (`+isovalue` and `-isovalue`) molecular orbital or spin-density isosurface mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct DualOrbitalMesh {
    pub isovalue: f32,
    pub positive: IsosurfaceMesh,
    pub negative: IsosurfaceMesh,
    pub units_angstrom: bool,
}

/// Cube corner offsets `(dx, dy, dz)` in standard Marching Cubes order `0..8`.
#[cfg(test)]
const CORNER_OFFSETS: [[usize; 3]; 8] = [
    [0, 0, 0], // 0
    [1, 0, 0], // 1
    [1, 1, 0], // 2
    [0, 1, 0], // 3
    [0, 0, 1], // 4
    [1, 0, 1], // 5
    [1, 1, 1], // 6
    [0, 1, 1], // 7
];

/// Cube edges `0..12` as corner pairs `(c0, c1)`.
const CUBE_EDGES: [[u8; 2]; 12] = [
    [0, 1], // 0: bottom -Y (along X)
    [1, 2], // 1: bottom +X (along Y)
    [2, 3], // 2: bottom +Y (along X)
    [3, 0], // 3: bottom -X (along Y)
    [4, 5], // 4: top    -Y (along X)
    [5, 6], // 5: top    +X (along Y)
    [6, 7], // 6: top    +Y (along X)
    [7, 4], // 7: top    -X (along Y)
    [0, 4], // 8: vertical (-X, -Y) (along Z)
    [1, 5], // 9: vertical (+X, -Y) (along Z)
    [2, 6], // 10: vertical (+X, +Y) (along Z)
    [3, 7], // 11: vertical (-X, +Y) (along Z)
];

struct FaceDef {
    /// Four corners in CCW order around the outward cube face normal.
    ccw_corners: [u8; 4],
    /// Cube edge indices `[e01, e12, e23, e30]` matching `ccw_corners` segments.
    ccw_edges: [u8; 4],
    /// Indices into `ccw_corners` (`0..4`) of the two world-canonical diagonal anchor corners
    /// `(00)` and `(11)` in the face's 2D plane (`(x,y)`, `(x,z)`, or `(y,z)`).
    /// Using world-canonical anchors guarantees both cubes sharing a grid face make the exact
    /// same diagonal pairing decision on saddle faces (Montani-Scateni-Scopigno disambiguation).
    canonical_diag_anchors: [usize; 2],
}

const CUBE_FACES: [FaceDef; 6] = [
    // Z = 0 (outward normal -Z): CCW = (0, 3, 2, 1); world (x,y): (00)=0@idx0, (11)=2@idx2
    FaceDef {
        ccw_corners: [0, 3, 2, 1],
        ccw_edges: [3, 2, 1, 0],
        canonical_diag_anchors: [0, 2],
    },
    // Z = 1 (outward normal +Z): CCW = (4, 5, 6, 7); world (x,y): (00)=4@idx0, (11)=6@idx2
    FaceDef {
        ccw_corners: [4, 5, 6, 7],
        ccw_edges: [4, 5, 6, 7],
        canonical_diag_anchors: [0, 2],
    },
    // Y = 0 (outward normal -Y): CCW = (0, 1, 5, 4); world (x,z): (00)=0@idx0, (11)=5@idx2
    FaceDef {
        ccw_corners: [0, 1, 5, 4],
        ccw_edges: [0, 9, 4, 8],
        canonical_diag_anchors: [0, 2],
    },
    // Y = 1 (outward normal +Y): CCW = (3, 7, 6, 2); world (x,z): (00)=3@idx0, (11)=6@idx2
    FaceDef {
        ccw_corners: [3, 7, 6, 2],
        ccw_edges: [11, 6, 10, 2],
        canonical_diag_anchors: [0, 2],
    },
    // X = 0 (outward normal -X): CCW = (0, 4, 7, 3); world (y,z): (00)=0@idx0, (11)=7@idx2
    FaceDef {
        ccw_corners: [0, 4, 7, 3],
        ccw_edges: [8, 7, 11, 3],
        canonical_diag_anchors: [0, 2],
    },
    // X = 1 (outward normal +X): CCW = (1, 2, 6, 5); world (y,z): (00)=1@idx0, (11)=6@idx2
    FaceDef {
        ccw_corners: [1, 2, 6, 5],
        ccw_edges: [1, 10, 5, 9],
        canonical_diag_anchors: [0, 2],
    },
];

struct McTables {
    edge_mask: [u16; 256],
    /// Up to 10 triangles per cube (`12`-edge cycle -> `10` triangles -> `30` indices + `-1` sentinel).
    tri_table: [[i8; 31]; 256],
}

fn build_mc_tables() -> McTables {
    let mut edge_mask = [0u16; 256];
    let mut tri_table = [[-1i8; 31]; 256];

    for mask in 0..256usize {
        let mut inside = [false; 8];
        for c in 0..8 {
            inside[c] = ((mask >> c) & 1) != 0;
        }

        let mut emask = 0u16;
        for (e_idx, &[c0, c1]) in CUBE_EDGES.iter().enumerate() {
            if inside[c0 as usize] != inside[c1 as usize] {
                emask |= 1u16 << e_idx;
            }
        }
        edge_mask[mask] = emask;
        if emask == 0 {
            continue;
        }

        // Directed successor map across active cube edges: next_edge[e_in] = e_out.
        let mut next_edge = [-1i8; 12];

        for face in &CUBE_FACES {
            // Classify the 4 CCW boundary steps k -> (k + 1) % 4:
            //   entry: !inside[a] &&  inside[b]  (0 -> 1)
            //   exit :  inside[a] && !inside[b]  (1 -> 0)
            let mut entries = [0usize; 2];
            let mut exits = [0usize; 2];
            let mut n_en = 0usize;
            let mut n_ex = 0usize;

            for k in 0..4 {
                let a = face.ccw_corners[k] as usize;
                let b = face.ccw_corners[(k + 1) % 4] as usize;
                if !inside[a] && inside[b] {
                    entries[n_en] = k;
                    n_en += 1;
                } else if inside[a] && !inside[b] {
                    exits[n_ex] = k;
                    n_ex += 1;
                }
            }
            debug_assert_eq!(n_en, n_ex);

            if n_en == 1 {
                let e_in = face.ccw_edges[entries[0]] as usize;
                let e_out = face.ccw_edges[exits[0]] as usize;
                next_edge[e_in] = e_out as i8;
            } else if n_en == 2 {
                // Ambiguous saddle face: pair the two edges incident to each canonical diagonal
                // anchor corner `idx` (incoming step `(idx + 3) % 4`, outgoing step `idx`),
                // always directing from the `0 -> 1` edge to the `1 -> 0` edge.
                for &anchor_idx in &face.canonical_diag_anchors {
                    let step_prev = (anchor_idx + 3) % 4;
                    let step_next = anchor_idx;
                    let corner_in = inside[face.ccw_corners[anchor_idx] as usize];
                    // If anchor corner is `inside` (1), `step_prev` is `0 -> 1` (entry) and
                    // `step_next` is `1 -> 0` (exit). Otherwise `step_next` is `0 -> 1` (entry)
                    // and `step_prev` is `1 -> 0` (exit).
                    let (en_step, ex_step) = if corner_in {
                        (step_prev, step_next)
                    } else {
                        (step_next, step_prev)
                    };
                    let e_in = face.ccw_edges[en_step] as usize;
                    let e_out = face.ccw_edges[ex_step] as usize;
                    next_edge[e_in] = e_out as i8;
                }
            }
        }

        // Trace closed directed cycles of active cube edges and fan-triangulate each cycle.
        let mut visited = [false; 12];
        let mut out_pos = 0usize;
        for start in 0..12usize {
            if (emask & (1u16 << start)) == 0 || visited[start] {
                continue;
            }
            let mut cycle = [0i8; 12];
            let mut len = 0usize;
            let mut cur = start;
            while !visited[cur] {
                visited[cur] = true;
                cycle[len] = cur as i8;
                len += 1;
                let nxt = next_edge[cur];
                debug_assert!(nxt >= 0);
                cur = nxt as usize;
            }
            debug_assert_eq!(cur, start);
            debug_assert!(len >= 3);

            for k in 1..(len - 1) {
                tri_table[mask][out_pos] = cycle[0];
                tri_table[mask][out_pos + 1] = cycle[k];
                tri_table[mask][out_pos + 2] = cycle[k + 1];
                out_pos += 3;
            }
        }
    }

    McTables {
        edge_mask,
        tri_table,
    }
}

#[inline]
fn mc_tables() -> &'static McTables {
    static TABLES: OnceLock<McTables> = OnceLock::new();
    TABLES.get_or_init(build_mc_tables)
}

/// Maps local cube edge `(ix, iy, iz, edge_id)` to a canonical world grid edge `(gx, gy, gz, axis)`
/// where `axis` is `0` (X), `1` (Y), or `2` (Z).
#[inline]
fn canonical_grid_edge(ix: usize, iy: usize, iz: usize, edge_id: u8) -> (usize, usize, usize, u8) {
    match edge_id {
        0 => (ix, iy, iz, 0),
        1 => (ix + 1, iy, iz, 1),
        2 => (ix, iy + 1, iz, 0),
        3 => (ix, iy, iz, 1),
        4 => (ix, iy, iz + 1, 0),
        5 => (ix + 1, iy, iz + 1, 1),
        6 => (ix, iy + 1, iz + 1, 0),
        7 => (ix, iy, iz + 1, 1),
        8 => (ix, iy, iz, 2),
        9 => (ix + 1, iy, iz, 2),
        10 => (ix + 1, iy + 1, iz, 2),
        11 => (ix, iy + 1, iz, 2),
        _ => unreachable!(),
    }
}

/// Computes the physical scalar gradient `\nabla f` at grid vertex `(ix, iy, iz)` using
/// second-order central differences in the interior and first-order one-sided differences on boundaries.
#[inline]
fn grid_gradient_at(
    grid: &[f32],
    nx: usize,
    ny: usize,
    nz: usize,
    inv_dx: f64,
    inv_dy: f64,
    inv_dz: f64,
    ix: usize,
    iy: usize,
    iz: usize,
) -> [f64; 3] {
    let idx = |x: usize, y: usize, z: usize| -> usize { (z * ny + y) * nx + x };

    let gx = if ix == 0 {
        (grid[idx(1, iy, iz)] as f64 - grid[idx(0, iy, iz)] as f64) * inv_dx
    } else if ix + 1 == nx {
        (grid[idx(nx - 1, iy, iz)] as f64 - grid[idx(nx - 2, iy, iz)] as f64) * inv_dx
    } else {
        0.5 * (grid[idx(ix + 1, iy, iz)] as f64 - grid[idx(ix - 1, iy, iz)] as f64) * inv_dx
    };

    let gy = if iy == 0 {
        (grid[idx(ix, 1, iz)] as f64 - grid[idx(ix, 0, iz)] as f64) * inv_dy
    } else if iy + 1 == ny {
        (grid[idx(ix, ny - 1, iz)] as f64 - grid[idx(ix, ny - 2, iz)] as f64) * inv_dy
    } else {
        0.5 * (grid[idx(ix, iy + 1, iz)] as f64 - grid[idx(ix, iy - 1, iz)] as f64) * inv_dy
    };

    let gz = if iz == 0 {
        (grid[idx(ix, iy, 1)] as f64 - grid[idx(ix, iy, 0)] as f64) * inv_dz
    } else if iz + 1 == nz {
        (grid[idx(ix, iy, nz - 1)] as f64 - grid[idx(ix, iy, nz - 2)] as f64) * inv_dz
    } else {
        0.5 * (grid[idx(ix, iy, iz + 1)] as f64 - grid[idx(ix, iy, iz - 1)] as f64) * inv_dz
    };

    [gx, gy, gz]
}

/// Extracts a single watertight Marching Cubes isosurface mesh from a `(nz, ny, nx)` scalar grid.
///
/// - When `isovalue >= 0.0`, extracts the superlevel lobe `{ r : f(r) >= isovalue }` with outward
///   normals pointing toward decreasing `f` (`-\nabla f / |\nabla f|`).
/// - When `isovalue < 0.0`, extracts the sublevel lobe `{ r : f(r) <= isovalue }` with outward
///   normals pointing toward increasing `f` (`+\nabla f / |\nabla f|`).
/// - Output vertex positions are multiplied by `coord_scale` (`1.0` keeps grid units; pass
///   `BOHR_TO_ANGSTROM` to convert a Bohr grid directly into Angstrom viewer coordinates).
pub fn extract_isosurface(
    grid: &[f32],
    origin: [f64; 3],
    spacing: [f64; 3],
    counts: [usize; 3], // [nx, ny, nz]
    isovalue: f32,
    coord_scale: f64,
) -> Result<IsosurfaceMesh, CalcError> {
    let [nx, ny, nz] = counts;
    if nx < 2 || ny < 2 || nz < 2 {
        return Err(CalcError::InvalidMolecule(format!(
            "Grid counts must be >= 2 along every axis, got [{nx}, {ny}, {nz}]"
        )));
    }
    let expected_len = nx
        .checked_mul(ny)
        .and_then(|v| v.checked_mul(nz))
        .ok_or_else(|| CalcError::InvalidMolecule("Grid dimensions overflow usize".into()))?;
    if grid.len() != expected_len {
        return Err(CalcError::InvalidMolecule(format!(
            "Grid length {} does not match counts [{nx}, {ny}, {nz}] (expected {expected_len})",
            grid.len()
        )));
    }
    if !spacing.iter().all(|&s| s.is_finite() && s > 0.0) {
        return Err(CalcError::InvalidMolecule(
            "Grid spacing must be finite and strictly positive".into(),
        ));
    }
    if !origin.iter().all(|&o| o.is_finite()) {
        return Err(CalcError::InvalidMolecule(
            "Grid origin must be finite".into(),
        ));
    }
    if !isovalue.is_finite() {
        return Err(CalcError::InvalidMolecule(
            "Isovalue must be finite".into(),
        ));
    }
    if !coord_scale.is_finite() || coord_scale <= 0.0 {
        return Err(CalcError::InvalidMolecule(
            "Coordinate scale factor must be finite and strictly positive".into(),
        ));
    }

    let tables = mc_tables();
    let sublevel = isovalue < 0.0;
    let iso_f64 = isovalue as f64;
    let grad_sign = if sublevel { 1.0 } else { -1.0 };

    let inv_dx = 1.0 / spacing[0];
    let inv_dy = 1.0 / spacing[1];
    let inv_dz = 1.0 / spacing[2];

    let mut edge_to_vertex: HashMap<(usize, usize, usize, u8), u32> = HashMap::new();
    let mut vertices: Vec<f32> = Vec::new();
    let mut normals: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let stride_y = nx;
    let stride_z = nx * ny;

    for iz in 0..(nz - 1) {
        for iy in 0..(ny - 1) {
            let base00 = iz * stride_z + iy * stride_y;
            for ix in 0..(nx - 1) {
                let b = base00 + ix;
                let vals = [
                    grid[b],
                    grid[b + 1],
                    grid[b + stride_y + 1],
                    grid[b + stride_y],
                    grid[b + stride_z],
                    grid[b + stride_z + 1],
                    grid[b + stride_z + stride_y + 1],
                    grid[b + stride_z + stride_y],
                ];

                let mut mask = 0usize;
                if sublevel {
                    for (k, &v) in vals.iter().enumerate() {
                        if v <= isovalue {
                            mask |= 1usize << k;
                        }
                    }
                } else {
                    for (k, &v) in vals.iter().enumerate() {
                        if v >= isovalue {
                            mask |= 1usize << k;
                        }
                    }
                }

                let emask = tables.edge_mask[mask];
                if emask == 0 {
                    continue;
                }

                let mut local_vert_idx = [0u32; 12];
                for e_id in 0..12u8 {
                    if (emask & (1u16 << e_id)) == 0 {
                        continue;
                    }
                    let key = canonical_grid_edge(ix, iy, iz, e_id);
                    if let Some(&v_idx) = edge_to_vertex.get(&key) {
                        local_vert_idx[e_id as usize] = v_idx;
                    } else {
                        let (gx, gy, gz, axis) = key;
                        let (gx1, gy1, gz1) = match axis {
                            0 => (gx + 1, gy, gz),
                            1 => (gx, gy + 1, gz),
                            2 => (gx, gy, gz + 1),
                            _ => unreachable!(),
                        };

                        let v0 = grid[(gz * ny + gy) * nx + gx] as f64;
                        let v1 = grid[(gz1 * ny + gy1) * nx + gx1] as f64;
                        let denom = v1 - v0;
                        let t = if denom.abs() > 1e-20 {
                            ((iso_f64 - v0) / denom).clamp(0.0, 1.0)
                        } else {
                            0.5
                        };

                        let p0 = [
                            origin[0] + (gx as f64) * spacing[0],
                            origin[1] + (gy as f64) * spacing[1],
                            origin[2] + (gz as f64) * spacing[2],
                        ];
                        let mut pos = p0;
                        pos[axis as usize] += t * spacing[axis as usize];

                        let g0 =
                            grid_gradient_at(grid, nx, ny, nz, inv_dx, inv_dy, inv_dz, gx, gy, gz);
                        let g1 = grid_gradient_at(
                            grid, nx, ny, nz, inv_dx, inv_dy, inv_dz, gx1, gy1, gz1,
                        );
                        let mut nx_val = grad_sign * (g0[0] + t * (g1[0] - g0[0]));
                        let mut ny_val = grad_sign * (g0[1] + t * (g1[1] - g0[1]));
                        let mut nz_val = grad_sign * (g0[2] + t * (g1[2] - g0[2]));
                        let norm_len =
                            (nx_val * nx_val + ny_val * ny_val + nz_val * nz_val).sqrt();
                        if norm_len > 1e-20 {
                            let inv_n = 1.0 / norm_len;
                            nx_val *= inv_n;
                            ny_val *= inv_n;
                            nz_val *= inv_n;
                        } else {
                            nx_val = 0.0;
                            ny_val = 0.0;
                            nz_val = 1.0;
                        }

                        let new_idx = u32::try_from(vertices.len() / 3).map_err(|_| {
                            CalcError::Internal("Mesh vertex count exceeded u32::MAX".into())
                        })?;
                        vertices.push((pos[0] * coord_scale) as f32);
                        vertices.push((pos[1] * coord_scale) as f32);
                        vertices.push((pos[2] * coord_scale) as f32);
                        normals.push(nx_val as f32);
                        normals.push(ny_val as f32);
                        normals.push(nz_val as f32);

                        edge_to_vertex.insert(key, new_idx);
                        local_vert_idx[e_id as usize] = new_idx;
                    }
                }

                let tri_row = &tables.tri_table[mask];
                let mut t_pos = 0usize;
                while tri_row[t_pos] >= 0 {
                    let i0 = local_vert_idx[tri_row[t_pos] as usize];
                    let i1 = local_vert_idx[tri_row[t_pos + 1] as usize];
                    let i2 = local_vert_idx[tri_row[t_pos + 2] as usize];
                    indices.push(i0);
                    indices.push(i1);
                    indices.push(i2);
                    t_pos += 3;
                }
            }
        }
    }

    Ok(IsosurfaceMesh {
        isovalue,
        vertices,
        normals,
        indices,
    })
}

/// Parallel dual-lobe extraction (`+isovalue` and `-isovalue`) from a Bohr volumetric grid.
///
/// When `output_angstrom` is `true`, vertex coordinates are scaled by `BOHR_TO_ANGSTROM` so
/// meshes align directly with Angstrom atomic coordinates in MolForge / VTK.
pub fn extract_dual_isosurface(
    grid: &[f32],
    origin_bohr: [f64; 3],
    spacing_bohr: [f64; 3],
    counts: [usize; 3],
    isovalue: f32,
    output_angstrom: bool,
) -> Result<DualOrbitalMesh, CalcError> {
    if !isovalue.is_finite() || isovalue <= 0.0 {
        return Err(CalcError::InvalidMolecule(format!(
            "Dual-lobe isovalue must be finite and strictly positive, got {isovalue}"
        )));
    }
    let scale = if output_angstrom {
        BOHR_TO_ANGSTROM
    } else {
        1.0
    };

    let (pos_res, neg_res) = rayon::join(
        || extract_isosurface(grid, origin_bohr, spacing_bohr, counts, isovalue, scale),
        || extract_isosurface(grid, origin_bohr, spacing_bohr, counts, -isovalue, scale),
    );

    Ok(DualOrbitalMesh {
        isovalue,
        positive: pos_res?,
        negative: neg_res?,
        units_angstrom: output_angstrom,
    })
}

/// Encodes a `DualOrbitalMesh` into the 64-byte-aligned little-endian `CCMSH\x01` binary wire format.
///
/// Binary layout:
/// - `[0..8]`   : `b"CCMSH\x01\x00\x00"`
/// - `[8..12]`  : `flags: u32` (bit 0 = `units_angstrom`, bit 1 = `dual_lobe`)
/// - `[12..16]` : `mo_index: i32` (`-1` for density fields)
/// - `[16..20]` : `spin_channel: u32` (`0` = RHF/alpha/total, `1` = beta, `2` = spin density)
/// - `[20..24]` : `isovalue: f32`
/// - `[24..28]` : `pos_n_verts: u32`
/// - `[28..32]` : `pos_n_tris: u32`
/// - `[32..36]` : `neg_n_verts: u32`
/// - `[36..40]` : `neg_n_tris: u32`
/// - `[40..64]` : reserved zero bytes
/// - `[64..]`   : contiguous `f32`/`u32` arrays in order:
///   `pos.vertices`, `pos.normals`, `pos.indices`, `neg.vertices`, `neg.normals`, `neg.indices`.
pub fn encode_dual_mesh_binary(
    mesh: &DualOrbitalMesh,
    mo_index: i32,
    spin_channel: u32,
) -> Vec<u8> {
    let pv = mesh.positive.n_vertices();
    let pt = mesh.positive.n_triangles();
    let nv = mesh.negative.n_vertices();
    let nt = mesh.negative.n_triangles();

    let payload_bytes = (pv * 6 + pt * 3 + nv * 6 + nt * 3) * 4;
    let mut buf = Vec::with_capacity(CCMSH_HEADER_BYTES + payload_bytes);

    let flags: u32 = u32::from(mesh.units_angstrom) | (1u32 << 1);
    buf.extend_from_slice(&CCMSH_MAGIC);
    buf.extend_from_slice(&flags.to_le_bytes());
    buf.extend_from_slice(&mo_index.to_le_bytes());
    buf.extend_from_slice(&spin_channel.to_le_bytes());
    buf.extend_from_slice(&mesh.isovalue.to_le_bytes());
    buf.extend_from_slice(&(pv as u32).to_le_bytes());
    buf.extend_from_slice(&(pt as u32).to_le_bytes());
    buf.extend_from_slice(&(nv as u32).to_le_bytes());
    buf.extend_from_slice(&(nt as u32).to_le_bytes());
    buf.extend_from_slice(&[0u8; 24]);
    debug_assert_eq!(buf.len(), CCMSH_HEADER_BYTES);

    let push_f32_slice = |out: &mut Vec<u8>, slice: &[f32]| {
        for &v in slice {
            out.extend_from_slice(&v.to_le_bytes());
        }
    };
    let push_u32_slice = |out: &mut Vec<u8>, slice: &[u32]| {
        for &v in slice {
            out.extend_from_slice(&v.to_le_bytes());
        }
    };

    push_f32_slice(&mut buf, &mesh.positive.vertices);
    push_f32_slice(&mut buf, &mesh.positive.normals);
    push_u32_slice(&mut buf, &mesh.positive.indices);
    push_f32_slice(&mut buf, &mesh.negative.vertices);
    push_f32_slice(&mut buf, &mesh.negative.normals);
    push_u32_slice(&mut buf, &mesh.negative.indices);

    buf
}

/// Decodes a `CCMSH\x01` binary packet back into `(DualOrbitalMesh, mo_index, spin_channel)`.
pub fn decode_dual_mesh_binary(bytes: &[u8]) -> Result<(DualOrbitalMesh, i32, u32), CalcError> {
    if bytes.len() < CCMSH_HEADER_BYTES {
        return Err(CalcError::InvalidMolecule(format!(
            "CCMSH packet too short ({} bytes < {CCMSH_HEADER_BYTES})",
            bytes.len()
        )));
    }
    if bytes[0..8] != CCMSH_MAGIC {
        return Err(CalcError::InvalidMolecule(
            "Invalid CCMSH packet magic or unsupported version".into(),
        ));
    }

    let u32_at = |off: usize| -> u32 {
        u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
    };
    let flags = u32_at(8);
    let mo_index = i32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
    let spin_channel = u32_at(16);
    let isovalue = f32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
    let pv = u32_at(24) as usize;
    let pt = u32_at(28) as usize;
    let nv = u32_at(32) as usize;
    let nt = u32_at(36) as usize;

    let expected_payload = (pv * 6 + pt * 3 + nv * 6 + nt * 3)
        .checked_mul(4)
        .ok_or_else(|| CalcError::InvalidMolecule("CCMSH payload size overflow".into()))?;
    if bytes.len() != CCMSH_HEADER_BYTES + expected_payload {
        return Err(CalcError::InvalidMolecule(format!(
            "CCMSH packet length mismatch: got {}, expected {}",
            bytes.len(),
            CCMSH_HEADER_BYTES + expected_payload
        )));
    }

    let mut cursor = CCMSH_HEADER_BYTES;
    let read_f32_vec = |cur: &mut usize, count: usize| -> Vec<f32> {
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let v = f32::from_le_bytes([
                bytes[*cur],
                bytes[*cur + 1],
                bytes[*cur + 2],
                bytes[*cur + 3],
            ]);
            *cur += 4;
            out.push(v);
        }
        out
    };
    let read_u32_vec = |cur: &mut usize, count: usize| -> Vec<u32> {
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let v = u32::from_le_bytes([
                bytes[*cur],
                bytes[*cur + 1],
                bytes[*cur + 2],
                bytes[*cur + 3],
            ]);
            *cur += 4;
            out.push(v);
        }
        out
    };

    let pos_vertices = read_f32_vec(&mut cursor, pv * 3);
    let pos_normals = read_f32_vec(&mut cursor, pv * 3);
    let pos_indices = read_u32_vec(&mut cursor, pt * 3);
    let neg_vertices = read_f32_vec(&mut cursor, nv * 3);
    let neg_normals = read_f32_vec(&mut cursor, nv * 3);
    let neg_indices = read_u32_vec(&mut cursor, nt * 3);

    let units_angstrom = (flags & 1) != 0;
    Ok((
        DualOrbitalMesh {
            isovalue,
            positive: IsosurfaceMesh {
                isovalue,
                vertices: pos_vertices,
                normals: pos_normals,
                indices: pos_indices,
            },
            negative: IsosurfaceMesh {
                isovalue: -isovalue,
                vertices: neg_vertices,
                normals: neg_normals,
                indices: neg_indices,
            },
            units_angstrom,
        },
        mo_index,
        spin_channel,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_256_mc_cases_valid_cycles() {
        let tables = mc_tables();
        assert_eq!(tables.edge_mask[0], 0);
        assert_eq!(tables.edge_mask[255], 0);

        for mask in 1..255usize {
            let emask = tables.edge_mask[mask];
            assert_ne!(emask, 0, "non-trivial mask {mask} must have active edges");

            let row = &tables.tri_table[mask];
            let mut used_mask = 0u16;
            let mut pos = 0usize;
            while row[pos] >= 0 {
                let a = row[pos] as u8;
                let b = row[pos + 1] as u8;
                let c = row[pos + 2] as u8;
                assert!(a < 12 && b < 12 && c < 12);
                assert!(a != b && b != c && a != c);
                used_mask |= (1u16 << a) | (1u16 << b) | (1u16 << c);
                pos += 3;
            }
            assert!(pos >= 3);
            assert_eq!(
                used_mask, emask,
                "mask {mask}: triangulation edges {used_mask:#014b} != active edges {emask:#014b}"
            );
        }

        // Verify corner-0 single-vertex orientation points outward (+1, +1, +1).
        let row1 = &tables.tri_table[1];
        let tri = [row1[0], row1[1], row1[2]];
        assert!(row1[3] == -1);
        // Expected cycle is a cyclic permutation of (0, 3, 8).
        assert!(
            tri == [0, 3, 8] || tri == [3, 8, 0] || tri == [8, 0, 3],
            "unexpected winding for corner 0: {tri:?}"
        );
    }

    #[test]
    fn test_corner_offsets_match_edges() {
        for &[c0, c1] in &CUBE_EDGES {
            let a = CORNER_OFFSETS[c0 as usize];
            let b = CORNER_OFFSETS[c1 as usize];
            let manhattan =
                a[0].abs_diff(b[0]) + a[1].abs_diff(b[1]) + a[2].abs_diff(b[2]);
            assert_eq!(manhattan, 1);
        }
    }
}
