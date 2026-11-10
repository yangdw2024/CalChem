//! Exact octahedral Lebedev-Laikov spherical quadrature rules on $S^2$.
//!
//! Reference:
//! - V. I. Lebedev and D. N. Laikov, *Doklady Mathematics* **59**, 477–481 (1999).
//!
//! Weights `w` include the $4\pi$ solid-angle factor ($\sum_i w_i = 4\pi$).

use std::f64::consts::PI;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LebedevDegree {
    /// 26 points, algebraic order L = 7
    N26,
    /// 50 points, algebraic order L = 11
    N50,
    /// 110 points, algebraic order L = 17
    N110,
    /// 194 points, algebraic order L = 23
    N194,
    /// 302 points, algebraic order L = 29
    N302,
}

impl LebedevDegree {
    pub fn max_order(self) -> usize {
        match self {
            Self::N26 => 7,
            Self::N50 => 11,
            Self::N110 => 17,
            Self::N194 => 23,
            Self::N302 => 29,
        }
    }

    pub fn points(self) -> &'static [([f64; 3], f64)] {
        match self {
            Self::N26 => {
                static G: OnceLock<Vec<([f64; 3], f64)>> = OnceLock::new();
                G.get_or_init(|| build_lebedev(Self::N26))
            }
            Self::N50 => {
                static G: OnceLock<Vec<([f64; 3], f64)>> = OnceLock::new();
                G.get_or_init(|| build_lebedev(Self::N50))
            }
            Self::N110 => {
                static G: OnceLock<Vec<([f64; 3], f64)>> = OnceLock::new();
                G.get_or_init(|| build_lebedev(Self::N110))
            }
            Self::N194 => {
                static G: OnceLock<Vec<([f64; 3], f64)>> = OnceLock::new();
                G.get_or_init(|| build_lebedev(Self::N194))
            }
            Self::N302 => {
                static G: OnceLock<Vec<([f64; 3], f64)>> = OnceLock::new();
                G.get_or_init(|| build_lebedev(Self::N302))
            }
        }
    }
}

fn push_octant_signs(out: &mut Vec<([f64; 3], f64)>, x: f64, y: f64, z: f64, w: f64) {
    for sx in [-1.0, 1.0] {
        for sy in [-1.0, 1.0] {
            for sz in [-1.0, 1.0] {
                out.push(([sx * x, sy * y, sz * z], w));
            }
        }
    }
}

fn push_orbit(out: &mut Vec<([f64; 3], f64)>, kind: u8, a: f64, b: f64, v: f64) {
    let w = 4.0 * PI * v;
    match kind {
        // A1: 6 axial vertices (±1, 0, 0)
        1 => {
            for s in [-1.0, 1.0] {
                out.push(([s, 0.0, 0.0], w));
                out.push(([0.0, s, 0.0], w));
                out.push(([0.0, 0.0, s], w));
            }
        }
        // A2: 12 edge midpoints (±1/sqrt(2), ±1/sqrt(2), 0)
        2 => {
            let s = 0.5_f64.sqrt();
            for s1 in [-s, s] {
                for s2 in [-s, s] {
                    out.push(([s1, s2, 0.0], w));
                    out.push(([s1, 0.0, s2], w));
                    out.push(([0.0, s1, s2], w));
                }
            }
        }
        // A3: 8 cube corners (±1/sqrt(3), ±1/sqrt(3), ±1/sqrt(3))
        3 => {
            let s = (1.0_f64 / 3.0).sqrt();
            push_octant_signs(out, s, s, s, w);
        }
        // B: 24 points (±a, ±a, ±b) with b = sqrt(1 - 2 a^2)
        4 => {
            let m = (1.0 - 2.0 * a * a).sqrt();
            push_octant_signs(out, a, a, m, w);
            push_octant_signs(out, a, m, a, w);
            push_octant_signs(out, m, a, a, w);
        }
        // C: 24 planar points (±a, ±b, 0) with b = sqrt(1 - a^2)
        5 => {
            let q = (1.0 - a * a).sqrt();
            for &(u, t) in &[(a, q), (q, a)] {
                for su in [-1.0, 1.0] {
                    for st in [-1.0, 1.0] {
                        out.push(([su * u, st * t, 0.0], w));
                        out.push(([su * u, 0.0, st * t], w));
                        out.push(([0.0, su * u, st * t], w));
                    }
                }
            }
        }
        // D: 48 general points (±a, ±b, ±c) with c = sqrt(1 - a^2 - b^2)
        6 => {
            let c = (1.0 - a * a - b * b).sqrt();
            for &(x, y, z) in &[
                (a, b, c),
                (b, a, c),
                (c, a, b),
                (a, c, b),
                (c, b, a),
                (b, c, a),
            ] {
                push_octant_signs(out, x, y, z, w);
            }
        }
        _ => unreachable!(),
    }
}

fn build_lebedev(deg: LebedevDegree) -> Vec<([f64; 3], f64)> {
    let mut out = Vec::new();
    match deg {
        LebedevDegree::N26 => {
            push_orbit(&mut out, 1, 0.0, 0.0, 0.4761904761904762e-1);
            push_orbit(&mut out, 2, 0.0, 0.0, 0.3809523809523810e-1);
            push_orbit(&mut out, 3, 0.0, 0.0, 0.3214285714285714e-1);
        }
        LebedevDegree::N50 => {
            push_orbit(&mut out, 1, 0.0, 0.0, 0.1269841269841270e-1);
            push_orbit(&mut out, 2, 0.0, 0.0, 0.2257495590828924e-1);
            push_orbit(&mut out, 3, 0.0, 0.0, 0.2109375000000000e-1);
            push_orbit(&mut out, 4, 0.3015113445777636, 0.0, 0.2017333553791887e-1);
        }
        LebedevDegree::N110 => {
            push_orbit(&mut out, 1, 0.0, 0.0, 0.3828270494937162e-2);
            push_orbit(&mut out, 3, 0.0, 0.0, 0.9793737512487512e-2);
            push_orbit(&mut out, 4, 0.1851156353447362, 0.0, 0.8211737283191111e-2);
            push_orbit(&mut out, 4, 0.6904210483822922, 0.0, 0.9942814891178103e-2);
            push_orbit(&mut out, 4, 0.3956894730559419, 0.0, 0.9595471336070963e-2);
            push_orbit(&mut out, 5, 0.4783690288121502, 0.0, 0.9694996361663028e-2);
        }
        LebedevDegree::N194 => {
            push_orbit(&mut out, 1, 0.0, 0.0, 0.1782340447244611e-2);
            push_orbit(&mut out, 2, 0.0, 0.0, 0.5716905949977102e-2);
            push_orbit(&mut out, 3, 0.0, 0.0, 0.5573383178848738e-2);
            push_orbit(&mut out, 4, 0.6712973442695226, 0.0, 0.5608704082587997e-2);
            push_orbit(&mut out, 4, 0.2892465627575439, 0.0, 0.5158237711805383e-2);
            push_orbit(&mut out, 4, 0.4446933178717437, 0.0, 0.5518771467273614e-2);
            push_orbit(&mut out, 4, 0.1299335447650067, 0.0, 0.4106777028169394e-2);
            push_orbit(&mut out, 5, 0.3457702197611283, 0.0, 0.5051846064614808e-2);
            push_orbit(
                &mut out,
                6,
                0.1590417105383530,
                0.8360360154824589,
                0.5530248916233094e-2,
            );
        }
        LebedevDegree::N302 => {
            push_orbit(&mut out, 1, 0.0, 0.0, 0.8545911725128148e-3);
            push_orbit(&mut out, 3, 0.0, 0.0, 0.3599119285025571e-2);
            push_orbit(&mut out, 4, 0.3515640345570105, 0.0, 0.3449788424305883e-2);
            push_orbit(&mut out, 4, 0.6566329410219612, 0.0, 0.3604822601419882e-2);
            push_orbit(&mut out, 4, 0.4729054132581005, 0.0, 0.3576729661743367e-2);
            push_orbit(&mut out, 4, 0.9618308522614784e-1, 0.0, 0.2352101413689164e-2);
            push_orbit(&mut out, 4, 0.2219645236294178, 0.0, 0.3108953122413675e-2);
            push_orbit(&mut out, 4, 0.7011766416089545, 0.0, 0.3650045807677255e-2);
            push_orbit(&mut out, 5, 0.2644152887060663, 0.0, 0.2982344963171804e-2);
            push_orbit(&mut out, 5, 0.5718955891878961, 0.0, 0.3600820932216460e-2);
            push_orbit(
                &mut out,
                6,
                0.2510034751770465,
                0.8000727494073952,
                0.3571540554273387e-2,
            );
            push_orbit(
                &mut out,
                6,
                0.1233548532583327,
                0.4127724083168531,
                0.3392312205006170e-2,
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn double_fact_f64(n: isize) -> f64 {
        if n <= 0 {
            1.0
        } else {
            let mut acc = 1.0;
            let mut k = n;
            while k > 1 {
                acc *= k as f64;
                k -= 2;
            }
            acc
        }
    }

    fn exact_sphere_monomial(p: usize, q: usize, r: usize) -> f64 {
        if p % 2 == 1 || q % 2 == 1 || r % 2 == 1 {
            return 0.0;
        }
        let num = double_fact_f64(p as isize - 1)
            * double_fact_f64(q as isize - 1)
            * double_fact_f64(r as isize - 1);
        let den = double_fact_f64((p + q + r + 1) as isize);
        4.0 * PI * num / den
    }

    #[test]
    fn test_lebedev_monomials_and_exp_benchmark() {
        let specs = [
            (LebedevDegree::N26, 26usize),
            (LebedevDegree::N50, 50),
            (LebedevDegree::N110, 110),
            (LebedevDegree::N194, 194),
            (LebedevDegree::N302, 302),
        ];

        for (deg, expected_len) in specs {
            let pts = deg.points();
            assert_eq!(pts.len(), expected_len);
            for &(u, w) in pts {
                let r2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
                assert!((r2 - 1.0).abs() < 1e-14);
                assert!(w > 0.0);
            }

            let l_max = deg.max_order();
            for p in 0..=l_max {
                for q in 0..=(l_max - p) {
                    for r in 0..=(l_max - p - q) {
                        let num_int: f64 = pts
                            .iter()
                            .map(|&(u, w)| {
                                w * u[0].powi(p as i32) * u[1].powi(q as i32) * u[2].powi(r as i32)
                            })
                            .sum();
                        let exact = exact_sphere_monomial(p, q, r);
                        assert!(
                            (num_int - exact).abs() < 1e-12,
                            "{deg:?} failed at ({p},{q},{r}): got {num_int}, exact {exact}"
                        );
                    }
                }
            }
        }

        // Bellet-Brachet-Croisille benchmark: \int_{S^2} exp(x) d\Omega = 4\pi sinh(1)
        let exact_exp = 4.0 * PI * 1.0_f64.sinh();
        let num_exp: f64 = LebedevDegree::N302
            .points()
            .iter()
            .map(|&(u, w)| w * u[0].exp())
            .sum();
        assert!((num_exp - exact_exp).abs() < 1e-13);
    }
}
