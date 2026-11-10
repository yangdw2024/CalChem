"""Independent high-precision references for Rust integral unit tests.

Uses mpmath (50 significant digits) and textbook closed forms only; nothing here
calls CalChem code, so the numbers are an independent oracle.

    F_n(t) = int_0^1 u^(2n) exp(-t u^2) du                       (definition, mp.quad)
    s/s Gaussian overlap / kinetic / (ss|ss) ERI closed forms      (Szabo & Ostlund App. A)

Basis coefficients/exponents are read from crates/calchem-basis/data/sto-3g.json, and
each contracted function is renormalised exactly as the Rust code does.
Run:  python tools/reference/gen_integral_refs.py   (prints JSON)
"""
from __future__ import annotations

import json
from pathlib import Path

import mpmath as mp

mp.mp.dps = 50
ROOT = Path(__file__).resolve().parents[2]


def boys(n: int, t) -> mp.mpf:
    t = mp.mpf(t)
    return mp.quad(lambda u: u ** (2 * n) * mp.e ** (-t * u * u), [0, 1])


def h_sto3g():
    data = json.loads((ROOT / "crates/calchem-basis/data/sto-3g.json").read_text())
    sh = data["elements"]["H"][0]
    exps = [mp.mpf(str(a)) for a in sh["exponents"]]
    cs = [mp.mpf(str(c)) for c in sh["coefficients"]]
    norms = [(2 * a / mp.pi) ** mp.mpf("0.75") for a in exps]
    raw = [c * n for c, n in zip(cs, norms)]
    self_ov = sum(ci * cj * (mp.pi / (a + b)) ** mp.mpf("1.5")
                  for a, ci in zip(exps, raw) for b, cj in zip(exps, raw))
    scale = 1 / mp.sqrt(self_ov)
    return exps, [c * scale for c in raw]


def ss_overlap(a, b, r2):
    p = a + b
    return (mp.pi / p) ** mp.mpf("1.5") * mp.e ** (-a * b / p * r2)


def ss_kinetic(a, b, r2):
    mu = a * b / (a + b)
    return mu * (3 - 2 * mu * r2) * ss_overlap(a, b, r2)


def ss_nuclear(a, A, b, B, C, Z=1):
    p = a + b
    P = [(a * x + b * y) / p for x, y in zip(A, B)]
    rab2 = sum((x - y) ** 2 for x, y in zip(A, B))
    rpc2 = sum((x - y) ** 2 for x, y in zip(P, C))
    return -Z * 2 * mp.pi / p * mp.e ** (-a * b / p * rab2) * boys(0, p * rpc2)


def ss_eri(a, A, b, B, c, C, d, D):
    p, q = a + b, c + d
    P = [(a * x + b * y) / p for x, y in zip(A, B)]
    Q = [(c * x + d * y) / q for x, y in zip(C, D)]
    rab2 = sum((x - y) ** 2 for x, y in zip(A, B))
    rcd2 = sum((x - y) ** 2 for x, y in zip(C, D))
    rpq2 = sum((x - y) ** 2 for x, y in zip(P, Q))
    pref = 2 * mp.pi ** mp.mpf("2.5") / (p * q * mp.sqrt(p + q))
    return pref * mp.e ** (-a * b / p * rab2 - c * d / q * rcd2) * boys(0, p * q / (p + q) * rpq2)


def contracted_eri(centers, idx, exps, cs):
    tot = mp.mpf(0)
    for a, ca in zip(exps, cs):
        for b, cb in zip(exps, cs):
            for c, cc in zip(exps, cs):
                for d, cd in zip(exps, cs):
                    tot += ca * cb * cc * cd * ss_eri(a, centers[idx[0]], b, centers[idx[1]],
                                                      c, centers[idx[2]], d, centers[idx[3]])
    return tot


def main():
    out = {"method": "mpmath dps=50; definition quadrature (Boys) and s-type closed forms"}
    out["boys"] = {f"F{n}({t})": mp.nstr(boys(n, mp.mpf(t)), 20)
                   for n, t in [(0, "1"), (1, "1"), (2, "1"), (0, "25"), (4, "25"),
                                (0, "29.34071349"), (0, "50"), (0, "1e-8")]}
    out["erf"] = {str(x): mp.nstr(mp.erf(mp.mpf(x)), 20) for x in ["0.5", "1", "2", "5"]}

    exps, cs = h_sto3g()
    r = mp.mpf("1.4")
    out["H2_sto3g_R1.4"] = {
        "S01": mp.nstr(sum(ci * cj * ss_overlap(a, b, r * r) for a, ci in zip(exps, cs)
                           for b, cj in zip(exps, cs)), 20),
        "T00": mp.nstr(sum(ci * cj * ss_kinetic(a, b, 0) for a, ci in zip(exps, cs)
                           for b, cj in zip(exps, cs)), 20),
        "T01": mp.nstr(sum(ci * cj * ss_kinetic(a, b, r * r) for a, ci in zip(exps, cs)
                           for b, cj in zip(exps, cs)), 20),
    }
    centers = [[0, 0, 0], [0, 0, r]]
    for name, (i, j) in [("V00", (0, 0)), ("V01", (0, 1))]:
        out["H2_sto3g_R1.4"][name] = mp.nstr(sum(
            ci * cj * ss_nuclear(a, centers[i], b, centers[j], C)
            for a, ci in zip(exps, cs) for b, cj in zip(exps, cs) for C in centers), 20)
    for name, idx in [("0000", (0, 0, 0, 0)), ("0011", (0, 0, 1, 1)),
                      ("0101", (0, 1, 0, 1)), ("0001", (0, 0, 0, 1))]:
        out["H2_sto3g_R1.4"][f"ERI{name}"] = mp.nstr(contracted_eri(centers, idx, exps, cs), 20)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
