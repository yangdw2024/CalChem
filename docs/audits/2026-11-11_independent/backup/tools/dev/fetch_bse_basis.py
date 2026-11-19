"""Fetch and normalize authoritative basis sets + ECPs from EMSL Basis Set Exchange API.

Generates embedded JSON data files for `crates/calchem-basis/data/`:
- `sto-3g.json`    (H-Xe, Z=1..54, preserving existing H-Ar bit-for-bit)
- `6-31g.json`     (H-Kr, Z=1..36, preserving existing H-Ar bit-for-bit)
- `6-31gd.json`    (6-31G(d) / 6-31G*, H-Kr, Z=1..36)
- `6-31gdp.json`   (6-31G(d,p) / 6-31G**, H-Kr, Z=1..36)
- `lanl2dz.json`   (LANL2DZ valence + Hay-Wadt scalar ECP, Z<=86)
- `lanl2dzdp.json` (LANL2DZdp polarized + diffuse valence + Hay-Wadt scalar ECP, Z<=86)
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

ELEMENTS = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar",
    19: "K", 20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe",
    27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se",
    35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo",
    43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La", 58: "Ce",
    59: "Pr", 60: "Nd", 61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy",
    67: "Ho", 68: "Er", 69: "Tm", 70: "Yb", 71: "Lu", 72: "Hf", 73: "Ta", 74: "W",
    75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg", 81: "Tl", 82: "Pb",
    83: "Bi", 84: "Po", 85: "At", 86: "Rn",
}

L_TO_SYM = {0: "s", 1: "p", 2: "d", 3: "f"}


def fetch_bse_json(bse_name: str) -> dict:
    quoted = urllib.parse.quote(bse_name, safe="")
    url = f"https://www.basissetexchange.org/api/basis/{quoted}/format/json/"
    req = urllib.request.Request(url, headers={"User-Agent": "CalChem-BasisBuilder/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_electron_shells(bse_shells: list[dict], sym: str) -> list[dict]:
    out = []
    for sh in bse_shells:
        ams = sh["angular_momentum"]
        raw_exps = [float(x) for x in sh["exponents"]]
        coeff_rows = sh["coefficients"]

        if len(ams) == len(coeff_rows):
            pairs = list(zip(ams, coeff_rows))
        elif len(ams) == 1:
            pairs = [(ams[0], row) for row in coeff_rows]
        else:
            raise ValueError(f"Unexpected angular_momentum={ams} vs rows={len(coeff_rows)} for {sym}")

        for l_val, row in pairs:
            if l_val not in L_TO_SYM:
                raise ValueError(f"Unsupported orbital angular momentum l={l_val} on {sym}")
            exps = []
            coeffs = []
            for e_str, c_str in zip(raw_exps, row):
                c_val = float(c_str)
                if abs(c_val) > 0.0:
                    exps.append(float(e_str))
                    coeffs.append(c_val)
            if not exps:
                raise ValueError(f"All-zero contraction row on {sym}")
            out.append({
                "shell_type": L_TO_SYM[l_val],
                "exponents": exps,
                "coefficients": coeffs,
            })
    return out


def parse_ecp(el_data: dict, z: int, sym: str) -> dict | None:
    n_core = int(el_data.get("ecp_electrons", 0) or 0)
    pots = el_data.get("ecp_potentials", [])
    if n_core == 0 and not pots:
        return None
    if n_core <= 0 or n_core >= z:
        raise ValueError(f"Invalid ecp_electrons={n_core} for {sym} (Z={z})")
    if not pots:
        raise ValueError(f"Element {sym} has ecp_electrons={n_core} but no ecp_potentials")

    by_l: dict[int, list[dict]] = {}
    for ep in pots:
        if ep.get("ecp_type", "scalar_ecp") != "scalar_ecp":
            continue
        ams = ep["angular_momentum"]
        if len(ams) != 1:
            raise ValueError(f"Multi-l ECP channel {ams} on {sym}")
        l_val = int(ams[0])
        r_exps = [int(r) for r in ep["r_exponents"]]
        g_exps = [float(g) for g in ep["gaussian_exponents"]]
        c_rows = ep["coefficients"]
        if len(c_rows) != 1:
            raise ValueError(f"Multiple coeff rows in ECP on {sym}")
        coeffs = [float(c) for c in c_rows[0]]
        terms = []
        for r_pow, g_exp, coeff in zip(r_exps, g_exps, coeffs):
            if r_pow < 0:
                raise ValueError(f"Negative r_exponent={r_pow} on {sym}")
            if abs(coeff) > 0.0:
                terms.append({
                    "r_power": r_pow,
                    "exponent": g_exp,
                    "coeff": coeff,
                })
        by_l.setdefault(l_val, []).extend(terms)

    l_max = max(by_l.keys())
    local = by_l[l_max]
    semilocal = [
        {"angular_momentum": l_val, "terms": by_l[l_val]}
        for l_val in sorted(k for k in by_l.keys() if k < l_max)
    ]
    return {
        "core_electrons": n_core,
        "l_max": l_val_check(l_max, sym),
        "local": local,
        "semilocal": semilocal,
    }


def l_val_check(l_max: int, sym: str) -> int:
    if l_max < 0 or l_max > 6:
        raise ValueError(f"Unexpected ECP l_max={l_max} on {sym}")
    return l_max


def build_basis_file(
    bse_name: str,
    canonical_name: str,
    source_tag: str,
    max_z: int,
    preserve_from: Path | None = None,
) -> dict:
    raw_bse = fetch_bse_json(bse_name)
    preserved_elements = {}
    if preserve_from is not None and preserve_from.exists():
        old = json.loads(preserve_from.read_text(encoding="utf-8"))
        preserved_elements = old.get("elements", {})

    elements_out: dict[str, list[dict]] = {}
    ecp_out: dict[str, dict] = {}

    for k_str, el_data in raw_bse["elements"].items():
        z = int(k_str)
        if z < 1 or z > max_z or z not in ELEMENTS:
            continue
        sym = ELEMENTS[z]
        if sym in preserved_elements:
            elements_out[sym] = preserved_elements[sym]
        else:
            if "electron_shells" not in el_data:
                continue
            try:
                elements_out[sym] = parse_electron_shells(el_data["electron_shells"], sym)
            except ValueError:
                # Skip lanthanides/actinides requiring g-orbitals if any
                continue

        ecp_entry = parse_ecp(el_data, z, sym)
        if ecp_entry is not None and sym in elements_out:
            ecp_out[sym] = ecp_entry

    sorted_elements = {k: elements_out[k] for k in sorted(elements_out.keys())}
    result: dict = {
        "name": canonical_name,
        "source": source_tag,
        "elements": sorted_elements,
    }
    if ecp_out:
        result["ecp"] = {k: ecp_out[k] for k in sorted(ecp_out.keys())}
    return result


def main() -> None:
    data_dir = Path(__file__).resolve().parents[2] / "crates" / "calchem-basis" / "data"

    targets = [
        ("sto-3g", "sto-3g", "EMSL-BSE-Hehre1969-Pietro1980-Pietro1981", 54, data_dir / "sto-3g.json", "sto-3g.json"),
        ("6-31g", "6-31g", "EMSL-BSE-HehreDitchfieldPople1972-Francl1982-Rassolov1998", 36, data_dir / "6-31g.json", "6-31g.json"),
        ("6-31g*", "6-31g(d)", "EMSL-BSE-HariharanPople1973-Francl1982-Rassolov1998", 36, None, "6-31gd.json"),
        ("6-31g**", "6-31g(d,p)", "EMSL-BSE-HariharanPople1973-Francl1982-Rassolov1998", 36, None, "6-31gdp.json"),
        ("lanl2dz", "lanl2dz", "EMSL-BSE-DunningHay1977-HayWadt1985", 86, None, "lanl2dz.json"),
        ("lanl2dzdp", "lanl2dzdp", "EMSL-BSE-DunningHay1977-HayWadt1985-Check2001", 86, None, "lanl2dzdp.json"),
    ]

    for bse_name, canon_name, src, max_z, preserve, filename in targets:
        out_data = build_basis_file(bse_name, canon_name, src, max_z, preserve)
        out_path = data_dir / filename
        out_path.write_text(json.dumps(out_data, indent=2) + "\n", encoding="utf-8")
        n_el = len(out_data["elements"])
        n_ecp = len(out_data.get("ecp", {}))
        has_br = "Br" in out_data["elements"]
        has_i = "I" in out_data["elements"]
        print(f"Wrote {filename:15s}: elements={n_el:2d}, ecp={n_ecp:2d}, Br={has_br}, I={has_i}")


if __name__ == "__main__":
    main()
