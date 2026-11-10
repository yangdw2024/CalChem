"""Closed-form s/s kinetic integral oracle, independent of Rust recurrence."""
import json
import math
from pathlib import Path

audit = Path(__file__).resolve().parents[1]
root = audit.parents[2]
data = json.loads((root / 'crates/calchem-basis/data/sto-3g.json').read_text())
shell = data['elements']['H'][0]
exps, coeffs = shell['exponents'], shell['coefficients']
norm = [(2 * a / math.pi) ** 0.75 for a in exps]
self_overlap = sum(ci*cj*ni*nj*(math.pi/(a+b))**1.5
    for a, ci, ni in zip(exps, coeffs, norm) for b, cj, nj in zip(exps, coeffs, norm))
distance = 1.4
kinetic = 0.0
for a, ci, ni in zip(exps, coeffs, norm):
    for b, cj, nj in zip(exps, coeffs, norm):
        p, mu = a+b, a*b/(a+b)
        overlap = (math.pi/p)**1.5 * math.exp(-mu*distance**2)
        kinetic += ci*cj*ni*nj*mu*(3-2*mu*distance**2)*overlap/self_overlap
rust = json.loads((audit / 'evidence/inspect-integrals.jsonl').read_text().splitlines()[0])['T01']
out = {'H2_distance_bohr': distance, 'closed_form_T01': kinetic, 'Rust_T01': rust,
       'Rust_minus_closed_form': rust-kinetic, 'existing_test_reference': 0.236454655845882,
       'test_reference_minus_closed_form': 0.236454655845882-kinetic,
       'formula': 'T_ab = mu * (3 - 2*mu*R^2) * S_ab; contracted with normalized source JSON coefficients'}
(audit / 'evidence/kinetic-reference.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
print(json.dumps(out, indent=2))
