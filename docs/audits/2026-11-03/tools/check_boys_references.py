"""Independent quadrature of Boys integral for the Rust test's pinned cases."""
import json
from pathlib import Path
import re
from scipy.integrate import quad
import numpy as np

audit = Path(__file__).resolve().parents[1]
root = audit.parents[2]
source = (root / 'crates/calchem-int/tests/boys.rs').read_text(encoding='utf-8')
rows = []
for n, t, reference in re.findall(r'\((\d+)usize,\s*([\d.eE+-]+)_f64,\s*([\d.eE+-]+)_f64\)', source):
    n, t, reference = int(n), float(t), float(reference)
    value, error = quad(lambda u: u ** (2 * n) * np.exp(-t * u * u), 0, 1, epsabs=1e-16, epsrel=1e-13)
    rows.append({'n': n, 't': t, 'test_reference': reference, 'quadrature': value,
                 'quad_error_estimate': error, 'reference_minus_quadrature': reference - value,
                 'outside_test_tolerance_1e-14': abs(reference - value) >= 1e-14})
(audit / 'evidence/boys-reference-quadrature.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
print(json.dumps(rows, indent=2))
