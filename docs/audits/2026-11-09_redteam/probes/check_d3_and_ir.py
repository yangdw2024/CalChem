import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from calchem import _core
import numpy as np

# 1. D3 check
syms = ['O', 'H', 'H']
xyz = [[0.0, 0.0, 0.1173], [0.0, 0.757, -0.4692], [0.0, -0.757, -0.4692]]
coords_bohr = [[c / 0.529177210903 for c in r] for r in xyz]
r_d3 = _core.run_d3_dispersion(syms, coords_bohr, 'b3lyp', 'bj')
g16_d3 = -0.0005738766
diff_d3 = abs(r_d3.energy_hartree - g16_d3)
print(f"New native D3(BJ): {r_d3.energy_hartree:.8f} Eh")
print(f"Gaussian 16 ref:   {g16_d3:.8f} Eh")
print(f"Diff D3:           {diff_d3:.2e} Eh ({diff_d3*1e6:.2f} uEh)")

# 2. IR Intensity check
o = _core.run_optimize(syms, coords_bohr, method='rhf', basis='sto-3g', max_steps=80)
x = np.asarray(o.coords).reshape(3, 3)
r_freq = _core.run_frequency(syms, x.tolist(), method='rhf', basis='sto-3g', step_bohr=0.0003, symmetry_number=2, conv_energy=1e-12, conv_density=1e-10)
g16_ir = [7.2374, 44.2880, 29.9732]
print("\n--- Vibrational Frequencies & IR Intensities ---")
for i, (f, ir, g_ir) in enumerate(zip(r_freq.frequencies_cm, r_freq.ir_intensities_km_mol, g16_ir)):
    print(f"Mode {i+1}: freq = {f:.2f} cm^-1 | native IR = {ir:.4f} km/mol | G16 IR = {g_ir:.4f} km/mol | ratio = {ir/g_ir:.4f}")
