"""验证VWN5导数的正确性"""
import numpy as np
import math

def vwn5_ec(rs):
    A = 0.0310907
    b = 0.21370
    c = 7.5957
    d = 3.5876
    e = 1.6382
    
    x = rs ** 0.5
    denom1 = x ** 2 + b * x + c
    denom2 = x ** 2 + d
    
    term1 = np.log(x ** 2 / denom1)
    term2 = 2.0 * b / e * np.arctan(e / x)
    term3 = -(c / e) * np.log(denom1 / denom2)
    
    return A * (term1 + term2 + term3)

def vwn5_vxc(rs):
    A = 0.0310907
    b = 0.21370
    c = 7.5957
    d = 3.5876
    e = 1.6382
    
    x = rs ** 0.5
    denom1 = x ** 2 + b * x + c
    denom2 = x ** 2 + d
    
    term1 = np.log(x ** 2 / denom1)
    term2 = 2.0 * b / e * np.arctan(e / x)
    term3 = -(c / e) * np.log(denom1 / denom2)
    
    ec = A * (term1 + term2 + term3)
    
    dec_dx = A * (
        2.0 / x - (2.0 * x + b) / denom1
        - 2.0 * b / (x ** 2 + e ** 2)
        - (c / e) * ((2.0 * x + b) / denom1 - (2.0 * x) / denom2)
    )
    dec_drs = dec_dx * (0.5 / x)
    
    vxc = ec - rs / 3.0 * dec_drs
    return vxc, ec

print("Testing VWN5 derivative at various rs values:")
print("-" * 70)
print(f"{'rs':>6} {'ec':>12} {'vxc_analytic':>15} {'vxc_numeric':>15} {'diff':>12}")
print("-" * 70)

for rs in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
    ec = vwn5_ec(rs)
    vxc_ana, ec_check = vwn5_vxc(rs)
    
    h = 1e-6
    ec_plus = vwn5_ec(rs + h)
    ec_minus = vwn5_ec(rs - h)
    dec_drs_num = (ec_plus - ec_minus) / (2.0 * h)
    vxc_num = ec - rs / 3.0 * dec_drs_num
    
    diff = vxc_ana - vxc_num
    print(f"{rs:6.2f} {ec:12.6f} {vxc_ana:15.6f} {vxc_num:15.6f} {diff:12.2e}")

print("\nExpected relations:")
print("For homogeneous electron gas:")
print("  vxc = ec + rho * d(ec)/d(rho)")
print("  vxc = ec - (rs/3) * d(ec)/d(rs)")

from calchem.dft.functionals import LDA
func = LDA()
rho = np.array([0.1, 1.0, 10.0])
exc, vxc = func.compute(rho)
print(f"\nLDA functional:")
print(f"  rho = {rho}")
print(f"  exc = {exc}")
print(f"  vxc = {vxc}")
print(f"  vxc/(4/3*exc) = {vxc / ((4/3)*exc)}")
