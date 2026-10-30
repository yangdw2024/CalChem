"""检查Lebedev网格权重和"""
import numpy as np
import math
from calchem.dft.grid import _lebedev_17, _lebedev_29, _lebedev_41, _lebedev_53

for name, func in [("17", _lebedev_17), ("29", _lebedev_29), ("41", _lebedev_41), ("53", _lebedev_53)]:
    pts, wts = func()
    print(f"degree {name}: n_points={len(wts)}, sum_weights={np.sum(wts):.8f}, 4pi={4*math.pi:.8f}")
