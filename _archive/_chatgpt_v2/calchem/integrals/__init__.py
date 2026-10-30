"""
CalChem 积分模块。

包含:
- 单电子积分 (动能、势能、重叠)
- 双电子积分 (ERI)
- 积分优化 (Schwarz预筛选、对称压缩)
- 直接SCF
- RI密度拟合
"""

from __future__ import annotations

from .one_electron import (
    compute_overlap,
    compute_kinetic,
    compute_nuclear,
    boys_function,
    boys_function_vec,
)
from .two_electron import (
    compute_eri,
    shell_eri,
    compute_schwarz_bounds,
    schwarz_filter,
)
from .screening import (
    compute_schwarz_screening,
    schwarz_prescreen,
    shell_eri_vectorized,
    RIFitter,
    compute_eri_with_screening,
)
from .direct_scf import (
    build_fock_direct,
    build_j_direct,
    build_k_direct,
    build_fock_direct_uhf,
)
from .ri import (
    RIJEngine,
    RIKEngine,
    build_aux_basis,
    compute_3center_integrals,
    compute_2center_integrals,
    build_coulomb_ri,
    build_exchange_ri,
)

__all__ = [
    'compute_overlap',
    'compute_kinetic',
    'compute_nuclear',
    'boys_function',
    'boys_function_vec',
    'compute_eri',
    'shell_eri',
    'compute_schwarz_bounds',
    'schwarz_filter',
    'compute_schwarz_screening',
    'schwarz_prescreen',
    'shell_eri_vectorized',
    'RIFitter',
    'compute_eri_with_screening',
    'build_fock_direct',
    'build_j_direct',
    'build_k_direct',
    'build_fock_direct_uhf',
    'RIJEngine',
    'RIKEngine',
    'build_aux_basis',
    'compute_3center_integrals',
    'compute_2center_integrals',
    'build_coulomb_ri',
    'build_exchange_ri',
]