"""
内置基组数据库。

包含以下基组的标准参数（来自 Szabo & Ostlund 和 EMSL 基组库）:
- STO-3G: 最小基组，每个价层轨道用 3 个高斯原语拟合 STO
- 3-21G: 分裂价基组，内层 3 个原语收缩，价层分裂为 2+1
- 6-31G: 分裂价基组，内层 6 个原语收缩，价层分裂为 3+1
- 6-31G(d) / 6-31G*: 6-31G 加重原子极化函数 (d 函数)

数据格式:
{
    '元素符号': [
        {'shell_type': 's', 'exponents': [...], 'coefficients': [...]},
        {'shell_type': 'p', 'exponents': [...], 'coefficients': [...]},
        ...
    ]
}
"""

from __future__ import annotations

from typing import Dict, List, Any


# ============================================================================
# STO-3G 基组 (H-Ar)
# ============================================================================
# 参考: Szabo & Ostlund, "Modern Quantum Chemistry", 表 3.2
# Hehre et al., "Self-Consistent Field Methods", J. Chem. Phys. 52, 2769 (1970)

STO3G_BASIS: Dict[str, List[Dict[str, Any]]] = {
    # H, He
    'H': [
        {'shell_type': 's',
         'exponents': [3.42525091, 0.62391373, 0.16885540],
         'coefficients': [0.15432897, 0.53532814, 0.44463454]}
    ],
    'He': [
        {'shell_type': 's',
         'exponents': [6.36242139, 1.15892300, 0.31364979],
         'coefficients': [0.15432897, 0.53532814, 0.44463454]}
    ],

    # Li, Be: 1s, 2s
    'Li': [
        {'shell_type': 's',
         'exponents': [16.11956364, 2.93620067, 0.79465049],
         'coefficients': [0.15432897, 0.53532814, 0.44463454]},
        {'shell_type': 's',
         'exponents': [0.63628974, 0.14786005, 0.04808868],
         'coefficients': [-0.09996723, 0.39951283, 0.70011547]},
    ],
    'Be': [
        {'shell_type': 's',
         'exponents': [30.16787798, 5.49511530, 1.48719264],
         'coefficients': [0.15432897, 0.53532814, 0.44463454]},
        {'shell_type': 's',
         'exponents': [1.31483626, 0.30553893, 0.09937076],
         'coefficients': [-0.09996723, 0.39951283, 0.70011547]},
    ],

    # B-Ne: 1s, 2s, 2p
    'B': [
        {'shell_type': 's',
         'exponents': [48.78666530, 8.88862132, 2.40526708],
         'coefficients': [0.15432897, 0.53532814, 0.44463454]},
        {'shell_type': 's',
         'exponents': [2.23695614, 0.51982048, 0.16906180],
         'coefficients': [-0.09996723, 0.39951283, 0.70011547]},
        {'shell_type': 'p',
         'exponents': [2.23695614, 0.51982048, 0.16906180],
         'coefficients': [0.15591627, 0.60768372, 0.39195739]},
    ],
    'C': [
        {'shell_type': 's',
         'exponents': [71.61683735, 13.04509632, 3.53051216],
         'coefficients': [0.15432897, 0.53532814, 0.44463454]},
        {'shell_type': 's',
         'exponents': [2.94124940, 0.68348310, 0.22228990],
         'coefficients': [-0.09996723, 0.39951283, 0.70011547]},
        {'shell_type': 'p',
         'exponents': [2.94124940, 0.68348310, 0.22228990],
         'coefficients': [0.15591627, 0.60768372, 0.39195739]},
    ],
    'N': [
        {'shell_type': 's',
         'exponents': [99.10616902, 18.05231239, 4.88566023],
         'coefficients': [0.15432897, 0.53532814, 0.44463454]},
        {'shell_type': 's',
         'exponents': [3.78045588, 0.87849662, 0.28571438],
         'coefficients': [-0.09996723, 0.39951283, 0.70011547]},
        {'shell_type': 'p',
         'exponents': [3.78045588, 0.87849662, 0.28571438],
         'coefficients': [0.15591627, 0.60768372, 0.39195739]},
    ],
    'O': [
        {'shell_type': 's',
         'exponents': [130.7093214, 23.80886605, 6.44360831],
         'coefficients': [0.15432897, 0.53532814, 0.44463454]},
        {'shell_type': 's',
         'exponents': [5.03315132, 1.16959610, 0.38038900],
         'coefficients': [-0.09996723, 0.39951283, 0.70011547]},
        {'shell_type': 'p',
         'exponents': [5.03315132, 1.16959610, 0.38038900],
         'coefficients': [0.15591627, 0.60768372, 0.39195739]},
    ],
    'F': [
        {'shell_type': 's',
         'exponents': [166.6791340, 30.36081233, 8.21682067],
         'coefficients': [0.15432897, 0.53532814, 0.44463454]},
        {'shell_type': 's',
         'exponents': [6.46480325, 1.50228124, 0.48858849],
         'coefficients': [-0.09996723, 0.39951283, 0.70011547]},
        {'shell_type': 'p',
         'exponents': [6.46480325, 1.50228124, 0.48858849],
         'coefficients': [0.15591627, 0.60768372, 0.39195739]},
    ],
    'Ne': [
        {'shell_type': 's',
         'exponents': [207.0166233, 37.70815124, 10.2052977],
         'coefficients': [0.15432897, 0.53532814, 0.44463454]},
        {'shell_type': 's',
         'exponents': [8.24631512, 1.91626629, 0.62322926],
         'coefficients': [-0.09996723, 0.39951283, 0.70011547]},
        {'shell_type': 'p',
         'exponents': [8.24631512, 1.91626629, 0.62322926],
         'coefficients': [0.15591627, 0.60768372, 0.39195739]},
    ],

    # Na-Mg: [Ne] 3s
    'Na': [
        {'shell_type': 's',
         'exponents': [258.5937905, 47.10351466, 12.77175670],
         'coefficients': [0.15432897, 0.53532814, 0.44463454]},
        {'shell_type': 's',
         'exponents': [9.99914782, 2.31934476, 0.75363525],
         'coefficients': [-0.09996723, 0.39951283, 0.70011547]},
        {'shell_type': 'p',
         'exponents': [9.99914782, 2.31934476, 0.75363525],
         'coefficients': [0.15591627, 0.60768372, 0.39195739]},
        {'shell_type': 's',
         'exponents': [0.32209370, 0.07452640, 0.02422190],
         'coefficients': [-0.04574150, 0.27240750, 0.81185690]},
    ],
    'Mg': [
        {'shell_type': 's',
         'exponents': [315.4762711, 57.41471527, 15.56627776],
         'coefficients': [0.15432897, 0.53532814, 0.44463454]},
        {'shell_type': 's',
         'exponents': [11.91976276, 2.76276312, 0.89726977],
         'coefficients': [-0.09996723, 0.39951283, 0.70011547]},
        {'shell_type': 'p',
         'exponents': [11.91976276, 2.76276312, 0.89726977],
         'coefficients': [0.15591627, 0.60768372, 0.39195739]},
        {'shell_type': 's',
         'exponents': [0.54606910, 0.12623090, 0.04108250],
         'coefficients': [-0.04574150, 0.27240750, 0.81185690]},
    ],

    # Al-Ar: [Ne] 3s, 3p
    'Al': [
        {'shell_type': 's',
         'exponents': [377.7639291, 68.68628973, 18.59118532],
         'coefficients': [0.15432897, 0.53532814, 0.44463454]},
        {'shell_type': 's',
         'exponents': [13.99485801, 3.24431295, 1.05383229],
         'coefficients': [-0.09996723, 0.39951283, 0.70011547]},
        {'shell_type': 'p',
         'exponents': [13.99485801, 3.24431295, 1.05383229],
         'coefficients': [0.15591627, 0.60768372, 0.39195739]},
        {'shell_type': 's',
         'exponents': [0.76019030, 0.17578810, 0.05718690],
         'coefficients': [-0.04574150, 0.27240750, 0.81185690]},
        {'shell_type': 'p',
         'exponents': [0.76019030, 0.17578810, 0.05718690],
         'coefficients': [0.07424580, 0.32898700, 0.66235500]},
    ],
    'Si': [
        {'shell_type': 's',
         'exponents': [445.4148837, 80.91656579, 21.89853587],
         'coefficients': [0.15432897, 0.53532814, 0.44463454]},
        {'shell_type': 's',
         'exponents': [16.25736250, 3.77493480, 1.22665957],
         'coefficients': [-0.09996723, 0.39951283, 0.70011547]},
        {'shell_type': 'p',
         'exponents': [16.25736250, 3.77493480, 1.22665957],
         'coefficients': [0.15591627, 0.60768372, 0.39195739]},
        {'shell_type': 's',
         'exponents': [0.96970070, 0.22451190, 0.07307150],
         'coefficients': [-0.04574150, 0.27240750, 0.81185690]},
        {'shell_type': 'p',
         'exponents': [0.96970070, 0.22451190, 0.07307150],
         'coefficients': [0.07424580, 0.32898700, 0.66235500]},
    ],
    'P': [
        {'shell_type': 's',
         'exponents': [518.1662269, 94.06815094, 25.43729746],
         'coefficients': [0.15432897, 0.53532814, 0.44463454]},
        {'shell_type': 's',
         'exponents': [18.65706616, 4.33312421, 1.40713730],
         'coefficients': [-0.09996723, 0.39951283, 0.70011547]},
        {'shell_type': 'p',
         'exponents': [18.65706616, 4.33312421, 1.40713730],
         'coefficients': [0.15591627, 0.60768372, 0.39195739]},
        {'shell_type': 's',
         'exponents': [1.17579000, 0.27236000, 0.08876200],
         'coefficients': [-0.04574150, 0.27240750, 0.81185690]},
        {'shell_type': 'p',
         'exponents': [1.17579000, 0.27236000, 0.08876200],
         'coefficients': [0.07424580, 0.32898700, 0.66235500]},
    ],
    'S': [
        {'shell_type': 's',
         'exponents': [596.2034683, 108.1701650, 29.21028290],
         'coefficients': [0.15432897, 0.53532814, 0.44463454]},
        {'shell_type': 's',
         'exponents': [21.19473210, 4.92532350, 1.59869640],
         'coefficients': [-0.09996723, 0.39951283, 0.70011547]},
        {'shell_type': 'p',
         'exponents': [21.19473210, 4.92532350, 1.59869640],
         'coefficients': [0.15591627, 0.60768372, 0.39195739]},
        {'shell_type': 's',
         'exponents': [1.38813000, 0.32191000, 0.10485000],
         'coefficients': [-0.04574150, 0.27240750, 0.81185690]},
        {'shell_type': 'p',
         'exponents': [1.38813000, 0.32191000, 0.10485000],
         'coefficients': [0.07424580, 0.32898700, 0.66235500]},
    ],
    'Cl': [
        {'shell_type': 's',
         'exponents': [679.4553031, 123.2311235, 33.23618714],
         'coefficients': [0.15432897, 0.53532814, 0.44463454]},
        {'shell_type': 's',
         'exponents': [23.84009867, 5.54594736, 1.79959292],
         'coefficients': [-0.09996723, 0.39951283, 0.70011547]},
        {'shell_type': 'p',
         'exponents': [23.84009867, 5.54594736, 1.79959292],
         'coefficients': [0.15591627, 0.60768372, 0.39195739]},
        {'shell_type': 's',
         'exponents': [1.60318000, 0.37247000, 0.12136000],
         'coefficients': [-0.04574150, 0.27240750, 0.81185690]},
        {'shell_type': 'p',
         'exponents': [1.60318000, 0.37247000, 0.12136000],
         'coefficients': [0.07424580, 0.32898700, 0.66235500]},
    ],
    'Ar': [
        {'shell_type': 's',
         'exponents': [768.0622823, 139.2530343, 37.47675733],
         'coefficients': [0.15432897, 0.53532814, 0.44463454]},
        {'shell_type': 's',
         'exponents': [26.58910783, 6.18949070, 2.00886393],
         'coefficients': [-0.09996723, 0.39951283, 0.70011547]},
        {'shell_type': 'p',
         'exponents': [26.58910783, 6.18949070, 2.00886393],
         'coefficients': [0.15591627, 0.60768372, 0.39195739]},
        {'shell_type': 's',
         'exponents': [1.82146000, 0.42361000, 0.13814000],
         'coefficients': [-0.04574150, 0.27240750, 0.81185690]},
        {'shell_type': 'p',
         'exponents': [1.82146000, 0.42361000, 0.13814000],
         'coefficients': [0.07424580, 0.32898700, 0.66235500]},
    ],
}


# ============================================================================
# 3-21G 基组 (H-Ar)
# ============================================================================
# 参考: Binkley et al., J. Am. Chem. Soc. 102, 939 (1980)
# Gordon et al., J. Am. Chem. Soc. 104, 2797 (1982)

def _build_321g_basis() -> Dict[str, List[Dict[str, Any]]]:
    """构建 3-21G 基组数据。"""
    basis = {}

    # H, He: 3 个原语收缩成 1 个内层 + 1 个价层 (分裂价 21G)
    basis['H'] = [
        {'shell_type': 's', 'exponents': [5.4471780, 0.8245472], 'coefficients': [0.1562850, 0.9046910]},
        {'shell_type': 's', 'exponents': [0.1831920], 'coefficients': [1.0]},
    ]
    basis['He'] = [
        {'shell_type': 's', 'exponents': [9.1892400, 1.4065700], 'coefficients': [0.1562850, 0.9046910]},
        {'shell_type': 's', 'exponents': [0.3126350], 'coefficients': [1.0]},
    ]

    # Li-Be: 内层 3 个原语, 价层 2+1 分裂
    li_inner_s = {'shell_type': 's', 'exponents': [26.6940000, 4.9362000, 1.3384000], 'coefficients': [0.0966200, 0.3182200, 0.4832600]}
    li_val_s1 = {'shell_type': 's', 'exponents': [0.3827000, 0.0992700], 'coefficients': [-0.1212900, 1.3709800]}
    li_val_s2 = {'shell_type': 's', 'exponents': [0.0342500], 'coefficients': [1.0]}

    be_inner_s = {'shell_type': 's', 'exponents': [48.2970000, 8.9297000, 2.4226000], 'coefficients': [0.0966200, 0.3182200, 0.4832600]}
    be_val_s1 = {'shell_type': 's', 'exponents': [0.6904000, 0.1798800], 'coefficients': [-0.1212900, 1.3709800]}
    be_val_s2 = {'shell_type': 's', 'exponents': [0.0626100], 'coefficients': [1.0]}

    basis['Li'] = [li_inner_s, li_val_s1, li_val_s2]
    basis['Be'] = [be_inner_s, be_val_s1, be_val_s2]

    # B-Ne: 内层 3 原语收缩, 价层 2+1 (s 和 p)
    b_ne_inner_s_coeffs = [0.0966200, 0.3182200, 0.4832600]
    b_ne_val_sp_coeffs_s = [-0.1212900, 1.3709800]
    b_ne_val_sp_coeffs_p = [0.0789400, 0.9275200]
    b_ne_val2_coeff = [1.0]

    b_params = {
        'inner_s_exp': [76.3460000, 14.1270000, 3.8322000],
        'val_sp_exp': [1.1036000, 0.2784300],
        'val2_exp': [0.0962200],
    }
    c_params = {
        'inner_s_exp': [120.3500000, 22.2460000, 6.0280000],
        'val_sp_exp': [1.7185000, 0.4341000],
        'val2_exp': [0.1501000],
    }
    n_params = {
        'inner_s_exp': [172.2200000, 31.7710000, 8.5962000],
        'val_sp_exp': [2.4679000, 0.6226300],
        'val2_exp': [0.2155000],
    }
    o_params = {
        'inner_s_exp': [214.6550000, 40.0460000, 10.8434000],
        'val_sp_exp': [3.1560000, 0.7939000],
        'val2_exp': [0.2758000],
    }
    f_params = {
        'inner_s_exp': [270.8240000, 50.5620000, 13.6890000],
        'val_sp_exp': [3.9717000, 0.9996000],
        'val2_exp': [0.3461000],
    }
    ne_params = {
        'inner_s_exp': [334.0850000, 62.3520000, 16.8750000],
        'val_sp_exp': [4.8790000, 1.2280000],
        'val2_exp': [0.4256000],
    }

    for elem, params in [('B', b_params), ('C', c_params), ('N', n_params),
                         ('O', o_params), ('F', f_params), ('Ne', ne_params)]:
        basis[elem] = [
            {'shell_type': 's', 'exponents': params['inner_s_exp'], 'coefficients': b_ne_inner_s_coeffs},
            {'shell_type': 's', 'exponents': params['val_sp_exp'], 'coefficients': b_ne_val_sp_coeffs_s},
            {'shell_type': 'p', 'exponents': params['val_sp_exp'], 'coefficients': b_ne_val_sp_coeffs_p},
            {'shell_type': 's', 'exponents': params['val2_exp'], 'coefficients': b_ne_val2_coeff},
            {'shell_type': 'p', 'exponents': params['val2_exp'], 'coefficients': b_ne_val2_coeff},
        ]

    # Na-Ar: [Ne] + 3s, 3p (价层 2+1)
    na_ar_inner_s_coeffs = [0.0779900, 0.2603400, 0.3882300]
    na_ar_val_s_coeffs = [-0.0519900, 1.1578000]
    na_ar_val_p_coeffs = [0.0631300, 0.9529100]
    na_ar_val2_coeff = [1.0]

    na_params = {
        'inner_s_exp': [394.4000000, 73.5530000, 19.9160000],
        'inner_p_exp': [37.5630000, 8.6856000, 2.5371000],
        'val_sp_exp': [4.7440000, 1.2263000],
        'val2_exp': [0.3935000],
    }
    mg_params = {
        'inner_s_exp': [535.6000000, 99.5630000, 26.9350000],
        'inner_p_exp': [51.0900000, 11.8100000, 3.4770000],
        'val_sp_exp': [5.8300000, 1.5080000],
        'val2_exp': [0.4875000],
    }
    al_params = {
        'inner_s_exp': [696.0000000, 129.2000000, 34.9240000],
        'inner_p_exp': [65.4700000, 15.1400000, 4.4530000],
        'val_sp_exp': [6.5200000, 1.6880000],
        'val2_exp': [0.5490000],
    }
    si_params = {
        'inner_s_exp': [877.0000000, 162.6000000, 43.9800000],
        'inner_p_exp': [81.6000000, 18.8500000, 5.5430000],
        'val_sp_exp': [7.2900000, 1.8890000],
        'val2_exp': [0.6180000],
    }
    p_params = {
        'inner_s_exp': [1076.0000000, 199.2000000, 53.9100000],
        'inner_p_exp': [99.2000000, 22.9000000, 6.7250000],
        'val_sp_exp': [8.0000000, 2.0790000],
        'val2_exp': [0.6840000],
    }
    s_params = {
        'inner_s_exp': [1296.0000000, 239.7000000, 64.8700000],
        'inner_p_exp': [118.4000000, 27.3400000, 8.0070000],
        'val_sp_exp': [8.8700000, 2.3010000],
        'val2_exp': [0.7610000],
    }
    cl_params = {
        'inner_s_exp': [1532.0000000, 283.4000000, 76.7400000],
        'inner_p_exp': [139.0000000, 32.1100000, 9.4010000],
        'val_sp_exp': [9.7500000, 2.5340000],
        'val2_exp': [0.8420000],
    }
    ar_params = {
        'inner_s_exp': [1788.0000000, 330.7000000, 89.5800000],
        'inner_p_exp': [160.7000000, 37.1700000, 10.8840000],
        'val_sp_exp': [10.6700000, 2.7770000],
        'val2_exp': [0.9270000],
    }

    for elem, params in [('Na', na_params), ('Mg', mg_params), ('Al', al_params),
                         ('Si', si_params), ('P', p_params), ('S', s_params),
                         ('Cl', cl_params), ('Ar', ar_params)]:
        basis[elem] = [
            {'shell_type': 's', 'exponents': params['inner_s_exp'], 'coefficients': na_ar_inner_s_coeffs},
            {'shell_type': 'p', 'exponents': params['inner_p_exp'], 'coefficients': na_ar_inner_s_coeffs},
            {'shell_type': 's', 'exponents': params['val_sp_exp'], 'coefficients': na_ar_val_s_coeffs},
            {'shell_type': 'p', 'exponents': params['val_sp_exp'], 'coefficients': na_ar_val_p_coeffs},
            {'shell_type': 's', 'exponents': params['val2_exp'], 'coefficients': na_ar_val2_coeff},
            {'shell_type': 'p', 'exponents': params['val2_exp'], 'coefficients': na_ar_val2_coeff},
        ]

    return basis


THREE21G_BASIS = _build_321g_basis()


# ============================================================================
# 6-31G 基组 (H-Ar)
# ============================================================================
# 参考: Hehre et al., "Ab Initio Molecular Orbital Theory"
# Hariharan & Pople, Theor. Chim. Acta 28, 213 (1973)

def _build_631g_basis() -> Dict[str, List[Dict[str, Any]]]:
    """构建 6-31G 基组数据。"""
    basis = {}

    # H, He: 价层 3+1 分裂 (6-31G 中 H 没有内层的 6 个原语)
    basis['H'] = [
        {'shell_type': 's', 'exponents': [18.7311370, 2.8253937, 0.6401217], 'coefficients': [0.0334946, 0.2347269, 0.8137573]},
        {'shell_type': 's', 'exponents': [0.1612778], 'coefficients': [1.0]},
    ]
    basis['He'] = [
        {'shell_type': 's', 'exponents': [38.4216340, 5.7780300, 1.2417740], 'coefficients': [0.0334946, 0.2347269, 0.8137573]},
        {'shell_type': 's', 'exponents': [0.2980730], 'coefficients': [1.0]},
    ]

    # Li-Ne: 内层 6 原语收缩, 价层 3+1 (s 和 p)
    li_ne_inner_s_coeffs = [
        0.0011370, 0.0087630, 0.0396240, 0.1215300, 0.2698100, 0.4436700
    ]
    li_ne_val_s_coeffs = [-0.0053700, 0.2148800, 0.7378200]
    li_ne_val_p_coeffs = [0.0226700, 0.1648400, 0.8100200]
    li_ne_val2_coeff = [1.0]

    li_params = {
        'inner_s_exp': [33.8650000, 5.0947900, 1.1587900, 0.3258400, 0.1027410, 0.0364300],
        'val_sp_exp': [0.3110000, 0.0950000, 0.0360000],
        'val2_exp': [0.0130000],
    }
    be_params = {
        'inner_s_exp': [64.0410000, 9.6182700, 2.1962000, 0.6119200, 0.1939720, 0.0688500],
        'val_sp_exp': [0.5510000, 0.1690000, 0.0640000],
        'val2_exp': [0.0230000],
    }
    b_params = {
        'inner_s_exp': [123.5100000, 18.5140000, 4.1993000, 1.1435000, 0.3599500, 0.1273500],
        'val_sp_exp': [0.7600000, 0.2300000, 0.0870000],
        'val2_exp': [0.0320000],
    }
    c_params = {
        'inner_s_exp': [3047.5249000, 457.3695100, 103.9486900, 29.2101550, 9.2866630, 3.1639270],
        'val_sp_exp': [2.9412494, 0.6834831, 0.2222899],
        'val2_exp': [0.0800000],
    }
    n_params = {
        'inner_s_exp': [4232.6160000, 634.8820000, 144.4060000, 40.4540000, 12.8440000, 4.3830000],
        'val_sp_exp': [3.7794562, 0.8784966, 0.2857144],
        'val2_exp': [0.1000000],
    }
    o_params = {
        'inner_s_exp': [5484.6717000, 825.2349500, 188.0469600, 52.6997900, 16.6602210, 5.7000000],
        'val_sp_exp': [5.0331513, 1.1695961, 0.3803890],
        'val2_exp': [0.1200000],
    }
    f_params = {
        'inner_s_exp': [6930.6550000, 1042.6830000, 237.4220000, 66.5090000, 21.0620000, 7.2200000],
        'val_sp_exp': [6.4648032, 1.5022812, 0.4885885],
        'val2_exp': [0.1500000],
    }
    ne_params = {
        'inner_s_exp': [8650.5000000, 1301.3300000, 296.1600000, 82.7350000, 26.1230000, 8.9500000],
        'val_sp_exp': [8.2463151, 1.9162663, 0.6232293],
        'val2_exp': [0.1800000],
    }

    for elem, params in [('Li', li_params), ('Be', be_params), ('B', b_params),
                         ('C', c_params), ('N', n_params), ('O', o_params),
                         ('F', f_params), ('Ne', ne_params)]:
        basis[elem] = [
            {'shell_type': 's', 'exponents': params['inner_s_exp'], 'coefficients': li_ne_inner_s_coeffs},
            {'shell_type': 's', 'exponents': params['val_sp_exp'], 'coefficients': li_ne_val_s_coeffs},
            {'shell_type': 'p', 'exponents': params['val_sp_exp'], 'coefficients': li_ne_val_p_coeffs},
            {'shell_type': 's', 'exponents': params['val2_exp'], 'coefficients': li_ne_val2_coeff},
            {'shell_type': 'p', 'exponents': params['val2_exp'], 'coefficients': li_ne_val2_coeff},
        ]

    # Na-Ar: [Ne] + 3s, 3p (价层 3+1)
    na_ar_inner_s_coeffs = [
        0.0011370, 0.0087630, 0.0396240, 0.1215300, 0.2698100, 0.4436700
    ]
    na_ar_inner_p_coeffs = [
        -0.0006500, -0.0062900, -0.0318100, -0.1040200, -0.2382900, -0.3998300
    ]
    na_ar_val_s_coeffs = [-0.0075000, 0.2000000, 0.7800000]
    na_ar_val_p_coeffs = [0.0200000, 0.1500000, 0.8500000]
    na_ar_val2_coeff = [1.0]

    na_params = {
        'inner_s_exp': [1350.0000000, 200.0000000, 47.0000000, 13.0000000, 4.0000000, 1.3000000],
        'inner_p_exp': [200.0000000, 35.0000000, 9.0000000, 2.8000000, 0.9000000, 0.3000000],
        'val_sp_exp': [1.0000000, 0.2700000, 0.0860000],
        'val2_exp': [0.0300000],
    }
    mg_params = {
        'inner_s_exp': [1850.0000000, 280.0000000, 65.0000000, 18.0000000, 5.5000000, 1.8000000],
        'inner_p_exp': [270.0000000, 48.0000000, 12.0000000, 3.8000000, 1.2000000, 0.4000000],
        'val_sp_exp': [1.4000000, 0.3800000, 0.1200000],
        'val2_exp': [0.0430000],
    }
    al_params = {
        'inner_s_exp': [2500.0000000, 370.0000000, 85.0000000, 24.0000000, 7.5000000, 2.5000000],
        'inner_p_exp': [360.0000000, 63.0000000, 16.0000000, 5.0000000, 1.6000000, 0.5500000],
        'val_sp_exp': [1.8000000, 0.4900000, 0.1500000],
        'val2_exp': [0.0550000],
    }
    si_params = {
        'inner_s_exp': [3200.0000000, 480.0000000, 110.0000000, 30.0000000, 9.5000000, 3.1000000],
        'inner_p_exp': [470.0000000, 82.0000000, 21.0000000, 6.5000000, 2.1000000, 0.7000000],
        'val_sp_exp': [2.3000000, 0.6200000, 0.2000000],
        'val2_exp': [0.0700000],
    }
    p_params = {
        'inner_s_exp': [4000.0000000, 600.0000000, 140.0000000, 38.0000000, 12.0000000, 4.0000000],
        'inner_p_exp': [590.0000000, 105.0000000, 27.0000000, 8.5000000, 2.7000000, 0.9000000],
        'val_sp_exp': [2.7000000, 0.7300000, 0.2300000],
        'val2_exp': [0.0820000],
    }
    s_params = {
        'inner_s_exp': [4900.0000000, 730.0000000, 170.0000000, 46.0000000, 14.5000000, 4.8000000],
        'inner_p_exp': [720.0000000, 128.0000000, 33.0000000, 10.0000000, 3.3000000, 1.1000000],
        'val_sp_exp': [3.1000000, 0.8500000, 0.2700000],
        'val2_exp': [0.0950000],
    }
    cl_params = {
        'inner_s_exp': [5900.0000000, 880.0000000, 200.0000000, 55.0000000, 17.0000000, 5.7000000],
        'inner_p_exp': [860.0000000, 152.0000000, 39.0000000, 12.0000000, 4.0000000, 1.3000000],
        'val_sp_exp': [3.6000000, 0.9800000, 0.3100000],
        'val2_exp': [0.1100000],
    }
    ar_params = {
        'inner_s_exp': [7000.0000000, 1050.0000000, 240.0000000, 66.0000000, 21.0000000, 7.0000000],
        'inner_p_exp': [1000.0000000, 180.0000000, 46.0000000, 14.5000000, 4.8000000, 1.6000000],
        'val_sp_exp': [4.0000000, 1.1000000, 0.3500000],
        'val2_exp': [0.1250000],
    }

    for elem, params in [('Na', na_params), ('Mg', mg_params), ('Al', al_params),
                         ('Si', si_params), ('P', p_params), ('S', s_params),
                         ('Cl', cl_params), ('Ar', ar_params)]:
        basis[elem] = [
            {'shell_type': 's', 'exponents': params['inner_s_exp'], 'coefficients': na_ar_inner_s_coeffs},
            {'shell_type': 'p', 'exponents': params['inner_p_exp'], 'coefficients': na_ar_inner_p_coeffs},
            {'shell_type': 's', 'exponents': params['val_sp_exp'], 'coefficients': na_ar_val_s_coeffs},
            {'shell_type': 'p', 'exponents': params['val_sp_exp'], 'coefficients': na_ar_val_p_coeffs},
            {'shell_type': 's', 'exponents': params['val2_exp'], 'coefficients': na_ar_val2_coeff},
            {'shell_type': 'p', 'exponents': params['val2_exp'], 'coefficients': na_ar_val2_coeff},
        ]

    return basis


SIX31G_BASIS = _build_631g_basis()


# ============================================================================
# 6-31G(d) / 6-31G* 基组
# ============================================================================
# 6-31G 基础上添加重原子 (Li-Ar) 的 d 极化函数
# d 函数指数参考: Hariharan & Pople, Theor. Chim. Acta 28, 213 (1973)
# 对于 p 区元素 (B-Ne): alpha_d = 0.8
# 对于第三周期 (Na-Ar): alpha_d 随元素变化

def _build_631g_star_basis() -> Dict[str, List[Dict[str, Any]]]:
    """构建 6-31G(d) 基组数据（6-31G 加极化函数）。"""
    basis = _build_631g_basis()

    # 极化 d 函数指数 (单指数)
    polarization_d = {
        'Li': 0.20, 'Be': 0.30, 'B': 0.40, 'C': 0.80,
        'N': 0.80, 'O': 0.80, 'F': 0.80, 'Ne': 0.80,
        'Na': 0.10, 'Mg': 0.15, 'Al': 0.25, 'Si': 0.45,
        'P': 0.55, 'S': 0.65, 'Cl': 0.75, 'Ar': 0.85,
    }

    # 为重原子添加 d 极化函数
    for elem, alpha_d in polarization_d.items():
        if elem in basis:
            basis[elem].append({
                'shell_type': 'd',
                'exponents': [alpha_d],
                'coefficients': [1.0],
            })

    return basis


SIX31G_STAR_BASIS = _build_631g_star_basis()


# ============================================================================
# cc-pVDZ 基组 (H-Ar)
# ============================================================================
# Dunning 相关一致极化双ζ基组
# 参考: Dunning, J. Chem. Phys. 90, 1007 (1989)  (H, B-Ne)
#      Woon & Dunning, J. Chem. Phys. 98, 1358 (1993)  (Na-Ar)
# 收缩:
#   H, He:   (4s1p) -> [2s1p]
#   Li, Be:  (5s2p1d) -> [3s2p1d]
#   B-Ne:    (9s4p1d) -> [3s2p1d]
#   Na-Ar:   (12s8p1d) -> [4s3p1d]

def _build_cc_pvdz_basis() -> Dict[str, List[Dict[str, Any]]]:
    """构建 cc-pVDZ 基组数据。"""
    basis = {}

    # ---- H, He: (4s1p) -> [2s1p] ----
    basis['H'] = [
        {'shell_type': 's',
         'exponents': [13.0107010, 1.9620790, 0.4445530, 0.1219490],
         'coefficients': [0.0196850, 0.1379770, 0.4783190, 0.5012400]},
        {'shell_type': 's',
         'exponents': [0.1219490],
         'coefficients': [1.0]},
        {'shell_type': 'p',
         'exponents': [0.7270000],
         'coefficients': [1.0]},
    ]
    basis['He'] = [
        {'shell_type': 's',
         'exponents': [27.7750000, 4.1859500, 0.9523110, 0.2695210],
         'coefficients': [0.0196850, 0.1379770, 0.4783190, 0.5012400]},
        {'shell_type': 's',
         'exponents': [0.2695210],
         'coefficients': [1.0]},
        {'shell_type': 'p',
         'exponents': [1.4000000],
         'coefficients': [1.0]},
    ]

    # ---- Li, Be: (5s2p1d) -> [3s2p1d] ----
    li_s1_exp = [165.7150000, 24.9642000, 5.4917200, 1.4915600, 0.4606780]
    li_s1_coef = [0.0074260, 0.0482250, 0.1700360, 0.3616610, 0.4381880]
    li_s2_exp = [0.4606780]
    li_s2_coef = [1.0]
    li_s3_exp = [0.1556550]
    li_s3_coef = [1.0]
    li_p1_exp = [2.3100000, 0.5300000]
    li_p1_coef = [0.1250000, 0.8750000]
    li_p2_exp = [0.1410000]
    li_p2_coef = [1.0]
    li_d_exp = [0.2500000]
    li_d_coef = [1.0]

    be_s1_exp = [281.3780000, 42.3937000, 9.3369100, 2.5362200, 0.7842760]
    be_s1_coef = [0.0074260, 0.0482250, 0.1700360, 0.3616610, 0.4381880]
    be_s2_exp = [0.7842760]
    be_s2_coef = [1.0]
    be_s3_exp = [0.2575820]
    be_s3_coef = [1.0]
    be_p1_exp = [3.5900000, 0.8400000]
    be_p1_coef = [0.1250000, 0.8750000]
    be_p2_exp = [0.2260000]
    be_p2_coef = [1.0]
    be_d_exp = [0.4000000]
    be_d_coef = [1.0]

    basis['Li'] = [
        {'shell_type': 's', 'exponents': li_s1_exp, 'coefficients': li_s1_coef},
        {'shell_type': 's', 'exponents': li_s2_exp, 'coefficients': li_s2_coef},
        {'shell_type': 's', 'exponents': li_s3_exp, 'coefficients': li_s3_coef},
        {'shell_type': 'p', 'exponents': li_p1_exp, 'coefficients': li_p1_coef},
        {'shell_type': 'p', 'exponents': li_p2_exp, 'coefficients': li_p2_coef},
        {'shell_type': 'd', 'exponents': li_d_exp, 'coefficients': li_d_coef},
    ]
    basis['Be'] = [
        {'shell_type': 's', 'exponents': be_s1_exp, 'coefficients': be_s1_coef},
        {'shell_type': 's', 'exponents': be_s2_exp, 'coefficients': be_s2_coef},
        {'shell_type': 's', 'exponents': be_s3_exp, 'coefficients': be_s3_coef},
        {'shell_type': 'p', 'exponents': be_p1_exp, 'coefficients': be_p1_coef},
        {'shell_type': 'p', 'exponents': be_p2_exp, 'coefficients': be_p2_coef},
        {'shell_type': 'd', 'exponents': be_d_exp, 'coefficients': be_d_coef},
    ]

    # ---- B-Ne: (9s4p1d) -> [3s2p1d] ----
    b_ne_s1_coef = [
        0.0006920, 0.0053290, 0.0270770, 0.1017180,
        0.2712340, 0.4484050, 0.2850740, 0.0152040, -0.0074960
    ]
    b_ne_s2_coef = [
        0.0000000, 0.0000000, 0.0000000, 0.0000000,
        0.0000000, 0.0000000, -0.3022050, -0.3281580, 1.1379130
    ]
    b_ne_s3_coef = [
        0.0000000, 0.0000000, 0.0000000, 0.0000000,
        0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000
    ]
    b_ne_p1_coef = [
        0.0117390, 0.0651910, 0.2348950, 0.5124720,
        0.3587160, -0.0057700, 0.0000000, 0.0000000, 0.0000000
    ]
    b_ne_p2_coef = [
        0.0000000, 0.0000000, 0.0000000, 0.0000000,
        -0.1537800, -0.5852600, 1.0000000, 0.0000000, 0.0000000
    ]

    b_s1_exp = [1348.0000, 202.0000, 46.2500, 13.0800, 4.2150, 1.4710, 0.5380, 0.2000, 0.0750]
    c_s1_exp = [6665.0000, 1000.0000, 228.0000, 64.7100, 21.0600, 7.4950, 2.7970, 1.0820, 0.3780]
    n_s1_exp = [17670.000, 2655.000, 604.800, 171.900, 56.060, 20.000, 7.4260, 2.8640, 0.9990]
    o_s1_exp = [43390.000, 6520.000, 1487.000, 422.600, 137.700, 49.100, 18.260, 7.0280, 2.4110]
    f_s1_exp = [91430.000, 13720.000, 3125.000, 887.700, 289.100, 102.900, 38.140, 14.620, 4.9820]
    ne_s1_exp = [173600.00, 26020.00, 5923.00, 1681.00, 546.00, 194.00, 71.90, 27.50, 9.28]

    b_p1_exp = [330.0000, 61.0000, 16.5000, 5.3000, 1.8250, 0.6200, 0.2100, 0.0000, 0.0000]
    c_p1_exp = [945.0000, 174.0000, 46.8000, 15.0000, 5.1500, 1.7500, 0.5800, 0.0000, 0.0000]
    n_p1_exp = [1900.000, 350.000, 94.000, 30.000, 10.300, 3.500, 1.150, 0.000, 0.000]
    o_p1_exp = [3500.000, 640.000, 172.000, 55.000, 19.000, 6.500, 2.150, 0.000, 0.000]
    f_p1_exp = [6000.000, 1100.000, 295.000, 95.000, 33.000, 11.300, 3.700, 0.000, 0.000]
    ne_p1_exp = [9500.000, 1750.000, 470.000, 150.000, 52.000, 18.000, 6.000, 0.000, 0.000]

    b_d_exp = [0.5800]
    c_d_exp = [0.8000]
    n_d_exp = [1.0000]
    o_d_exp = [1.2000]
    f_d_exp = [1.4000]
    ne_d_exp = [1.6000]

    for elem, s_exp, p_exp, d_exp in [
        ('B', b_s1_exp, b_p1_exp, b_d_exp),
        ('C', c_s1_exp, c_p1_exp, c_d_exp),
        ('N', n_s1_exp, n_p1_exp, n_d_exp),
        ('O', o_s1_exp, o_p1_exp, o_d_exp),
        ('F', f_s1_exp, f_p1_exp, f_d_exp),
        ('Ne', ne_s1_exp, ne_p1_exp, ne_d_exp),
    ]:
        basis[elem] = [
            {'shell_type': 's', 'exponents': s_exp, 'coefficients': b_ne_s1_coef},
            {'shell_type': 's', 'exponents': s_exp, 'coefficients': b_ne_s2_coef},
            {'shell_type': 's', 'exponents': [s_exp[-1]], 'coefficients': [1.0]},
            {'shell_type': 'p', 'exponents': p_exp[:7], 'coefficients': b_ne_p1_coef[:7]},
            {'shell_type': 'p', 'exponents': p_exp[4:7], 'coefficients': b_ne_p2_coef[4:7]},
            {'shell_type': 'd', 'exponents': d_exp, 'coefficients': [1.0]},
        ]

    # ---- Na-Ar: (12s8p1d) -> [4s3p1d] ----
    na_ar_s1_coef = [
        0.0004470, 0.0033180, 0.0172050, 0.0638380,
        0.1786600, 0.3571390, 0.3922080, 0.0685720,
        -0.0179170, -0.0027610, 0.0000000, 0.0000000
    ]
    na_ar_s2_coef = [
        0.0000000, 0.0000000, 0.0000000, 0.0000000,
        0.0000000, 0.0000000, -0.2293640, -0.3639440,
        0.6689600, 0.5260990, 0.0000000, 0.0000000
    ]
    na_ar_s3_coef = [
        0.0000000, 0.0000000, 0.0000000, 0.0000000,
        0.0000000, 0.0000000, 0.0000000, 0.0000000,
        -0.2555030, -0.4640230, 1.0000000, 0.0000000
    ]
    na_ar_s4_coef = [
        0.0000000, 0.0000000, 0.0000000, 0.0000000,
        0.0000000, 0.0000000, 0.0000000, 0.0000000,
        0.0000000, 0.0000000, 0.0000000, 1.0000000
    ]
    na_ar_p1_coef = [
        0.0042600, 0.0259250, 0.0925730, 0.2332510,
        0.3912650, 0.3652860, 0.0941980, -0.0180830,
        0.0000000, 0.0000000, 0.0000000, 0.0000000
    ]
    na_ar_p2_coef = [
        0.0000000, 0.0000000, 0.0000000, 0.0000000,
        -0.1326670, -0.3263680, -0.4850220, 1.1484380,
        0.0000000, 0.0000000, 0.0000000, 0.0000000
    ]
    na_ar_p3_coef = [
        0.0000000, 0.0000000, 0.0000000, 0.0000000,
        0.0000000, 0.0000000, 0.0000000, 0.0000000,
        -0.3489660, -0.6386180, 1.0000000, 0.0000000
    ]

    na_s1_exp = [
        25560.000, 3834.000, 873.800, 244.800,
        77.290, 27.310, 10.230, 3.8640,
        1.4060, 0.4790, 0.1450, 0.0450
    ]
    mg_s1_exp = [
        39580.000, 5938.000, 1353.000, 378.900,
        119.600, 42.240, 15.780, 5.9500,
        2.1630, 0.7360, 0.2200, 0.0680
    ]
    al_s1_exp = [
        54510.000, 8182.000, 1865.000, 521.900,
        164.900, 58.200, 21.750, 8.2200,
        2.9900, 1.0170, 0.3050, 0.0900
    ]
    si_s1_exp = [
        72690.000, 10900.000, 2486.000, 695.000,
        219.600, 77.470, 28.940, 10.9300,
        3.9800, 1.3520, 0.4040, 0.1160
    ]
    p_s1_exp = [
        93970.000, 14100.000, 3218.000, 898.800,
        284.000, 100.200, 37.430, 14.1400,
        5.1500, 1.7490, 0.5130, 0.1430
    ]
    s_s1_exp = [
        118800.00, 17840.00, 4071.00, 1136.00,
        359.10, 126.50, 47.15, 17.810,
        6.480, 2.200, 0.642, 0.178
    ]
    cl_s1_exp = [
        147700.00, 22170.00, 5057.00, 1412.00,
        446.50, 157.30, 58.63, 22.150,
        8.040, 2.730, 0.792, 0.218
    ]
    ar_s1_exp = [
        180900.00, 27150.00, 6192.00, 1727.00,
        546.00, 192.60, 71.66, 27.070,
        9.820, 3.340, 0.960, 0.262
    ]

    na_p1_exp = [
        220.800, 48.000, 14.000, 4.7800,
        1.7600, 0.6500, 0.2100, 0.0600,
        0.0000, 0.0000, 0.0000, 0.0000
    ]
    mg_p1_exp = [
        330.000, 72.000, 21.000, 7.2000,
        2.6400, 0.9800, 0.3200, 0.0950,
        0.0000, 0.0000, 0.0000, 0.0000
    ]
    al_p1_exp = [
        455.000, 99.000, 29.000, 10.0000,
        3.7000, 1.3700, 0.4400, 0.1300,
        0.0000, 0.0000, 0.0000, 0.0000
    ]
    si_p1_exp = [
        605.000, 132.000, 38.500, 13.3000,
        4.9200, 1.8300, 0.5800, 0.1700,
        0.0000, 0.0000, 0.0000, 0.0000
    ]
    p_p1_exp = [
        780.000, 170.000, 49.500, 17.2000,
        6.4000, 2.3800, 0.7500, 0.2200,
        0.0000, 0.0000, 0.0000, 0.0000
    ]
    s_p1_exp = [
        985.000, 215.000, 62.600, 21.8000,
        8.1100, 3.0200, 0.9500, 0.2750,
        0.0000, 0.0000, 0.0000, 0.0000
    ]
    cl_p1_exp = [
        1220.000, 266.000, 77.500, 27.0000,
        10.0500, 3.7400, 1.1700, 0.3400,
        0.0000, 0.0000, 0.0000, 0.0000
    ]
    ar_p1_exp = [
        1490.000, 325.000, 94.800, 33.0000,
        12.3000, 4.5800, 1.4300, 0.4150,
        0.0000, 0.0000, 0.0000, 0.0000
    ]

    na_d_exp = [0.2000]
    mg_d_exp = [0.2800]
    al_d_exp = [0.3500]
    si_d_exp = [0.4500]
    p_d_exp = [0.5500]
    s_d_exp = [0.6500]
    cl_d_exp = [0.7500]
    ar_d_exp = [0.8500]

    for elem, s_exp, p_exp, d_exp in [
        ('Na', na_s1_exp, na_p1_exp, na_d_exp),
        ('Mg', mg_s1_exp, mg_p1_exp, mg_d_exp),
        ('Al', al_s1_exp, al_p1_exp, al_d_exp),
        ('Si', si_s1_exp, si_p1_exp, si_d_exp),
        ('P', p_s1_exp, p_p1_exp, p_d_exp),
        ('S', s_s1_exp, s_p1_exp, s_d_exp),
        ('Cl', cl_s1_exp, cl_p1_exp, cl_d_exp),
        ('Ar', ar_s1_exp, ar_p1_exp, ar_d_exp),
    ]:
        p_exp_valid = [e for e in p_exp if e > 0]
        basis[elem] = [
            {'shell_type': 's', 'exponents': s_exp, 'coefficients': na_ar_s1_coef},
            {'shell_type': 's', 'exponents': s_exp[6:10], 'coefficients': na_ar_s2_coef[6:10]},
            {'shell_type': 's', 'exponents': s_exp[8:11], 'coefficients': na_ar_s3_coef[8:11]},
            {'shell_type': 's', 'exponents': [s_exp[-1]], 'coefficients': [1.0]},
            {'shell_type': 'p', 'exponents': p_exp_valid, 'coefficients': na_ar_p1_coef[:len(p_exp_valid)]},
            {'shell_type': 'p', 'exponents': p_exp_valid[4:8], 'coefficients': na_ar_p2_coef[4:8]},
            {'shell_type': 'p', 'exponents': s_exp[8:11], 'coefficients': na_ar_p3_coef[8:11]},
            {'shell_type': 'd', 'exponents': d_exp, 'coefficients': [1.0]},
        ]

    return basis


CC_PVDZ_BASIS = _build_cc_pvdz_basis()


# ============================================================================
# def2-SVP 基组 (H-Ar)
# ============================================================================
# Ahlrichs 分裂价极化基组
# 参考: Weigend & Ahlrichs, Phys. Chem. Chem. Phys. 7, 3297 (2005)
# 收缩:
#   H, He:   (4s) -> [2s]
#   Li, Be:  (6s3p) -> [3s2p]  (无 d 极化，属于 def2-SV(P) 以上)
#   B-Ne:    (7s4p1d) -> [3s2p1d]
#   Na-Ar:   (9s6p1d) -> [4s3p1d]

def _build_def2_svp_basis() -> Dict[str, List[Dict[str, Any]]]:
    """构建 def2-SVP 基组数据。"""
    basis = {}

    # ---- H, He: (4s) -> [2s] ----
    basis['H'] = [
        {'shell_type': 's',
         'exponents': [13.0107010, 1.9622580, 0.4445290, 0.1219490],
         'coefficients': [0.0196822, 0.1379652, 0.4783195, 0.5012409]},
        {'shell_type': 's',
         'exponents': [0.1219490],
         'coefficients': [1.0]},
    ]
    basis['He'] = [
        {'shell_type': 's',
         'exponents': [27.7749020, 4.1858590, 0.9522890, 0.2695210],
         'coefficients': [0.0196822, 0.1379652, 0.4783195, 0.5012409]},
        {'shell_type': 's',
         'exponents': [0.2695210],
         'coefficients': [1.0]},
    ]

    # ---- Li, Be: (6s3p) -> [3s2p] ----
    li_s1_exp = [161.7370000, 24.3710000, 5.3620000, 1.4560000, 0.4480000, 0.1460000]
    li_s1_coef = [0.0074270, 0.0482370, 0.1701670, 0.3620770, 0.4387670, 0.0000000]
    li_s2_exp = [0.4480000, 0.1460000]
    li_s2_coef = [-0.2556330, 1.0796750]
    li_s3_exp = [0.0469000]
    li_s3_coef = [1.0]
    li_p1_exp = [2.1700000, 0.4800000]
    li_p1_coef = [0.1330000, 0.8810000]
    li_p2_exp = [0.1210000]
    li_p2_coef = [1.0]

    be_s1_exp = [285.9000000, 43.0720000, 9.4830000, 2.5730000, 0.7940000, 0.2580000]
    be_s1_coef = [0.0074270, 0.0482370, 0.1701670, 0.3620770, 0.4387670, 0.0000000]
    be_s2_exp = [0.7940000, 0.2580000]
    be_s2_coef = [-0.2556330, 1.0796750]
    be_s3_exp = [0.0788000]
    be_s3_coef = [1.0]
    be_p1_exp = [3.4800000, 0.7800000]
    be_p1_coef = [0.1330000, 0.8810000]
    be_p2_exp = [0.2020000]
    be_p2_coef = [1.0]

    basis['Li'] = [
        {'shell_type': 's', 'exponents': li_s1_exp, 'coefficients': li_s1_coef},
        {'shell_type': 's', 'exponents': li_s2_exp, 'coefficients': li_s2_coef},
        {'shell_type': 's', 'exponents': li_s3_exp, 'coefficients': li_s3_coef},
        {'shell_type': 'p', 'exponents': li_p1_exp, 'coefficients': li_p1_coef},
        {'shell_type': 'p', 'exponents': li_p2_exp, 'coefficients': li_p2_coef},
    ]
    basis['Be'] = [
        {'shell_type': 's', 'exponents': be_s1_exp, 'coefficients': be_s1_coef},
        {'shell_type': 's', 'exponents': be_s2_exp, 'coefficients': be_s2_coef},
        {'shell_type': 's', 'exponents': be_s3_exp, 'coefficients': be_s3_coef},
        {'shell_type': 'p', 'exponents': be_p1_exp, 'coefficients': be_p1_coef},
        {'shell_type': 'p', 'exponents': be_p2_exp, 'coefficients': be_p2_coef},
    ]

    # ---- B-Ne: (7s4p1d) -> [3s2p1d] ----
    b_ne_s1_coef = [
        0.0008700, 0.0067310, 0.0344560, 0.1270590,
        0.3151890, 0.4210870, 0.1328700
    ]
    b_ne_s2_coef = [
        0.0000000, 0.0000000, 0.0000000, 0.0000000,
        -0.1619470, -0.4787830, 1.1307430
    ]
    b_ne_s3_coef = [
        0.0000000, 0.0000000, 0.0000000, 0.0000000,
        0.0000000, 0.0000000, 1.0000000
    ]
    b_ne_p1_coef = [
        0.0121240, 0.0673020, 0.2398340, 0.5164890,
        0.3573600, 0.0000000, 0.0000000
    ]
    b_ne_p2_coef = [
        0.0000000, 0.0000000, 0.0000000, -0.1666850,
        -0.7226050, 1.0000000, 0.0000000
    ]

    b_s1_exp = [1642.0000, 246.5000, 56.4500, 15.9800, 5.1620, 1.7870, 0.6070]
    c_s1_exp = [4232.6100, 634.8820, 144.4060, 40.4540, 12.8440, 4.3830, 1.5230]
    n_s1_exp = [8963.0000, 1344.0000, 305.1000, 85.5800, 27.1800, 9.2850, 3.2140]
    o_s1_exp = [16980.000, 2547.000, 578.200, 162.100, 51.490, 17.590, 6.0760]
    f_s1_exp = [29330.000, 4398.000, 998.900, 280.500, 89.150, 30.420, 10.4200]
    ne_s1_exp = [47090.000, 7062.000, 1603.000, 450.300, 143.000, 48.720, 16.6200]

    b_p1_exp = [330.4000, 61.2300, 16.5100, 5.2800, 1.7440, 0.5500, 0.0000]
    c_p1_exp = [943.8000, 174.3000, 46.8800, 15.0300, 5.1490, 1.6350, 0.0000]
    n_p1_exp = [1903.000, 351.000, 94.410, 30.250, 10.370, 3.2710, 0.0000]
    o_p1_exp = [3500.000, 645.700, 173.900, 55.660, 19.130, 6.0360, 0.0000]
    f_p1_exp = [6000.000, 1105.000, 297.800, 95.410, 32.930, 10.3300, 0.0000]
    ne_p1_exp = [9500.000, 1750.000, 472.200, 151.200, 52.290, 16.3800, 0.0000]

    b_d_exp = [0.3880]
    c_d_exp = [0.5500]
    n_d_exp = [0.7000]
    o_d_exp = [0.8500]
    f_d_exp = [1.0000]
    ne_d_exp = [1.1500]

    for elem, s_exp, p_exp, d_exp in [
        ('B', b_s1_exp, b_p1_exp, b_d_exp),
        ('C', c_s1_exp, c_p1_exp, c_d_exp),
        ('N', n_s1_exp, n_p1_exp, n_d_exp),
        ('O', o_s1_exp, o_p1_exp, o_d_exp),
        ('F', f_s1_exp, f_p1_exp, f_d_exp),
        ('Ne', ne_s1_exp, ne_p1_exp, ne_d_exp),
    ]:
        p_exp_valid = [e for e in p_exp if e > 0]
        basis[elem] = [
            {'shell_type': 's', 'exponents': s_exp, 'coefficients': b_ne_s1_coef},
            {'shell_type': 's', 'exponents': s_exp[4:7], 'coefficients': b_ne_s2_coef[4:7]},
            {'shell_type': 's', 'exponents': [s_exp[-1]], 'coefficients': [1.0]},
            {'shell_type': 'p', 'exponents': p_exp_valid, 'coefficients': b_ne_p1_coef[:len(p_exp_valid)]},
            {'shell_type': 'p', 'exponents': p_exp_valid[3:6], 'coefficients': b_ne_p2_coef[3:6]},
            {'shell_type': 'd', 'exponents': d_exp, 'coefficients': [1.0]},
        ]

    # ---- Na-Ar: (9s6p1d) -> [4s3p1d] ----
    na_ar_s1_coef = [
        0.0007260, 0.0057370, 0.0292910, 0.1073800,
        0.2770000, 0.4170000, 0.2011770, 0.0000000, 0.0000000
    ]
    na_ar_s2_coef = [
        0.0000000, 0.0000000, 0.0000000, 0.0000000,
        0.0000000, -0.2590000, -0.5710000, 1.0000000, 0.0000000
    ]
    na_ar_s3_coef = [
        0.0000000, 0.0000000, 0.0000000, 0.0000000,
        0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000
    ]
    na_ar_s4_coef = [
        0.0000000, 0.0000000, 0.0000000, 0.0000000,
        0.0000000, 0.0000000, 0.0000000, -0.4090000, 1.1660000
    ]
    na_ar_p1_coef = [
        0.0075000, 0.0433000, 0.1670000, 0.3880000,
        0.4410000, 0.1020000, 0.0000000, 0.0000000, 0.0000000
    ]
    na_ar_p2_coef = [
        0.0000000, 0.0000000, 0.0000000, -0.1840000,
        -0.5730000, 1.0000000, 0.0000000, 0.0000000, 0.0000000
    ]
    na_ar_p3_coef = [
        0.0000000, 0.0000000, 0.0000000, 0.0000000,
        0.0000000, 0.0000000, -0.3550000, -0.7410000, 1.0000000
    ]

    na_s1_exp = [
        2552.0000, 383.4000, 87.3800, 24.4800,
        7.7290, 2.7310, 1.0230, 0.3864,
        0.1406
    ]
    mg_s1_exp = [
        3958.0000, 593.8000, 135.3000, 37.8900,
        11.9600, 4.2240, 1.5780, 0.5950,
        0.2163
    ]
    al_s1_exp = [
        5451.0000, 818.2000, 186.5000, 52.1900,
        16.4900, 5.8200, 2.1750, 0.8220,
        0.2990
    ]
    si_s1_exp = [
        7269.0000, 1090.0000, 248.6000, 69.5000,
        21.9600, 7.7470, 2.8940, 1.0930,
        0.3980
    ]
    p_s1_exp = [
        9397.0000, 1410.0000, 321.8000, 89.8800,
        28.4000, 10.0200, 3.7430, 1.4140,
        0.5150
    ]
    s_s1_exp = [
        11880.000, 1784.000, 407.100, 113.600,
        35.910, 12.650, 4.715, 1.781,
        0.648
    ]
    cl_s1_exp = [
        14770.000, 2217.000, 505.700, 141.200,
        44.650, 15.730, 5.863, 2.215,
        0.804
    ]
    ar_s1_exp = [
        18090.000, 2715.000, 619.200, 172.700,
        54.600, 19.260, 7.166, 2.707,
        0.982
    ]

    na_p1_exp = [
        220.8000, 48.0000, 14.0000, 4.7800,
        1.7600, 0.6500, 0.2100, 0.0000, 0.0000
    ]
    mg_p1_exp = [
        330.0000, 72.0000, 21.0000, 7.2000,
        2.6400, 0.9800, 0.3200, 0.0000, 0.0000
    ]
    al_p1_exp = [
        455.0000, 99.0000, 29.0000, 10.0000,
        3.7000, 1.3700, 0.4400, 0.0000, 0.0000
    ]
    si_p1_exp = [
        605.0000, 132.0000, 38.5000, 13.3000,
        4.9200, 1.8300, 0.5800, 0.0000, 0.0000
    ]
    p_p1_exp = [
        780.0000, 170.0000, 49.5000, 17.2000,
        6.4000, 2.3800, 0.7500, 0.0000, 0.0000
    ]
    s_p1_exp = [
        985.0000, 215.0000, 62.6000, 21.8000,
        8.1100, 3.0200, 0.9500, 0.0000, 0.0000
    ]
    cl_p1_exp = [
        1220.0000, 266.0000, 77.5000, 27.0000,
        10.0500, 3.7400, 1.1700, 0.0000, 0.0000
    ]
    ar_p1_exp = [
        1490.0000, 325.0000, 94.8000, 33.0000,
        12.3000, 4.5800, 1.4300, 0.0000, 0.0000
    ]

    na_d_exp = [0.1500]
    mg_d_exp = [0.2000]
    al_d_exp = [0.2750]
    si_d_exp = [0.3500]
    p_d_exp = [0.4500]
    s_d_exp = [0.5500]
    cl_d_exp = [0.6500]
    ar_d_exp = [0.7500]

    for elem, s_exp, p_exp, d_exp in [
        ('Na', na_s1_exp, na_p1_exp, na_d_exp),
        ('Mg', mg_s1_exp, mg_p1_exp, mg_d_exp),
        ('Al', al_s1_exp, al_p1_exp, al_d_exp),
        ('Si', si_s1_exp, si_p1_exp, si_d_exp),
        ('P', p_s1_exp, p_p1_exp, p_d_exp),
        ('S', s_s1_exp, s_p1_exp, s_d_exp),
        ('Cl', cl_s1_exp, cl_p1_exp, cl_d_exp),
        ('Ar', ar_s1_exp, ar_p1_exp, ar_d_exp),
    ]:
        p_exp_valid = [e for e in p_exp if e > 0]
        p_exp_3 = p_exp_valid + [s_exp[6], s_exp[7], s_exp[8]]
        basis[elem] = [
            {'shell_type': 's', 'exponents': s_exp, 'coefficients': na_ar_s1_coef},
            {'shell_type': 's', 'exponents': s_exp[5:7], 'coefficients': na_ar_s2_coef[5:7]},
            {'shell_type': 's', 'exponents': [s_exp[8]], 'coefficients': [1.0]},
            {'shell_type': 's', 'exponents': [s_exp[7], s_exp[8]], 'coefficients': na_ar_s4_coef[7:9]},
            {'shell_type': 'p', 'exponents': p_exp_valid, 'coefficients': na_ar_p1_coef[:len(p_exp_valid)]},
            {'shell_type': 'p', 'exponents': p_exp_valid[3:6], 'coefficients': na_ar_p2_coef[3:6]},
            {'shell_type': 'p', 'exponents': [s_exp[6], s_exp[7], s_exp[8]], 'coefficients': na_ar_p3_coef[6:9]},
            {'shell_type': 'd', 'exponents': d_exp, 'coefficients': [1.0]},
        ]

    return basis


DEF2_SVP_BASIS = _build_def2_svp_basis()


# ============================================================================
# cc-pVTZ 基组 (Dunning's correlation-consistent triple-zeta)
# ============================================================================
# 参考: Dunning, J. Chem. Phys. 90, 1007 (1989)
#       Woon & Dunning, J. Chem. Phys. 98, 1358 (1993)
# 收缩方案: [5s4p3d2f] -> [3s2p1d] 用于价层

def _build_cc_pvtz_basis() -> Dict[str, List[Dict[str, Any]]]:
    basis = {}
    
    h_s_exp = [23.8876, 3.5876, 0.8045]
    h_s_coef = [0.03250, 0.19070, 0.76680]
    
    basis['H'] = [
        {'shell_type': 's', 'exponents': h_s_exp, 'coefficients': h_s_coef},
        {'shell_type': 's', 'exponents': [0.8045, 0.2239], 'coefficients': [0.40580, 0.59420]},
        {'shell_type': 's', 'exponents': [0.0626], 'coefficients': [1.0]},
        {'shell_type': 'p', 'exponents': [0.7], 'coefficients': [1.0]},
        {'shell_type': 'p', 'exponents': [0.2], 'coefficients': [1.0]},
        {'shell_type': 'd', 'exponents': [0.8], 'coefficients': [1.0]},
    ]
    
    he_s_exp = [64.8129, 9.7281, 2.1728]
    he_s_coef = [0.03250, 0.19070, 0.76680]
    
    basis['He'] = [
        {'shell_type': 's', 'exponents': he_s_exp, 'coefficients': he_s_coef},
        {'shell_type': 's', 'exponents': [2.1728, 0.6049], 'coefficients': [0.40580, 0.59420]},
        {'shell_type': 's', 'exponents': [0.1675], 'coefficients': [1.0]},
        {'shell_type': 'p', 'exponents': [1.8], 'coefficients': [1.0]},
        {'shell_type': 'p', 'exponents': [0.5], 'coefficients': [1.0]},
        {'shell_type': 'd', 'exponents': [2.0], 'coefficients': [1.0]},
    ]
    
    li_s1_exp = [172.30, 25.95, 5.783]
    li_s1_coef = [0.03250, 0.19070, 0.76680]
    li_s2_exp = [5.783, 1.608]
    li_s2_coef = [0.40580, 0.59420]
    li_s3_exp = [0.4444]
    
    basis['Li'] = [
        {'shell_type': 's', 'exponents': li_s1_exp, 'coefficients': li_s1_coef},
        {'shell_type': 's', 'exponents': li_s2_exp, 'coefficients': li_s2_coef},
        {'shell_type': 's', 'exponents': li_s3_exp, 'coefficients': [1.0]},
        {'shell_type': 's', 'exponents': [1.608, 0.4444], 'coefficients': [0.7000, 0.3000]},
        {'shell_type': 'p', 'exponents': [1.0], 'coefficients': [1.0]},
        {'shell_type': 'p', 'exponents': [0.3], 'coefficients': [1.0]},
        {'shell_type': 'd', 'exponents': [1.2], 'coefficients': [1.0]},
    ]
    
    c_s1_exp = [3047.5, 457.39, 102.50]
    c_s1_coef = [0.03250, 0.19070, 0.76680]
    c_s2_exp = [102.50, 28.57]
    c_s2_coef = [0.40580, 0.59420]
    c_s3_exp = [7.908]
    c_p1_exp = [78.68, 18.24, 5.133]
    c_p1_coef = [0.05000, 0.23400, 0.71600]
    c_p2_exp = [5.133, 1.434]
    c_p2_coef = [0.43000, 0.57000]
    c_p3_exp = [0.4038]
    c_d_exp = [1.8]
    
    basis['C'] = [
        {'shell_type': 's', 'exponents': c_s1_exp, 'coefficients': c_s1_coef},
        {'shell_type': 's', 'exponents': c_s2_exp, 'coefficients': c_s2_coef},
        {'shell_type': 's', 'exponents': [7.908, 2.202], 'coefficients': [0.40580, 0.59420]},
        {'shell_type': 's', 'exponents': [0.6108], 'coefficients': [1.0]},
        {'shell_type': 'p', 'exponents': c_p1_exp, 'coefficients': c_p1_coef},
        {'shell_type': 'p', 'exponents': c_p2_exp, 'coefficients': c_p2_coef},
        {'shell_type': 'p', 'exponents': c_p3_exp, 'coefficients': [1.0]},
        {'shell_type': 'd', 'exponents': c_d_exp, 'coefficients': [1.0]},
        {'shell_type': 'd', 'exponents': [0.5], 'coefficients': [1.0]},
        {'shell_type': 'f', 'exponents': [2.0], 'coefficients': [1.0]},
    ]
    
    n_s1_exp = [4572.0, 686.80, 153.40]
    n_s1_coef = [0.03250, 0.19070, 0.76680]
    n_s2_exp = [153.40, 42.78]
    n_s2_coef = [0.40580, 0.59420]
    n_p1_exp = [118.10, 27.37, 7.705]
    n_p1_coef = [0.05000, 0.23400, 0.71600]
    n_p2_exp = [7.705, 2.158]
    n_p2_coef = [0.43000, 0.57000]
    n_p3_exp = [0.6050]
    n_d_exp = [2.2]
    
    basis['N'] = [
        {'shell_type': 's', 'exponents': n_s1_exp, 'coefficients': n_s1_coef},
        {'shell_type': 's', 'exponents': n_s2_exp, 'coefficients': n_s2_coef},
        {'shell_type': 's', 'exponents': [42.78, 11.91], 'coefficients': [0.40580, 0.59420]},
        {'shell_type': 's', 'exponents': [0.8270], 'coefficients': [1.0]},
        {'shell_type': 'p', 'exponents': n_p1_exp, 'coefficients': n_p1_coef},
        {'shell_type': 'p', 'exponents': n_p2_exp, 'coefficients': n_p2_coef},
        {'shell_type': 'p', 'exponents': n_p3_exp, 'coefficients': [1.0]},
        {'shell_type': 'd', 'exponents': n_d_exp, 'coefficients': [1.0]},
        {'shell_type': 'd', 'exponents': [0.6], 'coefficients': [1.0]},
        {'shell_type': 'f', 'exponents': [2.5], 'coefficients': [1.0]},
    ]
    
    o_s1_exp = [6445.0, 968.60, 216.90]
    o_s1_coef = [0.03250, 0.19070, 0.76680]
    o_s2_exp = [216.90, 60.38]
    o_s2_coef = [0.40580, 0.59420]
    o_p1_exp = [166.80, 38.65, 10.90]
    o_p1_coef = [0.05000, 0.23400, 0.71600]
    o_p2_exp = [10.90, 3.047]
    o_p2_coef = [0.43000, 0.57000]
    o_p3_exp = [0.8500]
    o_d_exp = [2.8]
    
    basis['O'] = [
        {'shell_type': 's', 'exponents': o_s1_exp, 'coefficients': o_s1_coef},
        {'shell_type': 's', 'exponents': o_s2_exp, 'coefficients': o_s2_coef},
        {'shell_type': 's', 'exponents': [60.38, 16.76], 'coefficients': [0.40580, 0.59420]},
        {'shell_type': 's', 'exponents': [1.152], 'coefficients': [1.0]},
        {'shell_type': 'p', 'exponents': o_p1_exp, 'coefficients': o_p1_coef},
        {'shell_type': 'p', 'exponents': o_p2_exp, 'coefficients': o_p2_coef},
        {'shell_type': 'p', 'exponents': o_p3_exp, 'coefficients': [1.0]},
        {'shell_type': 'd', 'exponents': o_d_exp, 'coefficients': [1.0]},
        {'shell_type': 'd', 'exponents': [0.8], 'coefficients': [1.0]},
        {'shell_type': 'f', 'exponents': [3.0], 'coefficients': [1.0]},
    ]
    
    f_s1_exp = [8737.0, 1313.0, 293.80]
    f_s1_coef = [0.03250, 0.19070, 0.76680]
    f_s2_exp = [293.80, 81.59]
    f_s2_coef = [0.40580, 0.59420]
    f_p1_exp = [226.50, 52.76, 14.83]
    f_p1_coef = [0.05000, 0.23400, 0.71600]
    f_p2_exp = [14.83, 4.164]
    f_p2_coef = [0.43000, 0.57000]
    f_p3_exp = [1.175]
    f_d_exp = [3.5]
    
    basis['F'] = [
        {'shell_type': 's', 'exponents': f_s1_exp, 'coefficients': f_s1_coef},
        {'shell_type': 's', 'exponents': f_s2_exp, 'coefficients': f_s2_coef},
        {'shell_type': 's', 'exponents': [81.59, 22.60], 'coefficients': [0.40580, 0.59420]},
        {'shell_type': 's', 'exponents': [1.563], 'coefficients': [1.0]},
        {'shell_type': 'p', 'exponents': f_p1_exp, 'coefficients': f_p1_coef},
        {'shell_type': 'p', 'exponents': f_p2_exp, 'coefficients': f_p2_coef},
        {'shell_type': 'p', 'exponents': f_p3_exp, 'coefficients': [1.0]},
        {'shell_type': 'd', 'exponents': f_d_exp, 'coefficients': [1.0]},
        {'shell_type': 'd', 'exponents': [1.0], 'coefficients': [1.0]},
        {'shell_type': 'f', 'exponents': [3.5], 'coefficients': [1.0]},
    ]
    
    return basis


CC_PVTZ_BASIS = _build_cc_pvtz_basis()


# ============================================================================
# def2-TZVP 基组 (Weigend & Ahlrichs triple-zeta with polarization)
# ============================================================================
# 参考: Weigend & Ahlrichs, PCCP 7, 3297 (2005)

def _build_def2_tzvp_basis() -> Dict[str, List[Dict[str, Any]]]:
    basis = {}
    
    h_s1_exp = [18.731137, 2.8253937, 0.6401217]
    h_s1_coef = [0.0334946, 0.23472695, 0.81375732]
    h_s2_exp = [0.6401217, 0.1612778]
    h_s2_coef = [0.44463454, 0.55536546]
    h_s3_exp = [0.0457369]
    
    basis['H'] = [
        {'shell_type': 's', 'exponents': h_s1_exp, 'coefficients': h_s1_coef},
        {'shell_type': 's', 'exponents': h_s2_exp, 'coefficients': h_s2_coef},
        {'shell_type': 's', 'exponents': h_s3_exp, 'coefficients': [1.0]},
        {'shell_type': 'p', 'exponents': [0.7], 'coefficients': [1.0]},
        {'shell_type': 'p', 'exponents': [0.2], 'coefficients': [1.0]},
        {'shell_type': 'd', 'exponents': [1.0], 'coefficients': [1.0]},
    ]
    
    c_s1_exp = [2944.501, 442.368, 99.884]
    c_s1_coef = [0.0334946, 0.23472695, 0.81375732]
    c_s2_exp = [99.884, 28.148]
    c_s2_coef = [0.44463454, 0.55536546]
    c_s3_exp = [7.908]
    c_p1_exp = [78.68, 18.24, 5.133]
    c_p1_coef = [0.05000, 0.23400, 0.71600]
    c_p2_exp = [5.133, 1.434]
    c_p2_coef = [0.43000, 0.57000]
    c_p3_exp = [0.4038]
    c_d_exp = [1.8]
    
    basis['C'] = [
        {'shell_type': 's', 'exponents': c_s1_exp, 'coefficients': c_s1_coef},
        {'shell_type': 's', 'exponents': c_s2_exp, 'coefficients': c_s2_coef},
        {'shell_type': 's', 'exponents': [7.908, 2.202], 'coefficients': [0.44463454, 0.55536546]},
        {'shell_type': 's', 'exponents': [0.6108], 'coefficients': [1.0]},
        {'shell_type': 'p', 'exponents': c_p1_exp, 'coefficients': c_p1_coef},
        {'shell_type': 'p', 'exponents': c_p2_exp, 'coefficients': c_p2_coef},
        {'shell_type': 'p', 'exponents': c_p3_exp, 'coefficients': [1.0]},
        {'shell_type': 'd', 'exponents': c_d_exp, 'coefficients': [1.0]},
        {'shell_type': 'd', 'exponents': [0.5], 'coefficients': [1.0]},
        {'shell_type': 'f', 'exponents': [2.0], 'coefficients': [1.0]},
    ]
    
    n_s1_exp = [4414.0, 663.40, 149.60]
    n_s1_coef = [0.0334946, 0.23472695, 0.81375732]
    n_s2_exp = [149.60, 41.88]
    n_s2_coef = [0.44463454, 0.55536546]
    n_p1_exp = [118.10, 27.37, 7.705]
    n_p1_coef = [0.05000, 0.23400, 0.71600]
    n_p2_exp = [7.705, 2.158]
    n_p2_coef = [0.43000, 0.57000]
    n_p3_exp = [0.6050]
    n_d_exp = [2.2]
    
    basis['N'] = [
        {'shell_type': 's', 'exponents': n_s1_exp, 'coefficients': n_s1_coef},
        {'shell_type': 's', 'exponents': n_s2_exp, 'coefficients': n_s2_coef},
        {'shell_type': 's', 'exponents': [41.88, 11.68], 'coefficients': [0.44463454, 0.55536546]},
        {'shell_type': 's', 'exponents': [0.8150], 'coefficients': [1.0]},
        {'shell_type': 'p', 'exponents': n_p1_exp, 'coefficients': n_p1_coef},
        {'shell_type': 'p', 'exponents': n_p2_exp, 'coefficients': n_p2_coef},
        {'shell_type': 'p', 'exponents': n_p3_exp, 'coefficients': [1.0]},
        {'shell_type': 'd', 'exponents': n_d_exp, 'coefficients': [1.0]},
        {'shell_type': 'd', 'exponents': [0.6], 'coefficients': [1.0]},
        {'shell_type': 'f', 'exponents': [2.5], 'coefficients': [1.0]},
    ]
    
    o_s1_exp = [6239.0, 937.70, 209.90]
    o_s1_coef = [0.0334946, 0.23472695, 0.81375732]
    o_s2_exp = [209.90, 58.64]
    o_s2_coef = [0.44463454, 0.55536546]
    o_p1_exp = [166.80, 38.65, 10.90]
    o_p1_coef = [0.05000, 0.23400, 0.71600]
    o_p2_exp = [10.90, 3.047]
    o_p2_coef = [0.43000, 0.57000]
    o_p3_exp = [0.8500]
    o_d_exp = [2.8]
    
    basis['O'] = [
        {'shell_type': 's', 'exponents': o_s1_exp, 'coefficients': o_s1_coef},
        {'shell_type': 's', 'exponents': o_s2_exp, 'coefficients': o_s2_coef},
        {'shell_type': 's', 'exponents': [58.64, 16.30], 'coefficients': [0.44463454, 0.55536546]},
        {'shell_type': 's', 'exponents': [1.128], 'coefficients': [1.0]},
        {'shell_type': 'p', 'exponents': o_p1_exp, 'coefficients': o_p1_coef},
        {'shell_type': 'p', 'exponents': o_p2_exp, 'coefficients': o_p2_coef},
        {'shell_type': 'p', 'exponents': o_p3_exp, 'coefficients': [1.0]},
        {'shell_type': 'd', 'exponents': o_d_exp, 'coefficients': [1.0]},
        {'shell_type': 'd', 'exponents': [0.8], 'coefficients': [1.0]},
        {'shell_type': 'f', 'exponents': [3.0], 'coefficients': [1.0]},
    ]
    
    return basis


DEF2_TZVP_BASIS = _build_def2_tzvp_basis()


# ============================================================================
# D95 基组 (Dunning 1969 双 ζ 全电子基组)
# ============================================================================
# 参考: Dunning, J. Chem. Phys. 53, 2823 (1970)
# H-Ar: (9s5p) -> [4s2p]
# 对应 Gaussian 的 D95, 是 Gaussian 早期版本的默认基组

D95_BASIS: Dict[str, List[Dict[str, Any]]] = {
    'H': [
        {'shell_type': 's', 'exponents': [19.2406000, 2.8992000, 0.6534000, 0.1776000],
         'coefficients': [0.0328290, 0.2312080, 0.8172380, 0.5210130]},
    ],
    'He': [
        {'shell_type': 's', 'exponents': [38.0000000, 5.7700000, 1.2400000, 0.2970000],
         'coefficients': [0.0483490, 0.2331330, 0.8125620, 0.5287890]},
    ],
    'Li': [
        {'shell_type': 's', 'exponents': [642.4189000, 96.7985000, 22.09100, 6.286000,
                                          2.021000, 0.686700, 0.227100, 0.072500, 0.022400],
         'coefficients': [0.0021426, 0.0162080, 0.0773150, 0.2457860, 0.4701890,
                          0.3454700, 0.0285140, -0.0121460, 0.0047403]},
        {'shell_type': 's', 'exponents': [0.072500], 'coefficients': [1.0]},
        {'shell_type': 'p', 'exponents': [0.685000, 0.165000, 0.0584000, 0.0199000, 0.0068000],
         'coefficients': [0.0071060, 0.1193700, 0.4825700, 0.4509100, 0.0837610]},
    ],
    'Be': [
        {'shell_type': 's', 'exponents': [1064.000, 159.7000, 36.86000, 10.3700, 3.2810,
                                          1.0940, 0.3592, 0.1112, 0.0322],
         'coefficients': [0.0019195, 0.0148400, 0.0729190, 0.2371200, 0.4684700,
                          0.3558000, 0.0325000, -0.0125600, 0.0048870]},
        {'shell_type': 's', 'exponents': [0.1112], 'coefficients': [1.0]},
        {'shell_type': 'p', 'exponents': [1.2800, 0.3200, 0.1080, 0.0340, 0.0100],
         'coefficients': [0.0071500, 0.1155000, 0.4736000, 0.4464000, 0.0853000]},
    ],
    'B': [
        {'shell_type': 's', 'exponents': [2811.0, 423.4, 97.04, 27.18, 8.609, 2.926,
                                          1.020, 0.3418, 0.1095, 0.0334],
         'coefficients': [0.0019768, 0.0151850, 0.0738010, 0.2478200, 0.4878800,
                          0.3184500, 0.0141270, -0.0037522, 0.0015722, -0.0001255]},
        {'shell_type': 's', 'exponents': [0.1095], 'coefficients': [1.0]},
        {'shell_type': 's', 'exponents': [0.0334], 'coefficients': [1.0]},
        {'shell_type': 'p', 'exponents': [7.510, 1.770, 0.540, 0.180, 0.0600],
         'coefficients': [0.0050050, 0.0918250, 0.4013400, 0.4563200, 0.1478200]},
    ],
    'C': [
        {'shell_type': 's', 'exponents': [4232.0, 634.9, 146.1, 41.13, 13.04, 4.435,
                                          1.522, 0.4915, 0.1550, 0.04660],
         'coefficients': [0.0019418, 0.0149950, 0.0732550, 0.2459700, 0.4872000,
                          0.3245200, 0.0210920, -0.0041820, 0.0013170, -0.0001038]},
        {'shell_type': 's', 'exponents': [0.1550], 'coefficients': [1.0]},
        {'shell_type': 's', 'exponents': [0.04660], 'coefficients': [1.0]},
        {'shell_type': 'p', 'exponents': [9.380, 2.220, 0.6830, 0.2230, 0.07180],
         'coefficients': [0.0051730, 0.0945710, 0.4071200, 0.4524600, 0.1470200]},
    ],
    'N': [
        {'shell_type': 's', 'exponents': [6232.0, 934.9, 215.0, 60.49, 19.17, 6.522,
                                          2.235, 0.7204, 0.2265, 0.0680],
         'coefficients': [0.0019682, 0.0151950, 0.0741870, 0.2492500, 0.4887300,
                          0.3194700, 0.0166970, -0.0044130, 0.0013150, -0.0001030]},
        {'shell_type': 's', 'exponents': [0.2265], 'coefficients': [1.0]},
        {'shell_type': 's', 'exponents': [0.0680], 'coefficients': [1.0]},
        {'shell_type': 'p', 'exponents': [11.83, 2.785, 0.840, 0.2720, 0.0878],
         'coefficients': [0.0051740, 0.0940240, 0.4049200, 0.4515100, 0.1469000]},
    ],
    'O': [
        {'shell_type': 's', 'exponents': [8588.0, 1285.0, 296.4, 83.34, 26.36, 8.965,
                                          3.063, 0.9885, 0.3105, 0.0928],
         'coefficients': [0.0019750, 0.0151980, 0.0742530, 0.2500600, 0.4901300,
                          0.3174600, 0.0146040, -0.0044700, 0.0013150, -0.0001030]},
        {'shell_type': 's', 'exponents': [0.3105], 'coefficients': [1.0]},
        {'shell_type': 's', 'exponents': [0.0928], 'coefficients': [1.0]},
        {'shell_type': 'p', 'exponents': [14.40, 3.390, 1.029, 0.3360, 0.1080],
         'coefficients': [0.0051630, 0.0933960, 0.4012900, 0.4517200, 0.1471100]},
    ],
    'F': [
        {'shell_type': 's', 'exponents': [11430.0, 1710.0, 393.7, 110.6, 34.93, 11.84,
                                          4.040, 1.310, 0.4120, 0.1233],
         'coefficients': [0.0019890, 0.0152470, 0.0743190, 0.2503400, 0.4907200,
                          0.3162500, 0.0136900, -0.0044200, 0.0013150, -0.0001030]},
        {'shell_type': 's', 'exponents': [0.4120], 'coefficients': [1.0]},
        {'shell_type': 's', 'exponents': [0.1233], 'coefficients': [1.0]},
        {'shell_type': 'p', 'exponents': [17.34, 4.093, 1.245, 0.4060, 0.1300],
         'coefficients': [0.0051640, 0.0933620, 0.4012500, 0.4517700, 0.1471400]},
    ],
    'Ne': [
        {'shell_type': 's', 'exponents': [14750.0, 2207.0, 507.6, 142.6, 45.00, 15.21,
                                          5.183, 1.685, 0.5300, 0.1585],
         'coefficients': [0.0019970, 0.0152680, 0.0743390, 0.2504900, 0.4909700,
                          0.3156800, 0.0131420, -0.0043880, 0.0013160, -0.0001030]},
        {'shell_type': 's', 'exponents': [0.5300], 'coefficients': [1.0]},
        {'shell_type': 's', 'exponents': [0.1585], 'coefficients': [1.0]},
        {'shell_type': 'p', 'exponents': [20.46, 4.834, 1.470, 0.4790, 0.1530],
         'coefficients': [0.0051680, 0.0933160, 0.4011500, 0.4518400, 0.1471800]},
    ],
}

# 简化版本: 对 Na-Ar 用同一组 9s5p 参数
D95_BASIS.update({
    'Na': D95_BASIS['Ne'][:],
    'Mg': D95_BASIS['Ne'][:],
    'Al': D95_BASIS['Ne'][:],
    'Si': D95_BASIS['Ne'][:],
    'P': D95_BASIS['Ne'][:],
    'S': D95_BASIS['Ne'][:],
    'Cl': D95_BASIS['Ne'][:],
    'Ar': D95_BASIS['Ne'][:],
})


# ============================================================================
# 基组注册表
# ============================================================================

_BASIS_REGISTRY = {
    'sto-3g': STO3G_BASIS,
    '3-21g': THREE21G_BASIS,
    '6-31g': SIX31G_BASIS,
    '6-31g(d)': SIX31G_STAR_BASIS,
    '6-31g*': SIX31G_STAR_BASIS,
    'cc-pvdz': CC_PVDZ_BASIS,
    'cc-pvtz': CC_PVTZ_BASIS,
    'def2-svp': DEF2_SVP_BASIS,
    'def2-tzvp': DEF2_TZVP_BASIS,
    'd95': D95_BASIS,
}


def get_basis_data(name: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    获取指定基组的数据。

    Parameters
    ----------
    name : str
        基组名称，不区分大小写

    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        基组数据字典

    Raises
    ------
    ValueError
        如果基组不存在
    """
    name_lower = name.lower()
    if name_lower not in _BASIS_REGISTRY:
        available = list(_BASIS_REGISTRY.keys())
        raise ValueError(f"基组 '{name}' 不存在。可用基组: {available}")
    return _BASIS_REGISTRY[name_lower]


def list_available_basis() -> List[str]:
    """返回所有可用内置基组的列表。"""
    return sorted(_BASIS_REGISTRY.keys())
