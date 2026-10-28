__version__ = "0.1.0"

BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

HARTREE_TO_EV = 27.211386245988
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV

HARTREE_TO_KCAL_MOL = 627.5094740631
KCAL_MOL_TO_HARTREE = 1.0 / HARTREE_TO_KCAL_MOL

BOLTZMANN_CONSTANT = 1.380649e-23
AVOGADRO_NUMBER = 6.02214076e23
SPEED_OF_LIGHT = 299792458.0
PLANCK_CONSTANT = 6.62607015e-34
ELECTRON_MASS = 9.1093837015e-31
PROTON_MASS = 1.67262192369e-27

ELEMENTS = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
    37: "Rb",
    38: "Sr",
    39: "Y",
    40: "Zr",
    41: "Nb",
    42: "Mo",
    43: "Tc",
    44: "Ru",
    45: "Rh",
    46: "Pd",
    47: "Ag",
    48: "Cd",
    49: "In",
    50: "Sn",
    51: "Sb",
    52: "Te",
    53: "I",
    54: "Xe",
    55: "Cs",
    56: "Ba",
    57: "La",
    58: "Ce",
    59: "Pr",
    60: "Nd",
    61: "Pm",
    62: "Sm",
    63: "Eu",
    64: "Gd",
    65: "Tb",
    66: "Dy",
    67: "Ho",
    68: "Er",
    69: "Tm",
    70: "Yb",
    71: "Lu",
    72: "Hf",
    73: "Ta",
    74: "W",
    75: "Re",
    76: "Os",
    77: "Ir",
    78: "Pt",
    79: "Au",
    80: "Hg",
    81: "Tl",
    82: "Pb",
    83: "Bi",
    84: "Po",
    85: "At",
    86: "Rn",
}

ATOMIC_NUMBERS = {v: k for k, v in ELEMENTS.items()}

ATOMIC_MASSES = {
    "H": 1.00782503223,
    "He": 4.00260325413,
    "Li": 7.0160034366,
    "Be": 9.012183065,
    "B": 10.01293695,
    "C": 12.0,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "F": 18.99840316273,
    "Ne": 19.99244017542,
    "Na": 22.989769282,
    "Mg": 23.985041697,
    "Al": 26.98153853,
    "Si": 27.976926535,
    "P": 30.97376199842,
    "S": 31.9720711744,
    "Cl": 34.968852682,
    "Ar": 39.948,
    "K": 38.963706486,
    "Ca": 39.962590863,
}

KB = BOLTZMANN_CONSTANT
HARTREE_TO_JOULE = 4.3597447222071e-18
R = 8.31446261815324
