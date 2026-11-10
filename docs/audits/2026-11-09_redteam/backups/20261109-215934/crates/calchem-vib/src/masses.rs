//! Standard isotopic and atomic masses (in atomic mass units / amu) for $Z = 1 \dots 86$.
//!
//! Uses most abundant isotope standard masses (IUPAC / NIST) matching standard
//! computational quantum chemistry codes (Gaussian, ORCA, PySCF).

pub const AMU_TO_ELECTRON_MASS: f64 = 1822.888486209;
pub const HARTREE_TO_EV: f64 = 27.211386245988;
pub const HARTREE_TO_KCAL_MOL: f64 = 627.5094740631;
pub const HARTREE_TO_KJ_MOL: f64 = 2625.4996394799;
pub const JOULE_TO_CALORIE: f64 = 1.0 / 4.184;

// Physical conversion constant from sqrt(Hartree / (Bohr^2 * amu)) to wavenumber cm^-1:
// 1 Hartree = 4.3597447222071e-18 J
// 1 Bohr = 5.29177210903e-11 m
// 1 amu = 1.66053906660e-27 kg
// c = 2.99792458e10 cm/s
// sqrt(Hartree / (Bohr^2 * amu)) / (2 * pi * c) = 5140.48714376 cm^-1
pub const HESSIAN_EVAL_TO_INV_CM: f64 = 5140.48714376;

// Prefactor for IR intensity: km/mol from |dmu/dQ|^2 with mu in e*a0, Q in amu^{1/2}*a0:
// I_k = (N_A * pi) / (3 * c^2) * |dmu/dQ|^2 = 42.2561 * |dmu/dQ|^2 (km/mol)
pub const IR_INTENSITY_PREFACTOR_KM_MOL: f64 = 42.2561;

// Ideal gas constant R = 8.314462618 J/(mol*K) = 1.9872042586 cal/(mol*K)
pub const GAS_CONSTANT_J_MOL_K: f64 = 8.314462618;
pub const GAS_CONSTANT_CAL_MOL_K: f64 = 1.9872042586;
pub const GAS_CONSTANT_HARTREE_K: f64 = 3.166811563e-6; // k_B in Hartree / K
pub const SPEED_OF_LIGHT_CM_S: f64 = 2.99792458e10;
pub const PLANCK_CONSTANT_J_S: f64 = 6.62607015e-34;
pub const BOLTZMANN_CONSTANT_J_K: f64 = 1.380649e-23;
pub const AVOGADRO_CONSTANT: f64 = 6.02214076e23;
pub const STANDARD_PRESSURE_PA: f64 = 101325.0; // 1 atm
pub const STANDARD_TEMPERATURE_K: f64 = 298.15;

/// Most abundant isotope atomic masses in amu for elements $Z = 1 \dots 86$.
const MOST_ABUNDANT_ISOTOPE_MASSES: [f64; 86] = [
    1.00782503223,  // 1  H  (^1H)
    4.00260325413,  // 2  He (^4He)
    7.0160034366,   // 3  Li (^7Li)
    9.012183065,    // 4  Be (^9Be)
    11.00930536,    // 5  B  (^11B)
    12.000000000,   // 6  C  (^12C)
    14.00307400443, // 7  N  (^14N)
    15.99491461957, // 8  O  (^16O)
    18.99840316273, // 9  F  (^19F)
    19.9924401762,  // 10 Ne (^20Ne)
    22.9897692820,  // 11 Na (^23Na)
    23.985041697,   // 12 Mg (^24Mg)
    26.98153853,    // 13 Al (^27Al)
    27.97692653465, // 14 Si (^28Si)
    30.97376199842, // 15 P  (^31P)
    31.9720711744,  // 16 S  (^32S)
    34.968852682,   // 17 Cl (^35Cl)
    39.9623831237,  // 18 Ar (^40Ar)
    38.9637064864,  // 19 K  (^39K)
    39.962590863,   // 20 Ca (^40Ca)
    44.95590828,    // 21 Sc (^45Sc)
    47.94794198,    // 22 Ti (^48Ti)
    50.9439570,     // 23 V  (^51V)
    51.94050623,    // 24 Cr (^52Cr)
    54.93804393,    // 25 Mn (^55Mn)
    55.93493633,    // 26 Fe (^56Fe)
    58.93319429,    // 27 Co (^59Co)
    57.93534166,    // 28 Ni (^58Ni)
    62.92959772,    // 29 Cu (^63Cu)
    63.92914201,    // 30 Zn (^64Zn)
    68.9255735,     // 31 Ga (^69Ga)
    73.9211774,     // 32 Ge (^74Ge)
    74.92159457,    // 33 As (^75As)
    79.9165218,     // 34 Se (^80Se)
    78.9183376,     // 35 Br (^79Br)
    83.9114977282,  // 36 Kr (^84Kr)
    84.9117897379,  // 37 Rb (^85Rb)
    87.9056125,     // 38 Sr (^88Sr)
    88.9058479,     // 39 Y  (^89Y)
    89.9046977,     // 40 Zr (^90Zr)
    92.9063772,     // 41 Nb (^93Nb)
    97.90540482,    // 42 Mo (^98Mo)
    98.9062547,     // 43 Tc (^99Tc)
    101.9043441,    // 44 Ru (^102Ru)
    102.9054980,    // 45 Rh (^103Rh)
    105.9034804,    // 46 Pd (^106Pd)
    106.9050916,    // 47 Ag (^107Ag)
    113.90336509,   // 48 Cd (^114Cd)
    114.903878776,  // 49 In (^115In)
    119.9021991,    // 50 Sn (^120Sn)
    120.9038157,    // 51 Sb (^121Sb)
    129.90622274,   // 52 Te (^130Te)
    126.9044719,    // 53 I  (^127I)
    131.9041550856, // 54 Xe (^132Xe)
    132.905451961,  // 55 Cs (^133Cs)
    137.9052470,    // 56 Ba (^138Ba)
    138.9063563,    // 57 La (^139La)
    139.9054409,    // 58 Ce (^140Ce)
    140.9076576,    // 59 Pr (^141Pr)
    141.9077290,    // 60 Nd (^142Nd)
    144.9127559,    // 61 Pm (^145Pm)
    151.9197397,    // 62 Sm (^152Sm)
    152.9212380,    // 63 Eu (^153Eu)
    157.9241123,    // 64 Gd (^158Gd)
    158.9253547,    // 65 Tb (^159Tb)
    163.9291819,    // 66 Dy (^164Dy)
    164.9303288,    // 67 Ho (^165Ho)
    165.9302995,    // 68 Er (^166Er)
    168.9342179,    // 69 Tm (^169Tm)
    173.9388664,    // 70 Yb (^174Yb)
    174.9407752,    // 71 Lu (^175Lu)
    179.9465570,    // 72 Hf (^180Hf)
    180.9479958,    // 73 Ta (^181Ta)
    183.95093092,   // 74 W  (^184W)
    186.9557501,    // 75 Re (^187Re)
    191.9614770,    // 76 Os (^192Os)
    192.9629216,    // 77 Ir (^193Ir)
    194.9647917,    // 78 Pt (^195Pt)
    196.96656879,   // 79 Au (^197Au)
    201.97064340,   // 80 Hg (^202Hg)
    204.9744278,    // 81 Tl (^205Tl)
    207.9766525,    // 82 Pb (^208Pb)
    208.9803991,    // 83 Bi (^209Bi)
    208.9824308,    // 84 Po (^209Po)
    209.9871479,    // 85 At (^210At)
    222.0175782,    // 86 Rn (^222Rn)
];

/// Returns the most abundant isotope mass in atomic mass units (amu).
pub fn atomic_mass_amu(z: u8) -> f64 {
    if z >= 1 && (z as usize) <= MOST_ABUNDANT_ISOTOPE_MASSES.len() {
        MOST_ABUNDANT_ISOTOPE_MASSES[(z - 1) as usize]
    } else {
        (z as f64) * 2.0 // Rough fallback for superheavies
    }
}
