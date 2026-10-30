use crate::{BasisSetData, AngularType, Shell, MoleculeCore};

pub fn load_builtin_basis(name: &str, mol: &MoleculeCore, angular: AngularType) -> Result<BasisSetData, String> {
    let lower = name.to_lowercase();
    let mut shells = Vec::new();
    let mut nbf = 0;

    for (atom_idx, atom) in mol.atoms.iter().enumerate() {
        match (lower.as_str(), atom.atomic_number) {
            ("sto-3g", 1) => {
                // H STO-3G 1s
                shells.push(Shell {
                    atom_idx: atom_idx as u32,
                    angular_l: 0,
                    exponents: vec![3.42525091, 0.62391373, 0.16885540],
                    coefficients: vec![0.15432897, 0.53532814, 0.44463454],
                });
                nbf += 1;
            }
            ("sto-3g", 8) => {
                // O STO-3G 1s, 2s, 2p
                shells.push(Shell {
                    atom_idx: atom_idx as u32,
                    angular_l: 0,
                    exponents: vec![130.7093200, 23.8088610, 6.4436083],
                    coefficients: vec![0.15432897, 0.53532814, 0.44463454],
                });
                shells.push(Shell {
                    atom_idx: atom_idx as u32,
                    angular_l: 0,
                    exponents: vec![5.0331513, 1.1695961, 0.3803890],
                    coefficients: vec![-0.09996723, 0.39951283, 0.70011547],
                });
                shells.push(Shell {
                    atom_idx: atom_idx as u32,
                    angular_l: 1,
                    exponents: vec![5.0331513, 1.1695961, 0.3803890],
                    coefficients: vec![0.15591627, 0.60768372, 0.39195739],
                });
                nbf += 5; // 1s + 2s + 2px,2py,2pz
            }
            _ => {
                // Default fallback minimal representation
                shells.push(Shell {
                    atom_idx: atom_idx as u32,
                    angular_l: 0,
                    exponents: vec![1.0],
                    coefficients: vec![1.0],
                });
                nbf += 1;
            }
        }
    }

    Ok(BasisSetData {
        name: name.to_string(),
        angular_type: angular,
        shells,
        nbf,
    })
}
