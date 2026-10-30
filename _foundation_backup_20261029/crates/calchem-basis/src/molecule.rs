#[derive(Debug, Clone)]
pub struct Atom {
    pub atomic_number: u8,
    pub symbol: String,
    pub coord_bohr: [f64; 3],
}

#[derive(Debug, Clone)]
pub struct MoleculeCore {
    pub atoms: Vec<Atom>,
    pub charge: i32,
    pub multiplicity: usize,
}

impl MoleculeCore {
    pub fn new(atomic_numbers: &[u8], coords_bohr: &[f64], charge: i32, multiplicity: usize) -> Self {
        let n = atomic_numbers.len();
        let mut atoms = Vec::with_capacity(n);
        for i in 0..n {
            let z = atomic_numbers[i];
            atoms.push(Atom {
                atomic_number: z,
                symbol: element_symbol(z).to_string(),
                coord_bohr: [coords_bohr[i * 3], coords_bohr[i * 3 + 1], coords_bohr[i * 3 + 2]],
            });
        }
        Self { atoms, charge, multiplicity }
    }

    pub fn nuclear_repulsion_energy(&self) -> f64 {
        let mut e = 0.0;
        let n = self.atoms.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = self.atoms[i].coord_bohr[0] - self.atoms[j].coord_bohr[0];
                let dy = self.atoms[i].coord_bohr[1] - self.atoms[j].coord_bohr[1];
                let dz = self.atoms[i].coord_bohr[2] - self.atoms[j].coord_bohr[2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                e += (self.atoms[i].atomic_number as f64) * (self.atoms[j].atomic_number as f64) / r;
            }
        }
        e
    }
}

pub fn element_symbol(z: u8) -> &'static str {
    match z {
        1 => "H", 2 => "He", 6 => "C", 7 => "N", 8 => "O", 9 => "F", 10 => "Ne",
        11 => "Na", 12 => "Mg", 14 => "Si", 15 => "P", 16 => "S", 17 => "Cl", 18 => "Ar",
        _ => "X",
    }
}
