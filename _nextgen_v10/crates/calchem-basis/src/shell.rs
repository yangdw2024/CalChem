#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AngularType {
    Spherical,
    Cartesian,
}

#[derive(Debug, Clone)]
pub struct Shell {
    pub atom_idx: u32,
    pub angular_l: u8,
    pub exponents: Vec<f64>,
    pub coefficients: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct BasisSetData {
    pub name: String,
    pub angular_type: AngularType,
    pub shells: Vec<Shell>,
    pub nbf: usize,
}
