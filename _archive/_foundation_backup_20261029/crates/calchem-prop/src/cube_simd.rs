pub fn eval_mo_on_regular_grid(
    _orbital_idx: usize,
    _origin: [f64; 3],
    _spacing: [f64; 3],
    dims: [usize; 3],
) -> Vec<f32> {
    let total = dims[0] * dims[1] * dims[2];
    vec![0.0f32; total]
}
