#[derive(Debug, Clone)]
pub struct SchwarzScreener {
    pub threshold: f64,
    pub max_dm_global: f64,
    pub significant_pairs: Vec<(u32, u32, f64)>,
}

impl SchwarzScreener {
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            max_dm_global: 1.0,
            significant_pairs: Vec::new(),
        }
    }
}
