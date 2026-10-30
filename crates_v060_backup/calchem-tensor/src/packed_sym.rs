//! 上三角紧凑对称矩阵（仅存储 N*(N+1)/2 个元素）。
//!
//! 存储约定：对逻辑对称矩阵 A，物理上只保存列主序的上三角，
//! 即 `data[col*(col+1)/2 + row]`（row <= col）。
//! 逻辑元素 A[i,j] 与 A[j,i] 映射到同一位置。

#[derive(Debug, Clone)]
pub struct PackedSymMatrix {
    pub dim: usize,
    pub data: Vec<f64>,
}

impl PackedSymMatrix {
    /// 全零矩阵。
    pub fn zeros(dim: usize) -> Self {
        let len = dim * (dim + 1) / 2;
        Self {
            dim,
            data: vec![0.0; len],
        }
    }

    /// 由已有的紧凑数据构造（调用方保证长度正确）。
    pub fn from_packed(dim: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), dim * (dim + 1) / 2);
        Self { dim, data }
    }

    /// 紧凑索引：返回逻辑 (i,j) 对应的物理位置（自动归一化到上三角）。
    #[inline(always)]
    pub fn idx(i: usize, j: usize) -> usize {
        let (r, c) = if i <= j { (i, j) } else { (j, i) };
        c * (c + 1) / 2 + r
    }

    #[inline(always)]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[Self::idx(i, j)]
    }

    #[inline(always)]
    pub fn set(&mut self, i: usize, j: usize, val: f64) {
        let idx = Self::idx(i, j);
        self.data[idx] = val;
    }

    #[inline(always)]
    pub fn add_assign(&mut self, i: usize, j: usize, val: f64) {
        let idx = Self::idx(i, j);
        self.data[idx] += val;
    }

    /// 原地缩放：self <- alpha * self。
    pub fn scale(&mut self, alpha: f64) {
        for x in self.data.iter_mut() {
            *x *= alpha;
        }
    }

    /// 自加：self <- self + alpha * other。
    pub fn axpy(&mut self, alpha: f64, other: &Self) {
        assert_eq!(self.dim, other.dim);
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += alpha * b;
        }
    }

    /// 取对角元素。
    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.dim).map(|i| self.get(i, i)).collect()
    }

    /// 展开为行主序稠密矩阵（长度 N*N）。
    pub fn to_dense(&self) -> Vec<f64> {
        let n = self.dim;
        let mut dense = vec![0.0; n * n];
        for i in 0..n {
            for j in i..n {
                let v = self.get(i, j);
                dense[i * n + j] = v;
                dense[j * n + i] = v;
            }
        }
        dense
    }

    /// 由行主序稠密对称矩阵压缩为紧凑矩阵（非对称输入只取对称化结果）。
    pub fn from_dense(dim: usize, dense: &[f64]) -> Self {
        assert_eq!(dense.len(), dim * dim);
        let mut out = Self::zeros(dim);
        for i in 0..dim {
            for j in i..dim {
                let v = 0.5 * (dense[i * dim + j] + dense[j * dim + i]);
                out.set(i, j, v);
            }
        }
        out
    }

    /// Frobenius 内积 <A, B>_F。
    pub fn dot(&self, other: &Self) -> f64 {
        assert_eq!(self.dim, other.dim);
        self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_roundtrip() {
        let n = 4;
        let dense = vec![
            1.0, 2.0, 3.0, 4.0,
            2.0, 5.0, 6.0, 7.0,
            3.0, 6.0, 8.0, 9.0,
            4.0, 7.0, 9.0, 10.0,
        ];
        let p = PackedSymMatrix::from_dense(n, &dense);
        let back = p.to_dense();
        for i in 0..n * n {
            assert!((back[i] - dense[i]).abs() < 1e-12);
        }
        assert_eq!(p.data.len(), n * (n + 1) / 2);
    }
}
