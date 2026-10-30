//! 壳层（Shell）与基组数据结构。
//!
//! 归一化约定（与 Python 参照 `calchem/basis/gto.py` 一致）：
//!   原语 = N(α,l,m,n) · x^l y^m z^n · exp(-α·r²)
//!   收缩 = Σ_p c_p · [N(α_p,·) · x^l y^m z^n · exp(-α_p·r²)]
//! 即存储的收缩系数 c_p 是“归一化原语”的系数，原语归一化 N 在积分代码中另行施加。

/// 角动量类型：球谐（5d,7f,9g）或笛卡尔（6d,10f,15g）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AngularType {
    /// 球谐函数（默认，消除冗余分量）。
    Spherical,
    /// 笛卡尔函数（用于历史基准比对）。
    Cartesian,
}

impl AngularType {
    /// 解析字符串："spherical"/"sph"/"sp" => Spherical；"cartesian"/"cart" => Cartesian。
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "cartesian" | "cart" => AngularType::Cartesian,
            _ => AngularType::Spherical,
        }
    }
}

/// 一个收缩壳层。
#[derive(Debug, Clone)]
pub struct Shell {
    /// 所属原子在分子中的索引。
    pub atom_idx: u32,
    /// 角动量量子数 l（0=s,1=p,2=d,...）。
    pub angular_l: u8,
    /// 壳层中心（Bohr，通常等于原子坐标）。
    pub center_bohr: [f64; 3],
    /// 原语指数（长度 = 原语数）。
    pub exponents: Vec<f64>,
    /// 归一化原语的收缩系数（与 exponents 等长）。
    pub coefficients: Vec<f64>,
    /// 在全局 AO 矩阵中的起始索引（按所选角动量类型排布）。
    pub ao_offset: u32,
}

impl Shell {
    /// 原语数目。
    pub fn n_prim(&self) -> usize {
        self.exponents.len()
    }

    /// 笛卡尔分量数 (l+1)(l+2)/2。
    pub fn n_cartesian(&self) -> usize {
        let l = self.angular_l as usize;
        (l + 1) * (l + 2) / 2
    }

    /// 球谐分量数 2l+1。
    pub fn n_spherical(&self) -> usize {
        2 * self.angular_l as usize + 1
    }

    /// 按角动量类型返回该壳层的基函数数。
    pub fn n_functions(&self, at: AngularType) -> usize {
        match at {
            AngularType::Spherical => self.n_spherical(),
            AngularType::Cartesian => self.n_cartesian(),
        }
    }
}

/// 完整基组数据。
#[derive(Debug, Clone)]
pub struct BasisSetData {
    /// 基组名称（如 "sto-3g"）。
    pub name: String,
    /// 角动量类型。
    pub angular_type: AngularType,
    /// 全部壳层（按 AO 排布顺序）。
    pub shells: Vec<Shell>,
    /// 总基函数数。
    pub nbf: usize,
}

impl BasisSetData {
    /// 第 i 个壳层的 AO 索引区间 [start, end)。
    pub fn shell_ao_range(&self, i: usize) -> (usize, usize) {
        let start = self.shells[i].ao_offset as usize;
        let end = start + self.shells[i].n_functions(self.angular_type);
        (start, end)
    }

    /// 壳层中心。
    pub fn shell_center(&self, i: usize) -> [f64; 3] {
        self.shells[i].center_bohr
    }
}

/// 笛卡尔角指标 (l,m,n) 列表，顺序与 Python 参照 `cartesian_angular_indices` 一致。
pub fn cartesian_angular_indices(l: usize) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    for li in (0..=l).rev() {
        for mi in (0..=(l - li)).rev() {
            let ni = l - li - mi;
            out.push((li, mi, ni));
        }
    }
    out
}
