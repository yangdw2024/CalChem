//! 多中心积分层：单电子积分、双电子 ERI、Schwarz 筛选、Direct/In-Core JK、RI-DF。
//!
//! MVP 采用纯原生 Rust 实现（Obara-Saika + HRR），不依赖外部 C 库。

pub mod direct_jk;
pub mod eri;
pub mod obara_saika_fast;
pub mod one_electron;
pub mod ri_df;
pub mod screening;

pub use direct_jk::*;
pub use eri::*;
pub use obara_saika_fast::*;
pub use one_electron::*;
pub use ri_df::*;
pub use screening::*;

/// 单电子积分算子类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OneElectronOp {
    /// 重叠 S。
    Overlap,
    /// 动能 T。
    Kinetic,
    /// 核吸引 V。
    NuclearAttraction,
    /// 电偶极（0=x,1=y,2=z）。
    Dipole(usize),
}
