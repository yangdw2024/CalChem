use pyo3::prelude::*;
use calchem_model::{Atom, Molecule};
use calchem_basis::BasisSet;
use calchem_scf::rhf;

#[pyclass]
pub struct PyScfWavefunction {
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub iterations: usize,
    #[pyo3(get)]
    pub total_energy_hartree: f64,
}

// 嵌入基组数据
const STO3G_JSON: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../calchem-basis/data/sto-3g.json"));

#[pyfunction]
#[pyo3(signature = (atomic_numbers, coords_bohr, charge=0, multiplicity=1, method="rhf", basis="sto-3g"))]
pub fn run_single_point(
    _py: Python<'_>,
    atomic_numbers: Vec<u8>,
    coords_bohr: Vec<f64>,
    charge: i32,
    multiplicity: usize,
    method: &str,
    basis: &str,
) -> PyResult<PyScfWavefunction> {
    // 解析坐标
    if coords_bohr.len() % 3 != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("坐标数组长度必须是3的倍数"));
    }
    let natoms = atomic_numbers.len();
    if coords_bohr.len() / 3 != natoms {
        return Err(pyo3::exceptions::PyValueError::new_err("原子数与坐标数不匹配"));
    }

    let atoms: Vec<Atom> = atomic_numbers
        .iter()
        .zip(coords_bohr.chunks(3))
        .map(|(&z, c)| Atom {
            z,
            position_bohr: [c[0], c[1], c[2]],
        })
        .collect();

    let mol = Molecule::try_new(atoms, charge, multiplicity as u32)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    if method.to_lowercase() != "rhf" {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            format!("目前仅支持 RHF，收到 method={method}")
        ));
    }

    // 加载基组
    let atom_list: Vec<(u8, [f64; 3])> = mol.atoms.iter().map(|a| (a.z, a.position_bohr)).collect();
    let bs = BasisSet::build(basis, STO3G_JSON, &atom_list)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    // 执行 RHF
    let result = rhf(&mol, &bs)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(PyScfWavefunction {
        converged: result.converged,
        iterations: result.n_iter as usize,
        total_energy_hartree: result.energy,
    })
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScfWavefunction>()?;
    m.add_function(wrap_pyfunction!(run_single_point, m)?)?;
    Ok(())
}
