"""Public-boundary regression tests for scientifically unsupported requests."""
import numpy as np
import pytest
from calchem import _core
from calchem.dft.d3 import compute_d3_energy,compute_d3_gradient

def test_irc_binding_preserves_acceptance_and_endpoint_fields():
    for field in ('barrier_forward_kcal_mol','barrier_reverse_kcal_mol','reaction_energy_kcal_mol'):
        assert hasattr(_core.IrcResult,field)
    assert hasattr(_core.IrcBranch,'converged_to_minimum')

@pytest.mark.parametrize("method", ["lc-wpbe", "hse06", "m06-2x"])
def test_unsupported_range_separation_is_not_substituted(method):
    with pytest.raises(ValueError, match="Unsupported DFT XC functional"):
        _core.run_single_point(["He"], [[0., 0., 0.]], method=method)

@pytest.mark.parametrize("method", ["CAM-B3LYP", "wb97x-d", "wb97x"])
def test_supported_range_separation_runs_natively(method):
    res = _core.run_single_point(["He"], [[0., 0., 0.]], method=method, basis="sto-3g")
    assert res.converged
    assert -3.0 < res.total_energy_hartree < -2.5

@pytest.mark.parametrize("fn",[compute_d3_energy,compute_d3_gradient])
def test_unknown_dispersion_functional_and_custom_parameters_fail(fn):
    xyz=np.array([[0.,0.,0.],[0.,0.,2.]])
    with pytest.raises(ValueError):fn(["H","H"],xyz,functional="unknown")
    with pytest.raises(ValueError):fn(["H","H"],xyz,s8=1.)

def test_unimplemented_open_shell_scrf_is_not_reported_as_converged():
    with pytest.raises(ValueError,match="open-shell PCM"):
        _core.run_solvation(["O","H"],[[0.,0.,0.],[0.,0.,1.8]],"sto-3g",method="uhf",multiplicity=2)

def test_ordinary_minimum_cannot_launch_an_irc():
    result=_core.run_optimize(["H","H"],[[0.,0.,0.],[0.,0.,1.4]],basis="sto-3g",method="rhf")
    coords=np.asarray(result.coords).reshape(2,3).tolist()
    with pytest.raises(ValueError,match="imaginary vibrational mode"):
        _core.run_irc(["H","H"],coords,transition_vector=[0.,0.,-1.,0.,0.,1.],basis="sto-3g",method="rhf",max_points=1)
