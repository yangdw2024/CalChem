"""NativeCalchem: 把现有原生引擎包装成统一 Calculator 接口。

不重写任何 SCF/积分/DFT 算法——直接实例化计算类并提取结果。
"""

from __future__ import annotations

import time
from datetime import datetime

from ..core.calculator import Calculator, register_backend
from ..core.capabilities import BackendCapabilities
from ..core.errors import ConvergenceError
from ..core.request import CalculationRequest, MoleculeSpec
from ..core.result import OrbitalInfo, Result
from ..mol.molecule import Molecule


_NATIVE_METHODS = [
    "rhf", "hf", "uhf", "rohf",
    "rks", "uks", "dft",
    "mp2", "mp4", "ccsd", "ccsd(t)", "ccsdt", "ccd", "casscf",
]

_NATIVE_BASIS = [
    "sto-3g", "3-21g", "6-31g", "6-31g(d)",
    "cc-pvdz", "cc-pvtz", "def2-svp", "def2-tzvp", "d95",
]


def _spec_to_molecule(spec: MoleculeSpec) -> Molecule:
    """MoleculeSpec（Å）→ Molecule 对象。"""
    return Molecule(
        symbols=spec.symbols,
        coordinates=spec.coordinates_angstrom,
        charge=spec.charge,
        multiplicity=spec.multiplicity,
        units="angstrom",
    )


def _build_calculator(mol: Molecule, method: str, basis: str, xc: str | None, options: dict):
    """根据 method 实例化对应的计算类（与 high_level.single_point 逻辑一致）。"""
    m = method.lower()
    kwargs = dict(options)

    if m in ("hf", "rhf"):
        from ..scf.hf import RHF
        return RHF(mol, basis=basis, **kwargs)
    elif m == "uhf":
        from ..scf.hf import UHF
        return UHF(mol, basis=basis, **kwargs)
    elif m == "rohf":
        from ..scf.hf import ROHF
        return ROHF(mol, basis=basis, **kwargs)
    elif m in ("dft", "rks"):
        from ..scf.dft import RKS
        return RKS(mol, basis=basis, xc=xc or "b3lyp", **kwargs)
    elif m == "uks":
        from ..scf.dft import UKS
        return UKS(mol, basis=basis, xc=xc or "b3lyp", **kwargs)
    elif m == "mp2":
        from ..post_hf.mp2 import MP2
        return MP2(mol, basis=basis, **kwargs)
    elif m == "ccsd":
        from ..post_hf.ccsd import CCSD
        return CCSD(mol, basis=basis, **kwargs)
    elif m in ("ccsd(t)", "ccsdt"):
        from ..post_hf.ccsd_t import CCSDT
        return CCSDT(mol, basis=basis, **kwargs)
    elif m == "mp4":
        from ..post_hf.mp4 import MP4
        return MP4(mol, basis=basis, **kwargs)
    elif m == "ccd":
        from ..post_hf.ccd import CCD
        return CCD(mol, basis=basis, **kwargs)
    elif m == "casscf":
        from ..post_hf.casscf import CASSCF
        return CASSCF(mol, basis=basis, **kwargs)
    else:
        from ..core.errors import UnsupportedMethodError
        raise UnsupportedMethodError(f"Native backend does not support method '{method}'")


def _extract_orbitals(calc) -> tuple[list[OrbitalInfo], float | None, float | None]:
    """从计算对象提取轨道信息。兼容 mo_energy/mo_energies 两种属性名。"""
    mo_e = getattr(calc, "mo_energy", None)
    if mo_e is None:
        mo_e = getattr(calc, "mo_energies", None)
    mo_o = getattr(calc, "mo_occ", None)
    if mo_o is None:
        mo_o = getattr(calc, "mo_occupations", None)

    if mo_e is None:
        return [], None, None

    e_list = mo_e.tolist() if hasattr(mo_e, "tolist") else list(mo_e)
    o_list = [0.0] * len(e_list)
    if mo_o is not None:
        o_list = mo_o.tolist() if hasattr(mo_o, "tolist") else list(mo_o)

    orbitals = [
        OrbitalInfo(index=i, energy_hartree=float(e), occupation=float(o))
        for i, (e, o) in enumerate(zip(e_list, o_list))
    ]

    occupied = [o.energy_hartree for o in orbitals if o.occupation > 0.5]
    virtual = [o.energy_hartree for o in orbitals if o.occupation <= 0.5]
    homo = max(occupied) if occupied else None
    lumo = min(virtual) if virtual else None
    return orbitals, homo, lumo


@register_backend("native")
class NativeCalchem(Calculator):
    """CalChem 原生引擎后端。"""

    def __init__(self):
        self._caps = BackendCapabilities(
            name="native",
            version="0.3.0",
            tasks=["single_point", "optimize"],
            methods=_NATIVE_METHODS,
            basis_sets=_NATIVE_BASIS,
            requires_external_program=False,
            available=True,
        )

    def capabilities(self) -> BackendCapabilities:
        return self._caps

    def _try_rust(self, request: CalculationRequest, mol, started, t0) -> Result | None:
        """尝试使用 Rust 引擎计算。支持 RHF/STO-3G 任意分子（p 型积分已验证）。"""
        method = request.method.lower()
        basis = request.basis.lower()
        if method not in ("rhf", "hf"):
            return None
        if basis not in ("sto-3g", "sto3g"):
            return None

        # 检查 Rust 扩展是否可用
        try:
            from .. import _core as rust_core
        except ImportError:
            return None

        try:
            b2a = 0.529177210903
            symbols = list(request.molecule.symbols)
            coords_bohr = [[c / b2a for c in row] for row in request.molecule.coordinates_angstrom]

            result = rust_core.run_single_point(
                symbols, coords_bohr,
                request.molecule.charge,
                request.molecule.multiplicity,
                "rhf", "sto-3g",
            )
            walltime = time.time() - t0

            return Result(
                backend="native",
                backend_version="rust-0.2.0",
                task=request.task,
                method=request.method,
                actual_method="RHF-Rust",
                basis=request.basis,
                xc=request.xc,
                molecule=request.molecule,
                status="succeeded" if result.converged else "not_converged",
                started_at=started,
                finished_at=datetime.now(),
                walltime_seconds=walltime,
                scf_energy_hartree=result.total_energy_hartree,
                total_energy_hartree=result.total_energy_hartree,
                scf_converged=result.converged,
                scf_iterations=result.iterations,
                homo_energy_hartree=None,
                lumo_energy_hartree=None,
                orbitals=[],
                errors=[] if result.converged else ["Rust SCF not converged"],
            )
        except Exception:
            return None

    def single_point(self, request: CalculationRequest) -> Result:
        mol = _spec_to_molecule(request.molecule)
        started = datetime.now()
        t0 = time.time()

        # 尝试 Rust 引擎（优先路径）
        rust_result = self._try_rust(request, mol, started, t0)
        if rust_result is not None:
            return rust_result

        # Python 参考引擎（回退路径）
        calc = _build_calculator(mol, request.method, request.basis, request.xc, request.options)
        calc.kernel()
        walltime = time.time() - t0

        # 提取结果——严格禁止默认值假数据
        converged = getattr(calc, "converged", None)
        if converged is None:
            # 没有 converged 属性时，从能量是否有效判断
            energy_raw = getattr(calc, "energy", None)
            converged = energy_raw is not None and energy_raw != 0.0
        else:
            energy_raw = getattr(calc, "energy", None)

        # 能量必须是真实非零有限值
        energy = None
        if energy_raw is not None:
            try:
                e = float(energy_raw)
                if e != 0.0 and e == e:  # 非零且非 NaN
                    energy = e
            except (TypeError, ValueError):
                energy = None

        n_iter = getattr(calc, "n_cycles", None) or getattr(calc, "scf_cycles", None)

        orbitals, homo, lumo = _extract_orbitals(calc)

        dipole = getattr(calc, "dipole_moment", None)
        dipole_list = None
        if dipole is not None:
            try:
                dipole_list = [float(x) for x in (dipole.tolist() if hasattr(dipole, "tolist") else dipole)]
            except (TypeError, ValueError):
                dipole_list = None

        # Post-HF 关联能
        mp2_corr = getattr(calc, "e_corr", None)
        if mp2_corr is not None:
            try:
                mp2_corr = float(mp2_corr)
            except (TypeError, ValueError):
                mp2_corr = None

        # 严格判断成功条件
        errors = []
        if energy is None:
            errors.append("Could not extract valid energy from native calculation")
        if not converged:
            errors.append(f"SCF did not converge after {n_iter or '?'} iterations")

        # total_energy: 对 MP2/CCSD 等，total = SCF + correlation
        scf_e = energy
        total_e = energy
        if mp2_corr is not None and request.method.lower() in ("mp2", "mp4", "ccsd", "ccsd(t)"):
            total_e = energy + mp2_corr

        status = "succeeded" if (converged and energy is not None) else (
            "not_converged" if not converged and energy is not None else "failed"
        )

        return Result(
            backend="native",
            backend_version="0.2.0",
            task=request.task,
            method=request.method,
            actual_method=request.method,
            basis=request.basis,
            xc=request.xc,
            molecule=request.molecule,
            status=status,
            started_at=started,
            finished_at=datetime.now(),
            walltime_seconds=walltime,
            scf_energy_hartree=scf_e,
            mp2_correction_hartree=mp2_corr,
            total_energy_hartree=total_e,
            scf_converged=converged,
            scf_iterations=n_iter,
            homo_energy_hartree=homo,
            lumo_energy_hartree=lumo,
            orbitals=orbitals,
            dipole_debye=dipole_list,
            errors=errors,
        )

    def optimize(self, request: CalculationRequest) -> Result:
        """几何优化：RHF/STO-3G 用有限差分梯度 + BFGS。"""
        from ..grad.finite_difference import central_gradient
        from ..opt.numeric_optimizer import optimize_geometry

        started = datetime.now()
        t0 = time.time()

        # 能量函数
        def energy_fn(spec):
            req = CalculationRequest(
                task="single_point", molecule=spec, backend="native",
                method=request.method, basis=request.basis, xc=request.xc,
            )
            r = self.single_point(req)
            if r.scf_energy_hartree is None:
                return 1e10
            return r.scf_energy_hartree

        # 梯度函数（数值差分）
        def gradient_fn(spec):
            return central_gradient(energy_fn, spec, step_bohr=0.002)

        outcome = optimize_geometry(
            energy_fn, gradient_fn, request.molecule,
            max_iter=50, ftol=1e-7, gtol=1e-4,
        )
        walltime = time.time() - t0

        return Result(
            backend="native",
            backend_version="0.3.0",
            task="optimize",
            method=request.method,
            actual_method=request.method,
            basis=request.basis,
            xc=request.xc,
            molecule=request.molecule,
            status="succeeded" if outcome.converged else "not_converged",
            started_at=started,
            finished_at=datetime.now(),
            walltime_seconds=walltime,
            total_energy_hartree=outcome.final_energy_hartree,
            scf_energy_hartree=outcome.final_energy_hartree,
            optimized_coordinates_angstrom=outcome.final_molecule.coordinates_angstrom,
            errors=[] if outcome.converged else [f"Optimization: {outcome.message}"],
        )
