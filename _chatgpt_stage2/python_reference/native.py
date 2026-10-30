"""NativeCalchem: 把现有原生引擎包装成统一 Calculator 接口。

不重写任何 SCF/积分/DFT 算法——直接实例化计算类并提取结果。
在 Stage 2 中，优先调用已通过 s/p/d 积分与 RHF 验证的 `_calchem_rust` 原生核心，
若 `_calchem_rust` 未编译或当前任务超出 Stage 2 RHF 范围，则自动回退到 Python 原生实现。
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

_STAGE2_RUST_ELEMENTS = {
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
}

_STAGE2_RUST_BASIS = {"sto-3g", "sto3g", "6-31g", "631g"}


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

    if isinstance(mo_e, tuple):
        orbitals: list[OrbitalInfo] = []
        occ_tuple = mo_o if isinstance(mo_o, tuple) else (None, None)
        for spin_label, energies, occs in zip(("alpha", "beta"), mo_e, occ_tuple):
            for idx, e_val in enumerate(energies):
                occ_val = float(occs[idx]) if occs is not None else 0.0
                orbitals.append(
                    OrbitalInfo(
                        index=idx,
                        energy_hartree=float(e_val),
                        occupation=occ_val,
                        spin=spin_label,
                    )
                )
        occupied = [o.energy_hartree for o in orbitals if o.occupation > 0.5]
        virtual = [o.energy_hartree for o in orbitals if o.occupation <= 0.5]
        homo = max(occupied) if occupied else None
        lumo = min(virtual) if virtual else None
        return orbitals, homo, lumo

    orbitals = []
    for idx, e_val in enumerate(mo_e):
        occ_val = float(mo_o[idx]) if mo_o is not None else 0.0
        orbitals.append(
            OrbitalInfo(
                index=idx,
                energy_hartree=float(e_val),
                occupation=occ_val,
                spin="restricted",
            )
        )
    occupied = [o.energy_hartree for o in orbitals if o.occupation > 0.5]
    virtual = [o.energy_hartree for o in orbitals if o.occupation <= 0.5]
    homo = max(occupied) if occupied else None
    lumo = min(virtual) if virtual else None
    return orbitals, homo, lumo


class NativeCalchem(Calculator):
    """CalChem 自研量子化学引擎后端。"""

    name = "native"
    capabilities = BackendCapabilities(
        methods=_NATIVE_METHODS,
        basis_sets=_NATIVE_BASIS,
        supports_gradient=True,
        supports_hessian=True,
        supports_optimization=True,
        supports_irc=True,
        supports_ts=True,
        is_external=False,
    )

    def is_available(self) -> bool:
        return True

    def _run(self, request: CalculationRequest) -> Result:
        t0 = time.perf_counter()
        mol = _spec_to_molecule(request.molecule)
        task = request.task.value

        if task == "single_point":
            return self._run_single_point(request, mol, t0)
        elif task == "gradient":
            return self._run_gradient(request, mol, t0)
        elif task == "optimize":
            return self._run_optimize(request, mol, t0)
        elif task == "ts_search":
            return self._run_ts(request, mol, t0)
        elif task == "frequency":
            return self._run_frequency(request, mol, t0)
        elif task == "irc":
            return self._run_irc(request, mol, t0)
        else:
            from ..core.errors import UnsupportedMethodError
            raise UnsupportedMethodError(f"Task '{task}' not implemented in native backend")

    def _run_single_point(self, request: CalculationRequest, mol: Molecule, t0: float) -> Result:
        # 优先尝试 Rust Stage 2 Core 引擎 (支持已验证的闭壳层 RHF + STO-3G / 6-31G)
        def _try_rust():
            try:
                import _calchem_rust
                m_lower = request.method.lower()
                b_lower = request.basis.lower()
                if m_lower not in ("rhf", "hf"):
                    return None
                if b_lower not in _STAGE2_RUST_BASIS:
                    return None
                if request.molecule.multiplicity != 1:
                    return None
                if not all(sym in _STAGE2_RUST_ELEMENTS for sym in mol.symbols):
                    return None

                mol_bohr = mol.to_bohr()
                coords = [list(c) for c in mol_bohr.coordinates]
                wf = _calchem_rust.run_single_point(
                    symbols=mol.symbols,
                    geometry_bohr=coords,
                    charge=request.molecule.charge,
                    multiplicity=request.molecule.multiplicity,
                    method=m_lower,
                    basis=b_lower,
                    max_iter=request.options.get("max_iter", 100),
                    conv_energy=request.options.get("energy_tol", 1e-10),
                    conv_density=request.options.get("density_tol", 1e-8),
                )
                if wf.converged:
                    wall = time.perf_counter() - t0
                    return Result(
                        request=request,
                        backend_used=self.name,
                        total_energy_hartree=float(wf.total_energy_hartree),
                        converged=True,
                        scf_iterations=wf.iterations,
                        wall_time_seconds=wall,
                        timestamp=datetime.now(),
                    )
            except Exception:
                pass
            return None

        rust_res = _try_rust()
        if rust_res is not None:
            return rust_res

        calc = _build_calculator(mol, request.method, request.basis, request.xc_functional, request.options)
        energy = calc.run()

        scf_ref = getattr(calc, "scf", calc)
        converged = getattr(scf_ref, "converged", True)
        if not converged:
            raise ConvergenceError(
                f"{request.method.upper()} did not converge",
                iterations=getattr(scf_ref, "_n_iter", None),
            )

        orbitals, homo, lumo = _extract_orbitals(scf_ref)
        gap = (lumo - homo) if (homo is not None and lumo is not None) else None

        wall = time.perf_counter() - t0
        return Result(
            request=request,
            backend_used=self.name,
            total_energy_hartree=float(energy),
            converged=True,
            scf_iterations=getattr(scf_ref, "_n_iter", None),
            orbitals=orbitals,
            homo_hartree=homo,
            lumo_hartree=lumo,
            homo_lumo_gap_hartree=gap,
            wall_time_seconds=wall,
            timestamp=datetime.now(),
        )

    def _run_gradient(self, request: CalculationRequest, mol: Molecule, t0: float) -> Result:
        calc = _build_calculator(mol, request.method, request.basis, request.xc_functional, request.options)
        energy = calc.run()

        m = request.method.lower()
        if m in ("hf", "rhf", "uhf", "rohf"):
            from ..grad.hf_grad import HFGradient
            grad_obj = HFGradient(calc)
            grad = grad_obj.compute()
        elif m in ("dft", "rks", "uks"):
            from ..grad.dft_grad import DFTGradient
            grad_obj = DFTGradient(calc)
            grad = grad_obj.compute()
        else:
            from ..grad.finite_difference import FiniteDifferenceGradient
            fd = FiniteDifferenceGradient(calc)
            grad = fd.compute()

        wall = time.perf_counter() - t0
        return Result(
            request=request,
            backend_used=self.name,
            total_energy_hartree=float(energy),
            gradient_hartree_bohr=[list(row) for row in grad],
            converged=True,
            wall_time_seconds=wall,
            timestamp=datetime.now(),
        )

    def _run_optimize(self, request: CalculationRequest, mol: Molecule, t0: float) -> Result:
        from ..api.high_level import optimize as hl_opt
        opt_res = hl_opt(
            mol,
            method=request.method,
            basis=request.basis,
            xc=request.xc_functional or "b3lyp",
            **request.options,
        )
        wall = time.perf_counter() - t0
        return Result(
            request=request,
            backend_used=self.name,
            total_energy_hartree=float(opt_res.final_energy),
            converged=bool(opt_res.converged),
            wall_time_seconds=wall,
            timestamp=datetime.now(),
        )

    def _run_ts(self, request: CalculationRequest, mol: Molecule, t0: float) -> Result:
        from ..api.high_level import ts_search as hl_ts
        ts_res = hl_ts(
            mol,
            method=request.method,
            basis=request.basis,
            xc=request.xc_functional or "b3lyp",
            **request.options,
        )
        wall = time.perf_counter() - t0
        return Result(
            request=request,
            backend_used=self.name,
            total_energy_hartree=float(ts_res.final_energy),
            converged=bool(ts_res.converged),
            wall_time_seconds=wall,
            timestamp=datetime.now(),
        )

    def _run_frequency(self, request: CalculationRequest, mol: Molecule, t0: float) -> Result:
        from ..api.high_level import frequency as hl_freq
        freq_res = hl_freq(
            mol,
            method=request.method,
            basis=request.basis,
            xc=request.xc_functional or "b3lyp",
            **request.options,
        )
        wall = time.perf_counter() - t0
        return Result(
            request=request,
            backend_used=self.name,
            total_energy_hartree=float(getattr(freq_res, "energy", 0.0)),
            frequencies_cm1=[float(f) for f in getattr(freq_res, "frequencies", [])],
            zero_point_energy_hartree=float(getattr(freq_res, "zpe", 0.0)),
            converged=True,
            wall_time_seconds=wall,
            timestamp=datetime.now(),
        )

    def _run_irc(self, request: CalculationRequest, mol: Molecule, t0: float) -> Result:
        from ..api.high_level import irc as hl_irc
        irc_res = hl_irc(
            mol,
            method=request.method,
            basis=request.basis,
            xc=request.xc_functional or "b3lyp",
            **request.options,
        )
        wall = time.perf_counter() - t0
        energies = [float(e) for e in getattr(irc_res, "energies", [])]
        return Result(
            request=request,
            backend_used=self.name,
            total_energy_hartree=energies[-1] if energies else 0.0,
            irc_energies_hartree=energies,
            converged=True,
            wall_time_seconds=wall,
            timestamp=datetime.now(),
        )


register_backend(NativeCalchem())
