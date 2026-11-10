"""NativeCalchem: 把现有原生引擎包装成统一 Calculator 接口。

支持两条路径：
1. Rust 内核优先路径 (RHF / STO-3G / 6-31G, 闭壳层单重态)；
2. Python 参考引擎回退路径 (UHF, DFT, Post-HF, CASSCF)。
"""

from __future__ import annotations

import time
from datetime import datetime

from ..core.calculator import Calculator, register_backend
from ..core.capabilities import BackendCapabilities
from ..core.errors import ConvergenceError, UnsupportedMethodError
from ..core.request import CalculationRequest, MoleculeSpec
from ..core.result import OrbitalInfo, Result
from ..mol.molecule import Molecule


_NATIVE_METHODS = [
    "rhf", "hf", "uhf",
    "rks", "uks", "roks", "dft",
    "svwn", "svwn5", "lda", "blyp", "pbe", "b3lyp", "b3lyp5", "pbe0",
    "mp2", "mp4", "ccsd", "ccsd(t)", "ccsdt", "ccd", "casscf",
]

_RUST_DFT_FUNCTIONALS: dict[str, str] = {
    "svwn": "svwn5",
    "svwn5": "svwn5",
    "lda": "svwn5",
    "slatervwn5": "svwn5",
    "blyp": "blyp",
    "pbe": "pbe",
    "pbepbe": "pbe",
    "b3lyp": "b3lyp",
    "b3lyp5": "b3lyp5",
    "pbe0": "pbe0",
    "pbe1pbe": "pbe0",
}


def _normalize_xc_token(raw: str) -> str:
    return "".join(ch for ch in raw.strip().lower() if not ch.isspace() and ch not in "-_")


def _resolve_rust_scf_spec(
    method: str, xc: str | None, multiplicity: int, options: dict | None
) -> tuple[bool, str | None, bool, str | None]:
    """Resolves `(is_supported, canonical_xc_or_none, is_open_shell, reject_reason)` for Rust core."""
    m = method.strip().lower()
    m_norm = _normalize_xc_token(m)
    opts = options or {}

    if m in ("rhf", "hf", "uhf"):
        if m == "rhf" and multiplicity != 1:
            return False, None, False, f"multiplicity {multiplicity} != 1 (RHF requires closed-shell singlet)"
        use_u = m == "uhf" or multiplicity != 1
        return True, None, use_u, None

    if m in ("rks", "uks", "dft"):
        if m == "rks" and multiplicity != 1:
            return False, None, False, f"multiplicity {multiplicity} != 1 (RKS requires closed-shell singlet)"
        xc_raw = str(xc or opts.get("xc") or "b3lyp")
        xc_canon = _RUST_DFT_FUNCTIONALS.get(_normalize_xc_token(xc_raw))
        if xc_canon is None:
            return False, None, False, f"XC functional '{xc_raw}' not supported in Rust DFT core"
        use_u = m == "uks" or multiplicity != 1
        return True, xc_canon, use_u, None

    if m_norm in _RUST_DFT_FUNCTIONALS:
        xc_canon = _RUST_DFT_FUNCTIONALS[m_norm]
        use_u = multiplicity != 1
        return True, xc_canon, use_u, None

    return False, None, False, f"method '{method}' not handled by Rust HF/DFT core"

_NATIVE_BASIS = [
    "sto-3g", "3-21g", "6-31g", "6-31g(d)", "6-31g*", "6-31g(d,p)", "6-31g**",
    "lanl2dz", "lanl2dzdp",
    "cc-pvdz", "cc-pvtz", "def2-svp", "def2-tzvp", "d95",
]

_RUST_BASIS_ALIASES: dict[str, str] = {
    "sto-3g": "sto-3g",
    "sto3g": "sto-3g",
    "6-31g": "6-31g",
    "631g": "6-31g",
    "6-31g(d)": "6-31g(d)",
    "6-31g*": "6-31g(d)",
    "631gd": "6-31g(d)",
    "631g*": "6-31g(d)",
    "6-31gd": "6-31g(d)",
    "6-31g(d,p)": "6-31g(d,p)",
    "6-31g**": "6-31g(d,p)",
    "631gdp": "6-31g(d,p)",
    "631g**": "6-31g(d,p)",
    "6-31gdp": "6-31g(d,p)",
    "lanl2dz": "lanl2dz",
    "lanl-2dz": "lanl2dz",
    "lanl2dzdp": "lanl2dzdp",
    "lanl2dz(d,p)": "lanl2dzdp",
    "lanl2dz*": "lanl2dzdp",
}


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
    """根据 method 实例化对应的计算类。"""
    m = method.lower()
    kwargs = dict(options)

    if m in ("hf", "rhf"):
        from ..scf.hf import RHF
        return RHF(mol, basis=basis, **kwargs)
    elif m == "uhf":
        from ..scf.hf import UHF
        return UHF(mol, basis=basis, **kwargs)
    elif m == "rohf":
        raise UnsupportedMethodError(
            "Native backend does not support ROHF; use UHF or ROKS (DFT) instead."
        )
    elif m in ("dft", "rks"):
        from ..scf.dft import RKS
        return RKS(mol, basis=basis, xc=xc or "b3lyp", **kwargs)
    elif m == "uks":
        from ..scf.dft import UKS
        return UKS(mol, basis=basis, xc=xc or "b3lyp", **kwargs)
    elif m == "roks":
        from ..scf.dft import ROKS
        return ROKS(mol, basis=basis, xc=xc or "b3lyp", **kwargs)
    elif _normalize_xc_token(m) in _RUST_DFT_FUNCTIONALS:
        xc_canon = _RUST_DFT_FUNCTIONALS[_normalize_xc_token(m)]
        if mol.multiplicity > 1:
            from ..scf.dft import UKS
            return UKS(mol, basis=basis, xc=xc_canon, **kwargs)
        from ..scf.dft import RKS
        return RKS(mol, basis=basis, xc=xc_canon, **kwargs)
    elif m == "mp2":
        from ..scf.hf import RHF, UHF
        from ..post_hf.mp2 import RMP2, UMP2
        if mol.multiplicity > 1:
            mf = UHF(mol, basis=basis, **kwargs)
            return UMP2(mf)
        else:
            mf = RHF(mol, basis=basis, **kwargs)
            return RMP2(mf)
    elif m == "ccsd":
        from ..scf.hf import RHF
        from ..post_hf.ccsd import CCSD
        mf = RHF(mol, basis=basis, **kwargs)
        return CCSD(mf)
    elif m in ("ccsd(t)", "ccsdt"):
        from ..scf.hf import RHF
        from ..post_hf.ccsd_t import CCSDT
        mf = RHF(mol, basis=basis, **kwargs)
        return CCSDT(mf)
    elif m == "mp4":
        from ..scf.hf import RHF
        from ..post_hf.mp4 import MP4
        mf = RHF(mol, basis=basis)
        return MP4(mf, **kwargs)
    elif m == "ccd":
        from ..post_hf.ccd import CCD
        return CCD(mol, basis=basis, **kwargs)
    elif m == "casscf":
        from ..post_hf.casscf import CASSCF
        return CASSCF(mol, basis=basis, **kwargs)
    else:
        raise UnsupportedMethodError(f"Native backend does not support method '{method}'")


def _extract_orbitals(calc) -> tuple[list[OrbitalInfo], float | None, float | None]:
    """从计算对象提取轨道信息。兼容多种属性命名。"""
    mo_e = getattr(calc, "mo_energy", None)
    if mo_e is None:
        mo_e = getattr(calc, "mo_energies", None)
    mo_o = getattr(calc, "mo_occ", None)
    if mo_o is None:
        mo_o = getattr(calc, "mo_occupations", None)

    if mo_e is None:
        return [], None, None

    # UHF 情况下 mo_energy 为元组 (mo_e_a, mo_e_b)
    if isinstance(mo_e, (tuple, list)) and len(mo_e) == 2 and hasattr(mo_e[0], "__iter__"):
        mo_e = mo_e[0]
        if mo_o is not None and isinstance(mo_o, (tuple, list)) and len(mo_o) == 2:
            mo_o = mo_o[0]

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

    def _try_rust(self, request: CalculationRequest, mol: Molecule, started: datetime, t0: float) -> tuple[Result | None, str | None]:
        """尝试使用 Rust 引擎计算。

        支持闭壳层 RHF 与开壳层 UHF（STO-3G / 6-31G，任意合法电荷与多重度）。
        转发 request.options 中指定的迭代与收敛参数。
        若 Rust 抛出 ScfNotConvergedError，则如实返回未收敛结果，严禁静默回退。
        """
        method = request.method.lower()
        basis = request.basis.strip().lower()
        basis_arg = _RUST_BASIS_ALIASES.get(basis)

        opts = request.options or {}
        supported, xc_canon, use_unrestricted, reject_reason = _resolve_rust_scf_spec(
            method, request.xc, request.molecule.multiplicity, opts
        )
        if not supported:
            return None, reject_reason
        if basis_arg is None:
            return None, f"basis '{basis}' not embedded in Rust core"

        # 检查 Rust 扩展是否可用
        try:
            from .. import _core as rust_core
        except ImportError as e:
            return None, f"Rust extension import failed: {e}"

        b2a = 0.529177210903
        symbols = list(request.molecule.symbols)
        coords_bohr = [[c / b2a for c in row] for row in request.molecule.coordinates_angstrom]

        # 显式转发用户请求选项
        max_iter = int(opts.get("max_iter", opts.get("max_cycles", 100)))
        conv_energy = float(opts.get("conv_energy", opts.get("energy_tol", 1e-10)))
        conv_density = float(opts.get("conv_density", opts.get("density_tol", 1e-8)))
        use_diis = bool(opts.get("use_diis", opts.get("diis", True)))
        damping = float(opts.get("damping", 0.0))
        grid_level = str(opts.get("grid_level", opts.get("grid", "medium")))

        if xc_canon is None:
            actual_tag = f"UHF-Rust({basis_arg})" if use_unrestricted else f"RHF-Rust({basis_arg})"
        else:
            ks_tag = "UKS" if use_unrestricted else "RKS"
            actual_tag = f"{ks_tag}-{xc_canon.upper()}-Rust({basis_arg})"

        try:
            wf = rust_core.run_single_point(
                symbols, coords_bohr,
                request.molecule.charge,
                request.molecule.multiplicity,
                method, basis_arg,
                max_iter=max_iter,
                conv_energy=conv_energy,
                conv_density=conv_density,
                use_diis=use_diis,
                damping=damping,
                xc=xc_canon,
                grid_level=grid_level,
            )
            walltime = time.time() - t0

            orbitals = [
                OrbitalInfo(index=i, energy_hartree=float(e), occupation=float(o))
                for i, (e, o) in enumerate(zip(wf.mo_energies, wf.mo_occupations))
            ]
            dipole_list = [float(x) for x in wf.dipole_debye] if wf.dipole_debye is not None else None
            mull_q = low_q = mull_s = low_s = None
            if wf.converged and hasattr(rust_core, "compute_population_analysis"):
                pop = rust_core.compute_population_analysis(wf)
                mull_q = [float(x) for x in pop["mulliken"]]
                low_q = [float(x) for x in pop["lowdin"]]
                if pop.get("mulliken_spin") is not None:
                    mull_s = [float(x) for x in pop["mulliken_spin"]]
                if pop.get("lowdin_spin") is not None:
                    low_s = [float(x) for x in pop["lowdin_spin"]]

            return Result(
                job_id=request.job_id,
                project_id=request.project_id,
                backend="native",
                backend_version=f"rust-{getattr(rust_core, '__version__', '0.2.0')}",
                task=request.task,
                method=request.method,
                actual_method=actual_tag,
                basis=request.basis,
                xc=request.xc or xc_canon,
                molecule=request.molecule,
                status="succeeded" if wf.converged else "not_converged",
                started_at=started,
                finished_at=datetime.now(),
                walltime_seconds=walltime,
                scf_energy_hartree=wf.total_energy_hartree,
                total_energy_hartree=wf.total_energy_hartree,
                scf_converged=wf.converged,
                scf_iterations=wf.iterations,
                homo_energy_hartree=wf.homo_energy_hartree,
                lumo_energy_hartree=wf.lumo_energy_hartree,
                orbitals=orbitals,
                dipole_debye=dipole_list,
                dipole_magnitude_debye=wf.dipole_magnitude_debye,
                mulliken_charges=mull_q,
                lowdin_charges=low_q,
                mulliken_spin_populations=mull_s,
                lowdin_spin_populations=low_s,
                errors=[] if wf.converged else ["Rust SCF not converged"],
            ), None

        except getattr(rust_core, "ScfNotConvergedError", RuntimeError) as e:
            # 真实 SCF 未收敛：直接如实返回，不掩盖、不静默回退
            walltime = time.time() - t0
            return Result(
                job_id=request.job_id,
                project_id=request.project_id,
                backend="native",
                backend_version=f"rust-{getattr(rust_core, '__version__', '0.2.0')}",
                task=request.task,
                method=request.method,
                actual_method=actual_tag,
                basis=request.basis,
                xc=request.xc or xc_canon,
                molecule=request.molecule,
                status="not_converged",
                started_at=started,
                finished_at=datetime.now(),
                walltime_seconds=walltime,
                scf_converged=False,
                errors=[str(e)],
            ), None

        except Exception as e:
            # 其他非预期异常，记录原因并允许回退到 Python
            return None, str(e)

    def single_point(self, request: CalculationRequest) -> Result:
        mol = _spec_to_molecule(request.molecule)
        started = datetime.now()
        t0 = time.time()

        # 尝试 Rust 引擎（优先路径）
        rust_result, fallback_reason = self._try_rust(request, mol, started, t0)
        if rust_result is not None:
            return rust_result

        # Python 参考引擎（回退路径）
        calc = _build_calculator(mol, request.method, request.basis, request.xc, request.options)
        calc.kernel()
        walltime = time.time() - t0

        # 提取收敛与能量
        converged = getattr(calc, "converged", None)
        if converged is None:
            energy_raw = getattr(calc, "energy", None)
            if energy_raw is None:
                energy_raw = getattr(calc, "e_tot", None)
            converged = energy_raw is not None and energy_raw != 0.0
        else:
            energy_raw = getattr(calc, "energy", None)
            if energy_raw is None:
                energy_raw = getattr(calc, "e_tot", None)

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

        # 关联能处理（MP2, MP4, CCSD, CCSD(T)）
        m_lower = request.method.lower()
        is_post_hf = m_lower in ("mp2", "mp4", "ccsd", "ccsd(t)", "ccsdt")

        mp2_corr = None
        ccsd_corr = None
        ccsd_t_corr = None
        scf_e = energy
        total_e = energy

        if is_post_hf:
            # 对 post-HF，calc.energy/calc.e_tot 已经是总能量
            ref_hf = getattr(calc, "mf", getattr(calc, "hf", None))
            scf_e = getattr(ref_hf, "e_tot", getattr(calc, "e_hf", None))
            if m_lower == "mp2":
                mp2_corr = getattr(calc, "e_corr", None)
            elif m_lower in ("ccsd", "ccsd(t)", "ccsdt"):
                ccsd_corr = getattr(calc, "e_corr", None)
                ccsd_t_corr = getattr(calc, "e_t", None)
            elif m_lower == "mp4":
                mp2_corr = getattr(calc, "e_mp2", None)

        errors = []
        warnings = []
        if fallback_reason:
            warnings.append(f"Rust core bypassed ({fallback_reason}); used Python reference engine.")

        if energy is None:
            errors.append("Could not extract valid energy from native calculation")
        if not converged:
            errors.append(f"SCF did not converge after {n_iter or '?'} iterations")

        status = "succeeded" if (converged and energy is not None) else (
            "not_converged" if not converged and energy is not None else "failed"
        )

        return Result(
            job_id=request.job_id,
            project_id=request.project_id,
            backend="native",
            backend_version="0.3.0",
            task=request.task,
            method=request.method,
            actual_method=f"Python-{request.method}",
            basis=request.basis,
            xc=request.xc,
            molecule=request.molecule,
            status=status,
            started_at=started,
            finished_at=datetime.now(),
            walltime_seconds=walltime,
            scf_energy_hartree=scf_e,
            mp2_correction_hartree=mp2_corr,
            ccsd_correction_hartree=ccsd_corr,
            ccsd_t_correction_hartree=ccsd_t_corr,
            total_energy_hartree=total_e,
            scf_converged=converged,
            scf_iterations=n_iter,
            homo_energy_hartree=homo,
            lumo_energy_hartree=lumo,
            orbitals=orbitals,
            dipole_debye=dipole_list,
            errors=errors,
            warnings=warnings,
        )

    def optimize(self, request: CalculationRequest) -> Result:
        """几何优化：优先 Rust 解析梯度 BFGS，回退 Python 数值梯度。"""
        from ..utils.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM

        started = datetime.now()
        t0 = time.time()

        # 检查 Rust 扩展是否可用
        try:
            from .. import _core as rust_core
        except ImportError:
            rust_core = None

        # 优先尝试 Rust 原生 BFGS + 解析梯度（闭壳层 RHF/RKS 与开壳层/离子态 UHF/UKS，含极化基组与 ECP）
        m_low = request.method.lower()
        b_low = _RUST_BASIS_ALIASES.get(request.basis.strip().lower(), request.basis.strip().lower())
        opts = request.options or {}
        opt_supported, xc_canon, use_unrestricted, _ = _resolve_rust_scf_spec(
            m_low, request.xc, request.molecule.multiplicity, opts
        )
        can_rust_opt = (
            rust_core is not None
            and hasattr(rust_core, "run_optimize")
            and opt_supported
            and b_low in rust_core.supported_basis_sets()
        )
        if can_rust_opt:
            try:
                coords_bohr = [
                    [c * ANGSTROM_TO_BOHR for c in pos]
                    for pos in request.molecule.coordinates_angstrom
                ]
                max_iter = int(opts.get("max_iter", 100))
                grid_level = str(opts.get("grid_level", opts.get("grid", "medium")))
                opt_res = rust_core.run_optimize(
                    request.molecule.symbols,
                    coords_bohr,
                    b_low,
                    max_iter,
                    charge=request.molecule.charge,
                    multiplicity=request.molecule.multiplicity,
                    method=m_low,
                    xc=xc_canon,
                    grid_level=grid_level,
                )
                if opt_res.converged:
                    final_coords_bohr = [
                        opt_res.coords[3 * i : 3 * i + 3]
                        for i in range(len(request.molecule.symbols))
                    ]
                    final_coords_ang = [
                        [c * BOHR_TO_ANGSTROM for c in pos]
                        for pos in final_coords_bohr
                    ]
                    walltime = time.time() - t0
                    if xc_canon is None:
                        opt_tag = "Rust-UHF-BFGS" if use_unrestricted else "Rust-RHF-BFGS"
                    else:
                        ks_tag = "UKS" if use_unrestricted else "RKS"
                        opt_tag = f"Rust-{ks_tag}-{xc_canon.upper()}-BFGS"
                    return Result(
                        job_id=request.job_id,
                        project_id=request.project_id,
                        backend="native",
                        backend_version="0.3.0",
                        task="optimize",
                        method=request.method,
                        actual_method=opt_tag,
                        basis=request.basis,
                        xc=request.xc or xc_canon,
                        molecule=request.molecule,
                        status="succeeded",
                        started_at=started,
                        finished_at=datetime.now(),
                        walltime_seconds=walltime,
                        total_energy_hartree=opt_res.energy,
                        scf_energy_hartree=opt_res.energy,
                        optimized_coordinates_angstrom=final_coords_ang,
                        errors=[],
                    )
            except Exception:
                # 优雅回退到 Python 优化路径
                pass

        from ..grad.finite_difference import central_gradient
        from ..opt.numeric_optimizer import optimize_geometry

        # 能量函数：必须成功收敛
        def energy_fn(spec: MoleculeSpec) -> float:
            req = CalculationRequest(
                task="single_point",
                molecule=spec,
                backend="native",
                method=request.method,
                basis=request.basis,
                xc=request.xc,
                options=request.options,
            )
            r = self.single_point(req)
            if r.status != "succeeded" or r.total_energy_hartree is None:
                raise ConvergenceError(
                    f"Single point energy evaluation failed during optimization: {r.errors}"
                )
            return r.total_energy_hartree

        def gradient_fn(spec: MoleculeSpec):
            return central_gradient(energy_fn, spec, step_bohr=0.002)

        max_iter = int(request.options.get("max_iter", 50))
        gtol = float(request.options.get("gtol", 1e-4))

        try:
            outcome = optimize_geometry(
                energy_fn, gradient_fn, request.molecule,
                max_iter=max_iter, gtol=gtol,
            )
            status = "succeeded" if outcome.converged else "not_converged"
            errors = [] if outcome.converged else [f"Optimization: {outcome.message}"]
            final_coords = outcome.final_molecule.coordinates_angstrom
            final_energy = outcome.final_energy_hartree
        except Exception as e:
            status = "failed"
            errors = [f"Optimization failed: {e}"]
            final_coords = None
            final_energy = None

        walltime = time.time() - t0

        return Result(
            job_id=request.job_id,
            project_id=request.project_id,
            backend="native",
            backend_version="0.3.0",
            task="optimize",
            method=request.method,
            actual_method=request.method,
            basis=request.basis,
            xc=request.xc,
            molecule=request.molecule,
            status=status,
            started_at=started,
            finished_at=datetime.now(),
            walltime_seconds=walltime,
            total_energy_hartree=final_energy,
            scf_energy_hartree=final_energy,
            optimized_coordinates_angstrom=final_coords,
            errors=errors,
        )
