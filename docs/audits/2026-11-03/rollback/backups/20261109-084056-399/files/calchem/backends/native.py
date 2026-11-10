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
from ..core.result import ExcitedStateInfo, OrbitalInfo, OrbitalTransitionInfo, Result
from ..mol.molecule import Molecule


_NATIVE_METHODS = [
    "rhf", "hf", "uhf",
    "rks", "uks", "roks", "dft",
    "svwn", "svwn5", "lda", "blyp", "pbe", "b3lyp", "b3lyp5", "pbe0",
    "camb3lyp", "cam-b3lyp", "wb97xd", "wb97x-d", "wb97x",
    "mp2", "mp4", "ccsd", "ccsd(t)", "ccsdt", "ccd", "casscf",
    "cis", "tdhf", "tddft", "tda", "tda-dft",
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
    "camb3lyp": "cam-b3lyp",
    "wb97xd": "wb97x-d",
    "wb97x": "wb97x-d",
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
            tasks=["single_point", "optimize", "tddft", "cis", "excited_states", "frequency", "ts", "irc", "solvation"],
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

            # D3 色散能处理 (若请求 dispersion 或 d3)
            disp = opts.get("dispersion", opts.get("d3", None))
            d3_energy = None
            if disp and wf.converged and hasattr(rust_core, "run_d3_dispersion"):
                d3_damp = "d3bj" if (disp is True or str(disp).lower() in ("d3bj", "bj")) else "d3zero"
                try:
                    d3_calc = rust_core.run_d3_dispersion(
                        symbols, coords_bohr, damping=d3_damp, functional=xc_canon or "b3lyp"
                    )
                    d3_energy = float(d3_calc.energy_hartree)
                except Exception:
                    d3_energy = None

            tot_energy = wf.total_energy_hartree
            if d3_energy is not None:
                tot_energy += d3_energy

            res_extra = {}
            if d3_energy is not None:
                res_extra["d3_dispersion_hartree"] = d3_energy

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
                total_energy_hartree=tot_energy,
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
                extra=res_extra,
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
        if request.solvent is not None:
            return self.solvation(request)
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
                dispersion = opts.get("dispersion", opts.get("d3", None))
                if isinstance(dispersion, bool):
                    dispersion = "d3bj" if dispersion else None
                solvent = request.solvent or opts.get("solvent", None)
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
                    dispersion=dispersion,
                    solvent=solvent,
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

    def tddft(self, request: CalculationRequest) -> Result:
        """纯 Rust 线性响应激发态 (TD-DFT / TDA / CIS) 计算。"""
        started = datetime.now()
        t0 = time.time()
        opts = request.options or {}

        basis = request.basis.strip().lower()
        basis_arg = _RUST_BASIS_ALIASES.get(basis)
        if basis_arg is None:
            raise UnsupportedMethodError(f"Basis '{basis}' is not supported by native TD-DFT engine")

        try:
            from .. import _core as rust_core
        except ImportError as e:
            raise RuntimeError(f"Rust core not available for TD-DFT: {e}")

        b2a = 0.529177210903
        symbols = list(request.molecule.symbols)
        coords_bohr = [[c / b2a for c in row] for row in request.molecule.coordinates_angstrom]

        n_states = int(opts.get("n_states", opts.get("roots", 5)))
        tda = bool(opts.get("tda", request.method.lower() in ("cis", "tda", "tda-dft") or request.task == "cis"))
        manifold = str(opts.get("manifold", opts.get("spin", "singlet")))
        grid_level = str(opts.get("grid_level", opts.get("grid", "medium")))
        max_iter = int(opts.get("max_iter", opts.get("davidson_max_iter", 80)))
        conv_tol = float(opts.get("conv_tol", opts.get("tol", 1e-6)))
        force_davidson = bool(opts.get("force_davidson", False))
        scf_max_iter = int(opts.get("scf_max_iter", 100))
        scf_conv_energy = float(opts.get("scf_conv_energy", 1e-10))
        scf_conv_density = float(opts.get("scf_conv_density", 1e-8))

        xc = request.xc or opts.get("xc")
        method = request.method.lower()
        if method in ("cis", "tdhf"):
            xc = None

        try:
            td_res = rust_core.run_tddft(
                symbols,
                coords_bohr,
                charge=request.molecule.charge,
                multiplicity=request.molecule.multiplicity,
                method=method,
                basis=basis_arg,
                n_states=n_states,
                tda=tda,
                manifold=manifold,
                xc=xc,
                grid_level=grid_level,
                max_iter=max_iter,
                conv_tol=conv_tol,
                force_davidson=force_davidson,
                scf_max_iter=scf_max_iter,
                scf_conv_energy=scf_conv_energy,
                scf_conv_density=scf_conv_density,
            )
        except getattr(rust_core, "ScfNotConvergedError", RuntimeError) as e:
            walltime = time.time() - t0
            return Result(
                job_id=request.job_id,
                project_id=request.project_id,
                backend="native",
                backend_version=f"rust-{getattr(rust_core, '__version__', '0.2.0')}",
                task=request.task,
                method=request.method,
                basis=request.basis,
                xc=request.xc or xc,
                molecule=request.molecule,
                status="not_converged",
                started_at=started,
                finished_at=datetime.now(),
                walltime_seconds=walltime,
                scf_converged=False,
                errors=[str(e)],
            )

        walltime = time.time() - t0
        py_states = []
        for st in td_res.states:
            dom = [
                OrbitalTransitionInfo(
                    occ_index=t.occ_index,
                    virt_index=t.virt_index,
                    amplitude_x=t.amplitude_x,
                    amplitude_y=t.amplitude_y,
                    weight=t.weight,
                )
                for t in st.dominant_transitions
            ]
            py_states.append(
                ExcitedStateInfo(
                    state_index=st.state_index,
                    manifold_index=st.manifold_index,
                    label=st.label,
                    multiplicity=st.multiplicity,
                    excitation_energy_hartree=st.excitation_energy_hartree,
                    excitation_energy_ev=st.excitation_energy_ev,
                    total_energy_hartree=st.total_energy_hartree,
                    wavelength_nm=st.wavelength_nm,
                    transition_dipole_au=list(st.transition_dipole_au),
                    oscillator_strength=st.oscillator_strength,
                    dominant_transitions=dom,
                )
            )

        wf = td_res.scf_wavefunction
        orbitals = [
            OrbitalInfo(index=i, energy_hartree=float(e), occupation=float(o))
            for i, (e, o) in enumerate(zip(wf.mo_energies, wf.mo_occupations))
        ]
        dipole_list = [float(x) for x in wf.dipole_debye] if wf.dipole_debye is not None else None

        mull_q = low_q = None
        if hasattr(rust_core, "compute_population_analysis"):
            pop = rust_core.compute_population_analysis(wf)
            mull_q = [float(x) for x in pop["mulliken"]]
            low_q = [float(x) for x in pop["lowdin"]]

        actual_tag = f"TDDFT-{td_res.xc_functional or 'HF'}{'-TDA' if td_res.tda else ''}-Rust({basis_arg})"

        return Result(
            job_id=request.job_id,
            project_id=request.project_id,
            backend="native",
            backend_version=f"rust-{getattr(rust_core, '__version__', '0.2.0')}",
            task=request.task,
            method=request.method,
            actual_method=actual_tag,
            basis=request.basis,
            xc=td_res.xc_functional or request.xc,
            molecule=request.molecule,
            status="succeeded",
            started_at=started,
            finished_at=datetime.now(),
            walltime_seconds=walltime,
            scf_energy_hartree=td_res.ground_energy_hartree,
            total_energy_hartree=td_res.ground_energy_hartree,
            scf_converged=wf.converged,
            scf_iterations=wf.iterations,
            homo_energy_hartree=td_res.homo_energy_hartree,
            lumo_energy_hartree=td_res.lumo_energy_hartree,
            optical_gap_ev=td_res.optical_gap_ev,
            lowest_triplet_ev=td_res.lowest_triplet_ev,
            delta_est_ev=td_res.delta_est_ev,
            excited_states=py_states,
            orbitals=orbitals,
            dipole_debye=dipole_list,
            dipole_magnitude_debye=wf.dipole_magnitude_debye,
            mulliken_charges=mull_q,
            lowdin_charges=low_q,
            extra={
                "tddft_result": td_res,
                "davidson_iterations_singlet": td_res.davidson_iterations_singlet,
                "davidson_iterations_triplet": td_res.davidson_iterations_triplet,
                "ks_gap_ev": td_res.ks_gap_ev,
            },
        )

    def frequency(self, request: CalculationRequest) -> Result:
        """纯 Rust 核坐标有限差分二阶 Hessian 矩阵、简正模式分析与 RRHO 热力学计算。"""
        started = datetime.now()
        t0 = time.time()
        opts = request.options or {}

        basis = request.basis.strip().lower()
        basis_arg = _RUST_BASIS_ALIASES.get(basis)
        if basis_arg is None:
            raise UnsupportedMethodError(f"Basis '{basis}' is not supported by native frequency engine")

        try:
            from .. import _core as rust_core
        except ImportError as e:
            raise RuntimeError(f"Rust core not available for frequency: {e}")

        b2a = 0.529177210903
        symbols = list(request.molecule.symbols)
        coords_bohr = [[c / b2a for c in row] for row in request.molecule.coordinates_angstrom]

        step_bohr = float(opts.get("step_bohr", opts.get("step", 0.001)))
        grid_level = str(opts.get("grid_level", opts.get("grid", "medium")))
        temperature_k = float(opts.get("temperature_k", opts.get("temperature", 298.15)))
        pressure_pa = float(opts.get("pressure_pa", opts.get("pressure", 101325.0)))
        symmetry_number = opts.get("symmetry_number", opts.get("sigma"))
        if symmetry_number is not None:
            symmetry_number = int(symmetry_number)
        max_iter = int(opts.get("max_iter", opts.get("max_cycles", 100)))
        conv_energy = float(opts.get("conv_energy", opts.get("energy_tol", 1e-10)))
        conv_density = float(opts.get("conv_density", opts.get("density_tol", 1e-8)))

        xc = request.xc or opts.get("xc")
        method = request.method.lower()

        try:
            freq_res = rust_core.run_frequency(
                symbols,
                coords_bohr,
                charge=request.molecule.charge,
                multiplicity=request.molecule.multiplicity,
                method=method,
                basis=basis_arg,
                step_bohr=step_bohr,
                xc=xc,
                grid_level=grid_level,
                temperature_k=temperature_k,
                pressure_pa=pressure_pa,
                symmetry_number=symmetry_number,
                max_iter=max_iter,
                conv_energy=conv_energy,
                conv_density=conv_density,
            )
        except getattr(rust_core, "ScfNotConvergedError", RuntimeError) as e:
            walltime = time.time() - t0
            return Result(
                job_id=request.job_id,
                project_id=request.project_id,
                backend="native",
                backend_version=f"rust-{getattr(rust_core, '__version__', '0.2.0')}",
                task=request.task,
                method=request.method,
                basis=request.basis,
                xc=request.xc or xc,
                molecule=request.molecule,
                status="not_converged",
                started_at=started,
                finished_at=datetime.now(),
                walltime_seconds=walltime,
                scf_converged=False,
                errors=[str(e)],
            )

        walltime = time.time() - t0

        from ..core.result import VibrationalModeInfo

        vibrational_modes = [
            VibrationalModeInfo(
                mode_index=m.mode_index,
                frequency_cm=m.frequency_cm,
                is_imaginary=m.is_imaginary,
                reduced_mass_amu=m.reduced_mass_amu,
                force_constant_m_dyn_a=m.force_constant_m_dyn_a,
                ir_intensity_km_mol=m.ir_intensity_km_mol,
                displacement=[list(d) for d in m.displacement],
            )
            for m in freq_res.vibrational.modes
        ]

        actual_tag = f"{freq_res.method.upper()}-Rust({basis_arg})"
        if freq_res.xc_functional:
            actual_tag = f"{actual_tag}[{freq_res.xc_functional}]"

        return Result(
            job_id=request.job_id,
            project_id=request.project_id,
            backend="native",
            backend_version=f"rust-{getattr(rust_core, '__version__', '0.2.0')}",
            task=request.task,
            method=request.method,
            actual_method=actual_tag,
            basis=request.basis,
            xc=request.xc or xc,
            molecule=request.molecule,
            status="succeeded",
            started_at=started,
            finished_at=datetime.now(),
            walltime_seconds=walltime,
            total_energy_hartree=freq_res.energy_hartree,
            scf_energy_hartree=freq_res.energy_hartree,
            scf_converged=True,
            frequencies_cm=list(freq_res.frequencies_cm),
            ir_intensities_km_mol=list(freq_res.ir_intensities_km_mol),
            zpve_hartree=freq_res.zpve_hartree,
            zpve_kcal_mol=freq_res.thermo.zpve_kcal_mol,
            thermal_energy_hartree=freq_res.thermo.e_thermal_hartree,
            enthalpy_hartree=freq_res.enthalpy_hartree,
            gibbs_free_energy_hartree=freq_res.gibbs_free_energy_hartree,
            entropy_cal_mol_k=freq_res.entropy_cal_mol_k,
            heat_capacity_cv_cal_mol_k=freq_res.heat_capacity_cv_cal_mol_k,
            heat_capacity_cp_cal_mol_k=freq_res.heat_capacity_cp_cal_mol_k,
            vibrational_modes=vibrational_modes,
            extra={
                "n_vib": freq_res.vibrational.n_vib,
                "is_linear": freq_res.vibrational.is_linear,
                "principal_moments_amu_bohr2": list(freq_res.vibrational.principal_moments_amu_bohr2),
                "rotational_constants_ghz": list(freq_res.vibrational.rotational_constants_ghz),
                "trans_rot_residuals": list(freq_res.vibrational.trans_rot_residuals),
                "cartesian_hessian": freq_res.cartesian_hessian,
                "apt": freq_res.apt,
            },
        )

    def ts_search(self, request: CalculationRequest) -> Result:
        """纯 Rust 分区有理函数优化 (P-RFO) 过渡态鞍点搜索。"""
        started = datetime.now()
        t0 = time.time()
        opts = request.options or {}

        basis = request.basis.strip().lower()
        basis_arg = _RUST_BASIS_ALIASES.get(basis)
        if basis_arg is None:
            raise UnsupportedMethodError(f"Basis '{basis}' is not supported by native TS engine")

        try:
            from .. import _core as rust_core
        except ImportError as e:
            raise RuntimeError(f"Rust core not available for TS search: {e}")

        b2a = 0.529177210903
        symbols = list(request.molecule.symbols)
        coords_bohr = [[c / b2a for c in row] for row in request.molecule.coordinates_angstrom]

        max_steps = int(opts.get("max_steps", opts.get("max_cycles", 50)))
        max_force = float(opts.get("max_force", 3e-4))
        rms_force = float(opts.get("rms_force", 1.5e-4))
        max_displacement = float(opts.get("max_displacement", 1.2e-3))
        trust_radius_bohr = float(opts.get("trust_radius_bohr", opts.get("trust_radius", 0.1)))
        mode_index = opts.get("mode_index")
        if mode_index is not None:
            mode_index = int(mode_index)
        recalc_hessian_interval = int(opts.get("recalc_hessian_interval", 0))
        grid_level = str(opts.get("grid_level", opts.get("grid", "medium")))
        scf_max_iter = int(opts.get("scf_max_iter", opts.get("max_iter", 100)))
        scf_conv_energy = float(opts.get("scf_conv_energy", opts.get("conv_energy", 1e-11)))
        scf_conv_density = float(opts.get("scf_conv_density", opts.get("conv_density", 1e-10)))

        xc = request.xc or opts.get("xc")
        method = request.method.lower()

        try:
            ts_res = rust_core.run_ts_search(
                symbols,
                coords_bohr,
                charge=request.molecule.charge,
                multiplicity=request.molecule.multiplicity,
                method=method,
                basis=basis_arg,
                xc=xc,
                grid_level=grid_level,
                max_steps=max_steps,
                max_force=max_force,
                rms_force=rms_force,
                max_displacement=max_displacement,
                trust_radius_bohr=trust_radius_bohr,
                mode_index=mode_index,
                recalc_hessian_interval=recalc_hessian_interval,
                scf_max_iter=scf_max_iter,
                scf_conv_energy=scf_conv_energy,
                scf_conv_density=scf_conv_density,
            )
        except getattr(rust_core, "ScfNotConvergedError", RuntimeError) as e:
            walltime = time.time() - t0
            return Result(
                job_id=request.job_id,
                project_id=request.project_id,
                backend="native",
                backend_version=f"rust-{getattr(rust_core, '__version__', '0.2.0')}",
                task=request.task,
                method=request.method,
                basis=request.basis,
                xc=request.xc or xc,
                molecule=request.molecule,
                status="not_converged",
                started_at=started,
                finished_at=datetime.now(),
                walltime_seconds=walltime,
                scf_converged=False,
                errors=[str(e)],
            )

        walltime = time.time() - t0
        opt_coords_ang = [[c * b2a for c in row] for row in ts_res.geometry_bohr]
        opt_mol = request.molecule.model_copy(update={"coordinates_angstrom": opt_coords_ang})

        actual_tag = f"Rust-PRFO-{method.upper()}({basis_arg})"
        if xc:
            actual_tag = f"{actual_tag}[{xc}]"

        return Result(
            job_id=request.job_id,
            project_id=request.project_id,
            backend="native",
            backend_version=f"rust-{getattr(rust_core, '__version__', '0.2.0')}",
            task=request.task,
            method=request.method,
            actual_method=actual_tag,
            basis=request.basis,
            xc=request.xc or xc,
            molecule=opt_mol,
            status="succeeded" if ts_res.converged else "not_converged",
            started_at=started,
            finished_at=datetime.now(),
            walltime_seconds=walltime,
            total_energy_hartree=ts_res.energy_hartree,
            scf_energy_hartree=ts_res.energy_hartree,
            scf_converged=ts_res.converged,
            optimized_coordinates_angstrom=opt_coords_ang,
            gradients=ts_res.gradient,
            is_transition_state=True,
            imaginary_frequency_cm=ts_res.imaginary_frequency_cm,
            extra={
                "steps": ts_res.steps,
                "transition_vector": ts_res.transition_vector,
            },
        )

    def irc(self, request: CalculationRequest) -> Result:
        """纯 Rust Gonzalez-Schlegel 内禀反应坐标 (IRC) 路径追踪。"""
        started = datetime.now()
        t0 = time.time()
        opts = request.options or {}

        basis = request.basis.strip().lower()
        basis_arg = _RUST_BASIS_ALIASES.get(basis)
        if basis_arg is None:
            raise UnsupportedMethodError(f"Basis '{basis}' is not supported by native IRC engine")

        try:
            from .. import _core as rust_core
        except ImportError as e:
            raise RuntimeError(f"Rust core not available for IRC: {e}")

        b2a = 0.529177210903
        symbols = list(request.molecule.symbols)
        coords_bohr = [[c / b2a for c in row] for row in request.molecule.coordinates_angstrom]

        transition_vector = opts.get("transition_vector")
        if transition_vector is None:
            # If transition_vector not provided, run a 1-step TS search to obtain initial Hessian transition vector
            ts_res = rust_core.run_ts_search(
                symbols,
                coords_bohr,
                charge=request.molecule.charge,
                multiplicity=request.molecule.multiplicity,
                method=request.method.lower(),
                basis=basis_arg,
                xc=request.xc or opts.get("xc"),
                max_steps=5,
            )
            transition_vector = ts_res.transition_vector

        step_size_bohr = float(opts.get("step_size_bohr", opts.get("step_size", 0.1)))
        max_points = int(opts.get("max_points", 20))
        direction = str(opts.get("direction", "both"))
        convergence_tol = float(opts.get("convergence_tol", 1e-3))
        max_gs_iter = int(opts.get("max_gs_iter", 10))
        grid_level = str(opts.get("grid_level", opts.get("grid", "medium")))
        scf_max_iter = int(opts.get("scf_max_iter", opts.get("max_iter", 100)))
        scf_conv_energy = float(opts.get("scf_conv_energy", opts.get("conv_energy", 1e-11)))
        scf_conv_density = float(opts.get("scf_conv_density", opts.get("conv_density", 1e-10)))

        xc = request.xc or opts.get("xc")
        method = request.method.lower()

        try:
            irc_res = rust_core.run_irc(
                symbols,
                coords_bohr,
                transition_vector=transition_vector,
                charge=request.molecule.charge,
                multiplicity=request.molecule.multiplicity,
                method=method,
                basis=basis_arg,
                xc=xc,
                grid_level=grid_level,
                step_size_bohr=step_size_bohr,
                max_points=max_points,
                direction=direction,
                convergence_tol=convergence_tol,
                max_gs_iter=max_gs_iter,
                scf_max_iter=scf_max_iter,
                scf_conv_energy=scf_conv_energy,
                scf_conv_density=scf_conv_density,
            )
        except getattr(rust_core, "ScfNotConvergedError", RuntimeError) as e:
            walltime = time.time() - t0
            return Result(
                job_id=request.job_id,
                project_id=request.project_id,
                backend="native",
                backend_version=f"rust-{getattr(rust_core, '__version__', '0.2.0')}",
                task=request.task,
                method=request.method,
                basis=request.basis,
                xc=request.xc or xc,
                molecule=request.molecule,
                status="not_converged",
                started_at=started,
                finished_at=datetime.now(),
                walltime_seconds=walltime,
                scf_converged=False,
                errors=[str(e)],
            )

        walltime = time.time() - t0

        from ..core.result import IrcPointInfo

        path_points: list[IrcPointInfo] = []
        ts_pt = irc_res.transition_state
        path_points.append(
            IrcPointInfo(
                point_index=ts_pt.point_index,
                direction="ts",
                path_coordinate_s=ts_pt.path_coordinate_s,
                energy_hartree=ts_pt.energy_hartree,
                delta_energy_kcal_mol=ts_pt.delta_energy_kcal_mol,
                gradient_norm=ts_pt.gradient_norm,
                geometry_bohr=[list(row) for row in ts_pt.geometry_bohr],
            )
        )

        fwd_barrier = None
        rev_barrier = None
        reaction_energy = None

        if irc_res.forward is not None:
            fwd_barrier = irc_res.forward.barrier_kcal_mol
            for p in irc_res.forward.points:
                path_points.append(
                    IrcPointInfo(
                        point_index=p.point_index,
                        direction="forward",
                        path_coordinate_s=p.path_coordinate_s,
                        energy_hartree=p.energy_hartree,
                        delta_energy_kcal_mol=p.delta_energy_kcal_mol,
                        gradient_norm=p.gradient_norm,
                        geometry_bohr=[list(row) for row in p.geometry_bohr],
                    )
                )

        if irc_res.reverse is not None:
            rev_barrier = irc_res.reverse.barrier_kcal_mol
            for p in irc_res.reverse.points:
                path_points.append(
                    IrcPointInfo(
                        point_index=p.point_index,
                        direction="reverse",
                        path_coordinate_s=p.path_coordinate_s,
                        energy_hartree=p.energy_hartree,
                        delta_energy_kcal_mol=p.delta_energy_kcal_mol,
                        gradient_norm=p.gradient_norm,
                        geometry_bohr=[list(row) for row in p.geometry_bohr],
                    )
                )

        if irc_res.forward is not None and irc_res.reverse is not None:
            reaction_energy = rev_barrier - fwd_barrier

        actual_tag = f"Rust-IRC-GS-{method.upper()}({basis_arg})"
        if xc:
            actual_tag = f"{actual_tag}[{xc}]"

        return Result(
            job_id=request.job_id,
            project_id=request.project_id,
            backend="native",
            backend_version=f"rust-{getattr(rust_core, '__version__', '0.2.0')}",
            task=request.task,
            method=request.method,
            actual_method=actual_tag,
            basis=request.basis,
            xc=request.xc or xc,
            molecule=request.molecule,
            status="succeeded",
            started_at=started,
            finished_at=datetime.now(),
            walltime_seconds=walltime,
            total_energy_hartree=ts_pt.energy_hartree,
            scf_energy_hartree=ts_pt.energy_hartree,
            scf_converged=True,
            is_transition_state=True,
            irc_forward_barrier_kcal_mol=fwd_barrier,
            irc_reverse_barrier_kcal_mol=rev_barrier,
            reaction_energy_kcal_mol=reaction_energy,
            irc_path=path_points,
            extra={
                "total_points": irc_res.total_points,
                "has_forward": irc_res.forward is not None,
                "has_reverse": irc_res.reverse is not None,
            },
        )

    def solvation(self, request: CalculationRequest) -> Result:
        """纯 Rust 连续介质溶剂化能与自洽反应场计算 (PCM / COSMO / SMD / SCRF)。"""
        from .. import _core as rust_core
        started = datetime.now()
        t0 = time.time()
        symbols = list(request.molecule.symbols)
        coords = [list(r) for r in request.molecule.coordinates_bohr]
        charge = request.molecule.charge
        multiplicity = request.molecule.multiplicity

        basis_raw = request.basis.strip().lower()
        basis_arg = _RUST_BASIS_ALIASES.get(basis_raw, basis_raw)

        method = request.method.lower()
        supported, xc_canon, _, reject_reason = _resolve_rust_scf_spec(
            method, request.xc, request.molecule.multiplicity, request.options
        )
        if not supported:
            raise UnsupportedMethodError(f"Solvation not supported for method '{request.method}': {reject_reason}")

        solvent = request.solvent or request.options.get("solvent", "water")
        solvent_model = request.solvent_model or request.options.get("solvent_model", "cpcm")
        scaling_factor = float(request.options.get("scaling_factor", 1.20))
        include_cds = bool(request.options.get("include_cds", True))
        max_iter = int(request.options.get("max_iter", request.options.get("scf_max_iter", 100)))
        conv_energy = float(request.options.get("conv_energy", request.options.get("scf_conv_energy", 1e-8)))
        conv_density = float(request.options.get("conv_density", request.options.get("scf_conv_density", 1e-6)))

        res = rust_core.run_solvation(
            symbols=symbols,
            coords=coords,
            basis_name=basis_arg,
            solvent=solvent,
            solvent_model=solvent_model,
            charge=charge,
            multiplicity=multiplicity,
            method=method,
            xc=xc_canon,
            scaling_factor=scaling_factor,
            include_cds=include_cds,
            max_iter=max_iter,
            conv_energy=conv_energy,
            conv_density=conv_density,
        )

        walltime = time.time() - t0
        actual_tag = f"Rust-SCRF-{method.upper()}({basis_arg})/{res.solvent_name.capitalize()}"

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
            status="succeeded" if res.converged else "not_converged",
            started_at=started,
            finished_at=datetime.now(),
            walltime_seconds=walltime,
            total_energy_hartree=res.solvated_energy,
            scf_energy_hartree=res.gas_energy,
            scf_converged=res.converged,
            scf_iterations=res.scf_iterations,
            solvated_energy_hartree=res.solvated_energy,
            delta_g_solv_kcal_mol=res.delta_g_solv_kcal_mol,
            delta_g_elec_kcal_mol=res.delta_g_elec_kcal_mol,
            delta_g_cds_kcal_mol=res.delta_g_cds_kcal_mol,
            cavity_surface_area_angstrom2=res.cavity_surface_area_angstrom2,
            solvent_name=res.solvent_name,
            solvent_dielectric=res.dielectric_constant,
            errors=[] if res.converged else ["SCRF not converged"],
            extra={
                "total_surface_charge": res.total_surface_charge,
                "n_tesserae": len(res.tessera_coords_bohr),
            },
        )
