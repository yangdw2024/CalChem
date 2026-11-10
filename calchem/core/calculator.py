"""Calculator 抽象基类 + 显式后端注册表。

所有后端实现此接口。工厂方法 Calculator.backend(name) 通过注册表查找。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Type

from .capabilities import BackendCapabilities
from .errors import BackendNotFoundError, UnsupportedMethodError, UnsupportedTaskError
from .request import CalculationRequest
from .result import Result


class Calculator(ABC):
    """统一计算后端接口。"""

    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """返回此后端的能力描述。"""

    @abstractmethod
    def single_point(self, request: CalculationRequest) -> Result:
        """单点能计算。"""

    def optimize(self, request: CalculationRequest) -> Result:
        """几何优化。默认不支持，子类可覆盖。"""
        raise UnsupportedTaskError(
            f"Backend '{self.capabilities().name}' does not support geometry optimization"
        )

    def frequency(self, request: CalculationRequest) -> Result:
        """频率计算。默认不支持，子类可覆盖。"""
        raise UnsupportedTaskError(
            f"Backend '{self.capabilities().name}' does not support frequency calculation"
        )

    def tddft(self, request: CalculationRequest) -> Result:
        """激发态与紫外-可见吸收光谱计算 (TD-DFT / TDA / CIS)。默认不支持，子类可覆盖。"""
        raise UnsupportedTaskError(
            f"Backend '{self.capabilities().name}' does not support TD-DFT / CIS excited states"
        )

    def ts_search(self, request: CalculationRequest) -> Result:
        """过渡态搜索 (TS)。默认不支持，子类可覆盖。"""
        raise UnsupportedTaskError(
            f"Backend '{self.capabilities().name}' does not support transition state search"
        )

    def irc(self, request: CalculationRequest) -> Result:
        """内禀反应坐标 (IRC) 路径追踪。默认不支持，子类可覆盖。"""
        raise UnsupportedTaskError(
            f"Backend '{self.capabilities().name}' does not support intrinsic reaction coordinate (IRC)"
        )

    def solvation(self, request: CalculationRequest) -> Result:
        """连续介质溶剂化能计算 (PCM / COSMO / SMD / SCRF)。默认不支持，子类可覆盖。"""
        raise UnsupportedTaskError(
            f"Backend '{self.capabilities().name}' does not support continuum solvation"
        )

    def run(self, request: CalculationRequest) -> Result:
        """根据 request.task 自动分发到对应方法。"""
        caps = self.capabilities()
        if not caps.available:
            raise BackendNotFoundError(
                f"Backend '{caps.name}' not available: {caps.unavailable_reason}"
            )
        if not caps.supports_task(request.task):
            raise UnsupportedTaskError(
                f"Backend '{caps.name}' does not support task '{request.task}'"
            )
        if not caps.supports_method(request.method):
            raise UnsupportedMethodError(
                f"Backend '{caps.name}' does not support method '{request.method}'"
            )
        if not caps.supports_basis(request.basis):
            from .errors import UnsupportedBasisError
            raise UnsupportedBasisError(
                f"Backend '{caps.name}' does not support basis '{request.basis}'"
            )

        if request.task == "single_point":
            return self.single_point(request)
        elif request.task == "optimize":
            return self.optimize(request)
        elif request.task == "frequency":
            return self.frequency(request)
        elif request.task in ("tddft", "cis", "excited_states"):
            return self.tddft(request)
        elif request.task == "ts":
            return self.ts_search(request)
        elif request.task == "irc":
            return self.irc(request)
        elif request.task == "solvation":
            return self.solvation(request)
        else:
            raise UnsupportedTaskError(f"Unknown task: {request.task}")

    # ---- 便捷 facade ----
    def energy(self, molecule, method: str, basis: str, **kwargs) -> float:
        """便捷方法：算单点能并返回 float。"""
        from .request import MoleculeSpec, CalculationRequest as CR

        mol_spec = MoleculeSpec(
            symbols=molecule.symbols,
            coordinates_angstrom=molecule.to_angstrom().coordinates.tolist(),
            charge=molecule.charge,
            multiplicity=molecule.multiplicity,
        )
        req = CR(
            task="single_point",
            molecule=mol_spec,
            backend=self.capabilities().name,
            method=method,
            basis=basis,
            options=kwargs,
        )
        result = self.run(req)
        if result.total_energy_hartree is not None:
            return result.total_energy_hartree
        if result.scf_energy_hartree is not None:
            return result.scf_energy_hartree
        raise RuntimeError(f"No energy in result: {result.errors}")


# ---- 显式注册表 ----
_BACKEND_REGISTRY: Dict[str, Type[Calculator]] = {}


def register_backend(name: str):
    """装饰器：把后端类注册到工厂。"""

    def decorator(cls: Type[Calculator]):
        _BACKEND_REGISTRY[name] = cls
        return cls

    return decorator


def available_backends() -> Dict[str, Type[Calculator]]:
    """返回注册表的副本。"""
    return dict(_BACKEND_REGISTRY)


def get_backend_class(name: str) -> Type[Calculator]:
    if name not in _BACKEND_REGISTRY:
        raise BackendNotFoundError(
            f"Backend '{name}' not registered. "
            f"Available: {list(_BACKEND_REGISTRY.keys())}"
        )
    return _BACKEND_REGISTRY[name]


def list_backend_capabilities() -> list:
    """实例化每个后端并返回其能力表（用于 CLI/GUI 展示）。"""
    results = []
    for name, cls in _BACKEND_REGISTRY.items():
        try:
            instance = cls()
            results.append(instance.capabilities())
        except Exception as e:
            results.append(
                BackendCapabilities(
                    name=name, available=False, unavailable_reason=str(e)
                )
            )
    return results
