"""后端能力描述。每个后端在初始化时上报自己支持什么。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .request import TaskType


@dataclass
class BackendCapabilities:
    """描述一个计算后端的能力边界。"""

    name: str
    version: Optional[str] = None
    executable_path: Optional[str] = None

    # 支持的任务类型
    tasks: List[TaskType] = field(default_factory=lambda: ["single_point"])

    # 支持的方法（rhf, uhf, b3lyp, wb97xd, mp2, ccsd(t) 等）
    methods: List[str] = field(default_factory=list)

    # 支持的基组
    basis_sets: List[str] = field(default_factory=list)

    # 是否需要外部可执行程序（False = in-process）
    requires_external_program: bool = False

    # 后端是否可用（已检测到可执行文件 / 已安装）
    available: bool = False

    # 不可用原因
    unavailable_reason: Optional[str] = None

    # 是否支持解析输出文件
    can_parse: bool = True

    def supports_task(self, task: str) -> bool:
        return task in self.tasks

    def supports_method(self, method: str) -> bool:
        return method.lower() in [m.lower() for m in self.methods]

    def supports_basis(self, basis: str) -> bool:
        return basis.lower() in [b.lower() for b in self.basis_sets]
