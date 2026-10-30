"""REST 量子化学后端（预留接口）。

对接 REST-agent（基于 Qwen3-8B 的量子化学 agent）。
当前没有真实 REST 服务端点，注册为 unavailable。
"""

from __future__ import annotations

import os

from ..core.calculator import register_backend
from ..core.capabilities import BackendCapabilities
from .external import ExternalProgramBackend


@register_backend("rest")
class RESTBackend(ExternalProgramBackend):
    """REST API 后端。"""

    program_name = "rest"

    def __init__(self, endpoint: str = ""):
        endpoint = endpoint or os.environ.get("CALCHEM_REST_ENDPOINT", "")
        super().__init__(endpoint)
        self.endpoint = endpoint

    def capabilities(self):
        available = bool(self.endpoint)
        return BackendCapabilities(
            name="rest", version="0.1", available=available,
            requires_external_program=False,
            unavailable_reason=None if available else "REST endpoint not configured (set CALCHEM_REST_ENDPOINT)",
            tasks=["single_point"],
            methods=["rhf", "b3lyp"],
            basis_sets=["sto-3g", "def2-svp"],
        )

    def single_point(self, request):
        raise NotImplementedError("REST adapter: implement HTTP client to REST endpoint")
