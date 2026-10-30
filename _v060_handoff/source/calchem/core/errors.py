"""CalChem 核心异常定义。"""

from __future__ import annotations


class CalChemError(Exception):
    """所有 CalChem 异常的基类。"""


class InputValidationError(CalChemError):
    """输入验证失败（分子、请求参数等）。"""


class UnsupportedMethodError(CalChemError):
    """后端不支持该计算方法。"""


class UnsupportedBasisError(CalChemError):
    """后端不支持该基组。"""


class UnsupportedTaskError(CalChemError):
    """后端不支持该任务类型（单点/优化/频率）。"""


class BackendNotFoundError(CalChemError):
    """请求的后端未注册或不可用。"""


class BackendExecutionError(CalChemError):
    """后端执行失败（subprocess 错误、超时、崩溃等）。"""


class ConvergenceError(CalChemError):
    """SCF 或后处理方法未收敛。"""


class ParserError(CalChemError):
    """后端输出解析失败。原始日志应保留可追溯。"""


class ResourceLimitError(CalChemError):
    """资源限制（内存/核数/超时）超限。"""


class SchemaVersionError(CalChemError):
    """Result/Request 的 schema 版本不兼容。"""
