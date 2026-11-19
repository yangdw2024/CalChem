"""CalChem 核心数据契约：Request/Result/Calculator/能力/异常/单位。"""

from .calculator import (
    Calculator,
    available_backends,
    get_backend_class,
    list_backend_capabilities,
    register_backend,
)
from .capabilities import BackendCapabilities
from .errors import (
    BackendExecutionError,
    BackendNotFoundError,
    CalChemError,
    ConvergenceError,
    InputValidationError,
    ParserError,
    ResourceLimitError,
    SchemaVersionError,
    UnsupportedBasisError,
    UnsupportedMethodError,
    UnsupportedTaskError,
)
from .request import (
    REQUEST_SCHEMA_VERSION,
    CalculationRequest,
    MoleculeSpec,
    ResourceLimits,
)
from .result import RESULT_SCHEMA_VERSION, JobStatus, OrbitalInfo, Result

__all__ = [
    "Calculator",
    "register_backend",
    "available_backends",
    "get_backend_class",
    "list_backend_capabilities",
    "BackendCapabilities",
    "CalculationRequest",
    "MoleculeSpec",
    "ResourceLimits",
    "Result",
    "OrbitalInfo",
    "JobStatus",
    "REQUEST_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "CalChemError",
    "InputValidationError",
    "UnsupportedMethodError",
    "UnsupportedBasisError",
    "UnsupportedTaskError",
    "BackendNotFoundError",
    "BackendExecutionError",
    "ConvergenceError",
    "ParserError",
    "ResourceLimitError",
    "SchemaVersionError",
]
