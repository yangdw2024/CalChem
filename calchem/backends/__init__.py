"""后端适配层。各后端在导入时自动注册。"""

from . import native  # noqa: F401
from . import gaussian  # noqa: F401
from . import orca  # noqa: F401
