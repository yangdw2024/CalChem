"""后端适配层。各后端在 __init__ 中自动注册。"""

from . import native  # noqa: F401  (注册 NativeCalchem)
