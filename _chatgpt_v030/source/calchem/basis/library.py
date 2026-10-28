"""
基组库。

提供基组的加载和查询功能。
现在使用内置的 basis_data 模块作为数据源。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from ..utils.config import get_config
from ..utils.exceptions import BasisNotFoundError
from .basis_set import BasisSet
from .basis_data import get_basis_data, list_available_basis


def get_basis(name: str) -> BasisSet:
    """
    获取指定名称的基组。

    Parameters
    ----------
    name : str
        基组名称（不区分大小写）

    Returns
    -------
    BasisSet
        基组对象

    Raises
    ------
    BasisNotFoundError
        如果基组不存在
    """
    name_lower = name.lower()

    available = list_available_basis()
    if name_lower not in [a.lower() for a in available]:
        raise BasisNotFoundError(
            f"Basis set '{name}' not found. "
            f"Available: {', '.join(available)}"
        )

    return BasisSet(name_lower)
