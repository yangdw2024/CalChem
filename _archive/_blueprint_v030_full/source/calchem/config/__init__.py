"""CalChem 配置系统。"""

from .models import AppConfig, BackendPaths, ResourceConfig
from .loader import load_config, save_config, auto_detect_paths

__all__ = ["AppConfig", "BackendPaths", "ResourceConfig", "load_config", "save_config", "auto_detect_paths"]
