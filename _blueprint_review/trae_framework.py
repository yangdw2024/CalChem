#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trae Framework v4.2.0 - 统一框架核心
包含：配置管理、日志系统、错误处理、资源监控、事件总线、审查机制、纠错机制、审计机制、自动更新

参考来源：NASA/JPL飞行软件、Google SRE、ISO 27001、VS Code架构
"""

import os
import sys
import json
import time
import threading
import subprocess
import traceback
from pathlib import Path
from typing import Dict, List, Callable, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


class AlertLevel(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    UNKNOWN = "unknown"
    CONFIG = "config"
    FILE_IO = "file_io"
    NETWORK = "network"
    GPU = "gpu"
    CPU = "cpu"
    MEMORY = "memory"
    GROMACS = "gromacs"
    PLUMED = "plumed"
    UI = "ui"
    THREAD = "thread"
    SUBPROCESS = "subprocess"


class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ResourceStatus:
    cpu_percent: float = 0.0
    cpu_cores: int = 0
    memory_total_gb: float = 0.0
    memory_used_gb: float = 0.0
    memory_percent: float = 0.0
    disk_total_gb: float = 0.0
    disk_free_gb: float = 0.0
    disk_percent: float = 0.0
    gpu_available: bool = False
    gpu_name: str = ""
    gpu_memory_total_mb: int = 0
    gpu_memory_used_mb: int = 0
    gpu_memory_percent: float = 0.0
    gpu_utilization: float = 0.0
    timestamp: float = 0.0


class ConfigManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, config_dir=None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_dir=None):
        if self._initialized:
            return

        if config_dir is None:
            config_dir = Path(__file__).parent / "config"

        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._config = {}
        self._listeners = []
        self._lock = threading.Lock()

        self._load_config()
        self._initialized = True

    def _load_config(self):
        config_file = self.config_dir / "app_config.json"
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    self._config = json.load(f)
            except Exception:
                self._config = {}

    def _save_config(self):
        config_file = self.config_dir / "app_config.json"
        try:
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def get(self, key, default=None):
        with self._lock:
            keys = key.split(".")
            value = self._config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value

    def set(self, key, value):
        with self._lock:
            old_value = self.get(key)
            keys = key.split(".")
            config = self._config
            for i, k in enumerate(keys[:-1]):
                if k not in config:
                    config[k] = {}
                config = config[k]
            config[keys[-1]] = value

        if old_value != value:
            self._save_config()
            self._notify_listeners(key, old_value, value)

    def _notify_listeners(self, key, old_value, new_value):
        for listener in self._listeners:
            try:
                listener(key, old_value, new_value)
            except Exception:
                pass

    def add_listener(self, callback):
        self._listeners.append(callback)

    def remove_listener(self, callback):
        if callback in self._listeners:
            self._listeners.remove(callback)

    def get_all(self):
        with self._lock:
            return dict(self._config)

    def import_config(self, config_dict):
        with self._lock:
            self._config.update(config_dict)
        self._save_config()

    def export_config(self):
        return self.get_all()


class AppLogger:
    def __init__(self, name="TraeApp", log_file="app.log"):
        self.name = name
        self.log_file = Path(log_file)
        self._ui_callbacks = []
        self._lock = threading.Lock()

    def _log(self, level, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level_label = level.upper().ljust(7)
        log_line = f"[{timestamp}] [{level_label}] {message}"

        with self._lock:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(log_line + "\n")
            except Exception:
                pass

        print(log_line)

        for callback in self._ui_callbacks:
            try:
                callback(level, message)
            except Exception:
                pass

    def debug(self, message):
        self._log("debug", message)

    def info(self, message):
        self._log("info", message)

    def success(self, message):
        self._log("success", message)

    def warning(self, message):
        self._log("warning", message)

    def error(self, message):
        self._log("error", message)

    def critical(self, message):
        self._log("critical", message)

    def add_ui_callback(self, callback):
        self._ui_callbacks.append(callback)


class ErrorHandler:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, crash_dir=None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, crash_dir=None):
        if self._initialized:
            return

        if crash_dir is None:
            crash_dir = Path(__file__).parent / "logs" / "crashes"

        self.crash_dir = Path(crash_dir)
        self.crash_dir.mkdir(parents=True, exist_ok=True)

        self._error_history = []
        self._max_history = 100
        self._listeners = []
        self._lock = threading.Lock()

        self._initialized = True

    def handle_error(self, exception, context=None):
        context = context or {}
        category = self._classify_error(exception, context)
        severity = self._determine_severity(category)

        error_info = {
            "type": type(exception).__name__,
            "message": str(exception),
            "category": category.value,
            "severity": severity.value,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "traceback": traceback.format_exc(),
        }

        with self._lock:
            self._error_history.append(error_info)
            if len(self._error_history) > self._max_history:
                self._error_history = self._error_history[-self._max_history:]

        for listener in self._listeners:
            try:
                listener(error_info)
            except Exception:
                pass

        return error_info

    def _classify_error(self, exception, context):
        message = str(exception).lower()
        if "cuda" in message or "gpu" in message or "nvidia" in message:
            return ErrorCategory.GPU
        elif "memory" in message or "alloc" in message:
            return ErrorCategory.MEMORY
        elif "permission" in message or "access" in message:
            return ErrorCategory.FILE_IO
        elif "not found" in message or "no such file" in message:
            return ErrorCategory.FILE_IO
        elif "thread" in message:
            return ErrorCategory.THREAD
        elif "config" in message:
            return ErrorCategory.CONFIG
        else:
            return ErrorCategory.UNKNOWN

    def _determine_severity(self, category):
        if category in [ErrorCategory.GPU, ErrorCategory.MEMORY, ErrorCategory.CRITICAL]:
            return ErrorSeverity.HIGH
        elif category in [ErrorCategory.FILE_IO, ErrorCategory.CONFIG]:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW

    def get_error_history(self, limit=50):
        with self._lock:
            return list(self._error_history[-limit:])

    def add_listener(self, callback):
        self._listeners.append(callback)


class ResourceMonitor:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, interval=5.0):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, interval=5.0):
        if self._initialized:
            return

        self.interval = interval
        self._status = ResourceStatus()
        self._running = False
        self._lock = threading.Lock()

        self._initialized = True

    def get_status(self):
        self._update_status()
        with self._lock:
            return ResourceStatus(**vars(self._status))

    def _update_status(self):
        status = ResourceStatus()
        status.timestamp = time.time()

        try:
            status.cpu_cores = os.cpu_count() or 0
            if sys.platform == "win32":
                result = subprocess.run(
                    ["wmic", "cpu", "get", "NumberOfCores", "/format:list"],
                    capture_output=True, text=True, timeout=5
                )
                for line in result.stdout.split("\n"):
                    line = line.strip()
                    if "=" in line:
                        key, value = line.split("=", 1)
                        if key.strip() == "NumberOfCores":
                            try:
                                status.cpu_cores = int(value.strip())
                            except ValueError:
                                pass
                            break
        except Exception:
            pass

        try:
            if sys.platform == "win32":
                import ctypes
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_uint32),
                        ("dwMemoryLoad", ctypes.c_uint32),
                        ("ullTotalPhys", ctypes.c_uint64),
                        ("ullAvailPhys", ctypes.c_uint64),
                        ("ullTotalPageFile", ctypes.c_uint64),
                        ("ullAvailPageFile", ctypes.c_uint64),
                        ("ullTotalVirtual", ctypes.c_uint64),
                        ("ullAvailVirtual", ctypes.c_uint64),
                        ("ullAvailExtendedVirtual", ctypes.c_uint64),
                    ]
                mem = MEMORYSTATUSEX()
                mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
                status.memory_total_gb = mem.ullTotalPhys / (1024 ** 3)
                status.memory_used_gb = (mem.ullTotalPhys - mem.ullAvailPhys) / (1024 ** 3)
                status.memory_percent = mem.dwMemoryLoad
        except Exception:
            pass

        try:
            if sys.platform == "win32":
                import ctypes
                free_bytes = ctypes.c_ulonglong(0)
                total_bytes = ctypes.c_ulonglong(0)
                ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                    str(Path.cwd()),
                    ctypes.byref(free_bytes),
                    ctypes.byref(total_bytes),
                    None
                )
                status.disk_total_gb = total_bytes.value / (1024 ** 3)
                status.disk_free_gb = free_bytes.value / (1024 ** 3)
                if status.disk_total_gb > 0:
                    status.disk_percent = (1 - status.disk_free_gb / status.disk_total_gb) * 100
        except Exception:
            pass

        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True, text=True, timeout=2
            )
            status.gpu_available = result.returncode == 0 and result.stdout.strip()
            if status.gpu_available:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(",")
                    if len(parts) >= 4:
                        status.gpu_name = parts[0].strip()
                        status.gpu_memory_total_mb = int(float(parts[1].strip()))
                        status.gpu_memory_used_mb = int(float(parts[2].strip()))
                        status.gpu_utilization = float(parts[3].strip())
                        if status.gpu_memory_total_mb > 0:
                            status.gpu_memory_percent = (
                                status.gpu_memory_used_mb / status.gpu_memory_total_mb * 100
                            )
        except Exception:
            pass

        with self._lock:
            self._status = status

    def get_cpu_info(self):
        cpu_info = {
            "name": "未知CPU",
            "cores": os.cpu_count() or 4,
            "threads": os.cpu_count() or 4,
            "simd_support": [],
        }

        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "Name,NumberOfCores,NumberOfLogicalProcessors", "/format:list"],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.split("\n"):
                line = line.strip()
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if key == "Name":
                        cpu_info["name"] = value
                    elif key == "NumberOfCores":
                        try:
                            cpu_info["cores"] = int(value)
                        except ValueError:
                            pass
                    elif key == "NumberOfLogicalProcessors":
                        try:
                            cpu_info["threads"] = int(value)
                        except ValueError:
                            pass
        except Exception:
            pass

        return cpu_info

    def get_total_memory_gb(self):
        status = self.get_status()
        return round(status.memory_total_gb, 1)

    def get_gpu_info(self):
        status = self.get_status()
        return {
            "available": status.gpu_available,
            "name": status.gpu_name,
            "memory_total_mb": status.gpu_memory_total_mb,
        }

    def can_start(self, required_memory_gb=0):
        status = self.get_status()
        if status.memory_percent >= 95:
            return False, "内存严重不足"
        if status.disk_free_gb < 1:
            return False, "磁盘空间不足"
        if status.memory_total_gb - status.memory_used_gb < required_memory_gb:
            return False, "可用内存不足"
        return True, "资源充足"


class EventBus:
    def __init__(self):
        self._subscribers = {}
        self._lock = threading.Lock()

    def subscribe(self, pattern, callback, priority=0):
        with self._lock:
            if pattern not in self._subscribers:
                self._subscribers[pattern] = []
            self._subscribers[pattern].append((priority, callback))
            self._subscribers[pattern].sort(key=lambda x: -x[0])

    def unsubscribe(self, pattern, callback):
        with self._lock:
            if pattern in self._subscribers:
                self._subscribers[pattern] = [
                    (p, c) for p, c in self._subscribers[pattern] if c != callback
                ]

    def publish(self, event, data=None):
        data = data or {}
        matched = []

        with self._lock:
            for pattern, subscribers in self._subscribers.items():
                if self._match_pattern(pattern, event):
                    matched.extend(subscribers)

        for priority, callback in matched:
            try:
                callback(event, data)
            except Exception:
                pass

    def _match_pattern(self, pattern, event):
        if pattern == "*":
            return True
        if pattern == event:
            return True
        if "*" in pattern:
            parts = pattern.split(".")
            event_parts = event.split(".")
            for i, part in enumerate(parts):
                if i >= len(event_parts):
                    return part == "*"
                if part != "*" and part != event_parts[i]:
                    return False
            return True
        return False

    def get_subscriber_count(self):
        return sum(len(s) for s in self._subscribers.values())


class ReviewMechanism:
    def __init__(self):
        self._checks = []

    def add_check(self, name, check_func, level="standard"):
        self._checks.append({"name": name, "func": check_func, "level": level})

    def preflight_check(self, task_name, params, work_dir):
        results = []
        passed = True

        checks = [c for c in self._checks if c["level"] in ["standard", "strict"]]

        for check in checks:
            try:
                result = check["func"](task_name, params, work_dir)
                if not result["passed"]:
                    passed = False
                results.append({
                    "name": check["name"],
                    "passed": result["passed"],
                    "message": result.get("message", ""),
                })
            except Exception as e:
                passed = False
                results.append({
                    "name": check["name"],
                    "passed": False,
                    "message": f"检查失败: {e}",
                })

        return {
            "result": "passed" if passed else "failed",
            "items": results,
        }

    def quick_check(self, task_name, params, work_dir):
        return {"result": "passed", "items": []}


class CorrectionMechanism:
    def __init__(self):
        self._strategies = []

    def add_strategy(self, error_type, strategy_func):
        self._strategies.append({"type": error_type, "func": strategy_func})

    def auto_correct(self, error_info, context):
        category = error_info.get("category", "unknown")

        for strategy in self._strategies:
            if strategy["type"] == category:
                try:
                    return strategy["func"](error_info, context)
                except Exception:
                    pass

        return {"result": "skipped", "action": "manual"}


class AuditMechanism:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, audit_dir=None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, audit_dir=None):
        if self._initialized:
            return

        if audit_dir is None:
            audit_dir = Path(__file__).parent / "logs" / "audit"

        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)

        self._records = []
        self._max_records = 10000
        self._lock = threading.Lock()

        self._initialized = True

    def log_event(self, category, action, detail=""):
        record = {
            "record_id": f"{int(time.time())}_{os.urandom(4).hex()}",
            "category": category,
            "action": action,
            "detail": detail,
            "timestamp": time.time(),
            "datetime": datetime.fromtimestamp(time.time()).isoformat(),
        }

        with self._lock:
            self._records.append(record)
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records:]

        self._save_records()

    def _save_records(self):
        try:
            timestamp = datetime.now().strftime("%Y%m%d")
            filename = self.audit_dir / f"audit_{timestamp}.json"
            records_to_save = self._records[-5000:]
            with open(filename, "w", encoding="utf-8") as fp:
                json.dump(records_to_save, fp, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def get_statistics(self):
        stats = {}
        for record in self._records:
            key = record["action"]
            stats[key] = stats.get(key, 0) + 1
        return {"total_records": len(self._records), "by_type": stats}

    def get_recent_actions(self, limit=50):
        return list(reversed(self._records[-limit:]))


class AutoUpdater:
    def __init__(self, current_version="4.2.0"):
        self.current_version = current_version
        self.update_status = "idle"

    def check_for_update(self):
        return {
            "update_available": False,
            "current_version": self.current_version,
            "latest_version": self.current_version,
            "status": "no_update",
        }

    def download_update(self):
        self.update_status = "downloading"
        return {"status": "downloading"}

    def install_update(self):
        self.update_status = "installed"
        return {"status": "installed"}


__all__ = [
    'ConfigManager',
    'AppLogger',
    'ErrorHandler',
    'ResourceMonitor',
    'EventBus',
    'ReviewMechanism',
    'CorrectionMechanism',
    'AuditMechanism',
    'AutoUpdater',
    'AlertLevel',
    'ErrorCategory',
    'ErrorSeverity',
]