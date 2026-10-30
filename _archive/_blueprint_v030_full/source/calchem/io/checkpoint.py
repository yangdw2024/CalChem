from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class Checkpoint:
    def __init__(self, filename: str, mode: str = "r"):
        self.filename = filename
        self.mode = mode
        self._file = None

    def open(self) -> None:
        import h5py

        if self.mode == "r" and not Path(self.filename).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {self.filename}")

        self._file = h5py.File(self.filename, self.mode)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "Checkpoint":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def save(self, key: str, value: Any) -> None:
        if self._file is None:
            raise RuntimeError("Checkpoint file is not open")

        if isinstance(value, np.ndarray):
            if key in self._file:
                del self._file[key]
            self._file.create_dataset(key, data=value)
        elif isinstance(value, (int, float, str, bool)):
            if key in self._file:
                del self._file[key]
            self._file[key] = value
        elif isinstance(value, dict):
            if key in self._file:
                del self._file[key]
            group = self._file.create_group(key)
            for k, v in value.items():
                self._save_to_group(group, k, v)
        else:
            raise ValueError(f"Unsupported type for checkpoint: {type(value)}")

    def _save_to_group(self, group, key: str, value: Any) -> None:
        if isinstance(value, np.ndarray):
            group.create_dataset(key, data=value)
        elif isinstance(value, (int, float, str, bool)):
            group[key] = value
        elif isinstance(value, dict):
            sub_group = group.create_group(key)
            for k, v in value.items():
                self._save_to_group(sub_group, k, v)

    def load(self, key: str) -> Any:
        if self._file is None:
            raise RuntimeError("Checkpoint file is not open")

        if key not in self._file:
            raise KeyError(f"Key not found in checkpoint: {key}")

        item = self._file[key]
        import h5py

        if isinstance(item, h5py.Group):
            return self._load_group(item)
        elif isinstance(item, h5py.Dataset):
            value = item[()]
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return value
        else:
            return item[()]

    def _load_group(self, group) -> Dict[str, Any]:
        import h5py

        result = {}
        for key in group:
            item = group[key]
            if isinstance(item, h5py.Group):
                result[key] = self._load_group(item)
            elif isinstance(item, h5py.Dataset):
                value = item[()]
                if isinstance(value, bytes):
                    result[key] = value.decode("utf-8")
                else:
                    result[key] = value
        return result

    def keys(self) -> list:
        if self._file is None:
            raise RuntimeError("Checkpoint file is not open")
        return list(self._file.keys())

    def __contains__(self, key: str) -> bool:
        if self._file is None:
            raise RuntimeError("Checkpoint file is not open")
        return key in self._file

    @classmethod
    def save_data(cls, filename: str, key: str, value: Any) -> None:
        mode = "a" if Path(filename).exists() else "w"
        with cls(filename, mode) as chk:
            chk.save(key, value)

    @classmethod
    def load_data(cls, filename: str, key: str) -> Any:
        with cls(filename, "r") as chk:
            return chk.load(key)
