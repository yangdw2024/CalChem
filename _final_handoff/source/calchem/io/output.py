from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)


@dataclass
class Results:
    method: str = ""
    basis: str = ""
    energy: float = 0.0
    converged: bool = False
    n_cycles: int = 0
    mol: Optional[Dict[str, Any]] = None
    orbital_energies: Optional[List[float]] = None
    orbital_occupations: Optional[List[float]] = None
    dipole_moment: Optional[List[float]] = None
    charges: Optional[List[float]] = None
    frequencies: Optional[List[float]] = None
    thermochemistry: Optional[Dict[str, Any]] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), cls=NumpyEncoder, indent=indent)

    def save(self, filename: str) -> None:
        with open(filename, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_json(cls, json_str: str) -> "Results":
        data = json.loads(json_str)
        return cls(**data)

    @classmethod
    def from_file(cls, filename: str) -> "Results":
        with open(filename, "r") as f:
            data = json.load(f)
        return cls(**data)


def write_json(results: Results, filename: str) -> None:
    results.save(filename)
