# src/common/config.py
from __future__ import annotations
import os, random
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
import torch
import yaml

@dataclass
class Cfg:
    raw: Dict[str, Any]

    def __getitem__(self, k):  # cfg["project"]["seed"]
        return self.raw[k]

def load_cfg(path: str = "configs/config.yaml") -> Cfg:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    _set_seed(int(data["project"].get("seed", 42)))
    os.makedirs(data["project"].get("outputs_dir", "./outputs"), exist_ok=True)
    return Cfg(raw=data)

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
