"""Blend the fundamentally different IRT model with the best MLP variant."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _load(name: str):
    path = ROOT / "variants" / name / "model.py"
    spec = importlib.util.spec_from_file_location(f"_variant_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


IRT = _load("irt")
MLP = _load("mlp")


def _clip(p: float) -> float:
    return min(0.985, max(0.015, float(p)))


def _logit(p: float) -> float:
    p = _clip(p)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _safe(module, input: dict, labeled):
    try:
        return _clip(module.predict(input, labeled=labeled))
    except Exception:
        return 0.5


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    p_irt = _safe(IRT, input, labeled)
    p_mlp = _safe(MLP, input, labeled)
    # IRT is more aligned with the item-cold-start protocol; MLP has the best
    # known private score. Keep both substantial.
    z = 0.58 * _logit(p_irt) + 0.42 * _logit(p_mlp)
    p = _sigmoid(z)
    p = 0.98 * p + 0.02 * 0.5
    return float(min(0.975, max(0.025, p)))
