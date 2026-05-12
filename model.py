"""Per-benchmark specialist router over IRT/MLP/complex variants."""

from __future__ import annotations

import importlib.util
import math
import re
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
COMPLEX = _load("complex")


def _norm(x) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x or "").lower())


def _clip(p):
    return min(0.985, max(0.015, float(p)))


def _logit(p):
    p = _clip(p)
    return math.log(p / (1.0 - p))


def _sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _safe(module, input, labeled):
    try:
        return _clip(module.predict(input, labeled=labeled))
    except Exception:
        return 0.5


def _weights(benchmark: str):
    b = _norm(benchmark)
    # Specialist routing: IRT for knowledge/math/coding cold items, MLP for
    # broad/private robustness, complex for tool/preference-style priors.
    if any(k in b for k in ("mmlu", "hle", "ai2d", "mathvista", "matharena")):
        return 0.62, 0.33, 0.05
    if any(k in b for k in ("livecode", "swebench", "cybench")):
        return 0.55, 0.35, 0.10
    if any(k in b for k in ("agentdojo", "bfcl", "androidworld")):
        return 0.42, 0.38, 0.20
    if any(k in b for k in ("rewardbench", "ultrafeedback", "mtbench")):
        return 0.35, 0.50, 0.15
    if "mmbench" in b:
        return 0.50, 0.42, 0.08
    return 0.50, 0.42, 0.08


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    p_irt = _safe(IRT, input, labeled)
    p_mlp = _safe(MLP, input, labeled)
    p_complex = _safe(COMPLEX, input, labeled)
    wi, wm, wc = _weights(input.get("benchmark", ""))
    z = wi * _logit(p_irt) + wm * _logit(p_mlp) + wc * _logit(p_complex)
    p = _sigmoid(z)
    p = 0.98 * p + 0.02 * 0.5
    return float(min(0.975, max(0.025, p)))
