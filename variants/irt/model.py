"""IRT-style cold-item predictor.

Prediction combines known-subject ability, predicted item easiness from text,
benchmark/condition shifts, and conservative adaptive-label calibration.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+./-]*")

with (ROOT / "irt_artifact.json").open("r", encoding="utf-8") as f:
    A = json.load(f)

DIM = int(A["dim"])
GLOBAL_LOGIT = float(A["global_logit"])
ITEM_W = A["item_ease_weights"]
SUBJECT = A.get("subject_effects", {})
SUBJECT_ALIAS = A.get("subject_alias_effects", {})
BENCH = A.get("benchmark_effects", {})
COND = A.get("condition_effects", {})
SB = A.get("subject_benchmark_effects", {})
BENCH_ALIAS = A.get("benchmark_alias_to_id", {})

_SUBJECT_CACHE = {}
_ROUND_CACHE = {}


def _norm(x) -> str:
    return " ".join(str(x or "").lower().split())


def _compact(x) -> str:
    return re.sub(r"[^a-z0-9]+", "", _norm(x))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float) -> float:
    p = min(0.995, max(0.005, p))
    return math.log(p / (1.0 - p))


def _hash(feature: str) -> int:
    digest = hashlib.blake2b(feature.encode("utf-8", "ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % DIM


def _tokens(text: str, limit: int = 650) -> list[str]:
    return TOKEN_RE.findall(_norm(text)[:4500])[:limit]


def _features(benchmark: str, condition: str, item: str) -> dict[int, float]:
    b = _norm(benchmark)
    c = _norm(condition) or "none"
    toks = _tokens(item)
    feats = {}

    def add(name: str, value: float = 1.0) -> None:
        i = _hash(name)
        feats[i] = feats.get(i, 0.0) + value

    add("bias")
    add(f"benchmark={b}")
    add(f"bench_compact={_compact(b)}")
    add(f"condition={c}")
    add(f"bench_condition={b}|{c}")
    add(f"len_bucket={min(12, len(toks)//40)}")
    for tok in toks:
        add(f"tok={tok}")
    for a, b2 in zip(toks, toks[1:]):
        add(f"bi={a}_{b2}", 0.7)
    scale = math.sqrt(sum(v * v for v in feats.values())) or 1.0
    return {i: v / scale for i, v in feats.items()}


def _bench_keys(benchmark: str) -> list[str]:
    b = _norm(benchmark)
    c = _compact(benchmark)
    keys = [b, c]
    mapped = BENCH_ALIAS.get(b) or BENCH_ALIAS.get(c)
    if mapped:
        keys.extend([_norm(mapped), _compact(mapped)])
    out = []
    for key in keys:
        if key and key not in out:
            out.append(key)
    return out


def _subject_match(subject: str) -> tuple[float, str]:
    s = _norm(subject)
    cached = _SUBJECT_CACHE.get(s)
    if cached is not None:
        return cached
    for table in (SUBJECT, SUBJECT_ALIAS):
        if s in table:
            out = (float(table[s]), s)
            _SUBJECT_CACHE[s] = out
            return out
    sc = _compact(s)
    if sc in SUBJECT_ALIAS:
        out = (float(SUBJECT_ALIAS[sc]), sc)
        _SUBJECT_CACHE[s] = out
        return out
    best_len = 0
    best_key = s
    best = 0.0
    for key, val in SUBJECT_ALIAS.items():
        if len(key) <= best_len or len(key) < 5:
            continue
        if key in s or key in sc:
            best_len = len(key)
            best_key = key
            best = float(val)
    out = (best, best_key)
    _SUBJECT_CACHE[s] = out
    return out


def _lookup_bench(benchmark: str) -> tuple[float, str]:
    keys = _bench_keys(benchmark)
    for key in keys:
        if key in BENCH:
            return float(BENCH[key]), key
    return 0.0, keys[0] if keys else ""


def _item_ease(input: dict) -> float:
    feats = _features(input.get("benchmark", ""), input.get("condition", ""), input.get("item_content", ""))
    return sum(float(ITEM_W[i]) * v for i, v in feats.items())


def _sb_effect(subject_key: str, bench_key: str) -> float:
    if not subject_key or not bench_key:
        return 0.0
    for key in (
        f"{subject_key}||{bench_key}",
        f"{_compact(subject_key)}||{bench_key}",
        f"{subject_key}||{_compact(bench_key)}",
        f"{_compact(subject_key)}||{_compact(bench_key)}",
    ):
        if key in SB:
            return float(SB[key])
    return 0.0


def _base_logit(input: dict) -> float:
    subject_effect, subject_key = _subject_match(input.get("subject_content", ""))
    bench_effect, bench_key = _lookup_bench(input.get("benchmark", ""))
    condition = _norm(input.get("condition", "")) or "none"
    z = GLOBAL_LOGIT
    z += 0.72 * subject_effect
    z += 0.42 * _item_ease(input)
    z += 0.10 * bench_effect
    z += 0.03 * float(COND.get(condition, 0.0))
    z += 0.10 * _sb_effect(subject_key, bench_key)
    return z


def _calibration(labeled):
    if not labeled:
        return 0.0, {}
    key = id(labeled)
    if key in _ROUND_CACHE:
        return _ROUND_CACHE[key]
    residuals = []
    by_bench = {}
    for ex in labeled:
        try:
            y = 1.0 if int(ex.get("label", 0)) == 1 else 0.0
            p = _sigmoid(_base_logit(ex))
            r = _logit((y + 0.25) / 1.5) - _logit(p)
            residuals.append(r)
            b = _norm(ex.get("benchmark", ""))
            by_bench.setdefault(b, []).append(r)
        except Exception:
            pass
    if not residuals:
        out = (0.0, {})
    else:
        mean = sum(residuals) / len(residuals)
        global_shift = max(-0.7, min(0.7, mean * min(0.8, len(residuals) / 18.0)))
        bench_shift = {}
        for b, vals in by_bench.items():
            if len(vals) >= 2:
                local = sum(vals) / len(vals)
                bench_shift[b] = max(-0.45, min(0.45, local * min(0.5, len(vals) / 8.0)))
        out = (global_shift, bench_shift)
    _ROUND_CACHE[key] = out
    return out


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    try:
        z = _base_logit(input)
        global_shift, bench_shift = _calibration(labeled)
        z += global_shift
        z += bench_shift.get(_norm(input.get("benchmark", "")), 0.0)
        p = _sigmoid(z)
        p = 0.96 * p + 0.04 * 0.5
        return float(min(0.975, max(0.025, p)))
    except Exception:
        return 0.5
