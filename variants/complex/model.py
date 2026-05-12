"""Submission model for the Predictive AI Evaluation Challenge.

The predictor is deliberately self-contained: it imports only the Python
standard library and a small JSON artifact fitted offline by train_baseline.py.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


_ARTIFACT_PATH = Path(__file__).with_name("baseline_artifact.json")
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+./-]*")
_MAX_TEXT_CHARS = 5000

with _ARTIFACT_PATH.open("r", encoding="utf-8") as f:
    _A = json.load(f)

_DIM = int(_A["dim"])
_WEIGHTS = _A["weights"]
_GLOBAL_LOGIT = float(_A["global_logit"])
_SUBJECT = _A.get("subject_effects", {})
_SUBJECT_ALIASES = _A.get("subject_alias_effects", {})
_BENCH = _A.get("benchmark_effects", {})
_COND = _A.get("condition_effects", {})
_BENCH_ALIASES = _A.get("benchmark_alias_to_id", {})
_SB = _A.get("subject_benchmark_effects", {})
_SC = _A.get("subject_condition_effects", {})
_BC = _A.get("benchmark_condition_effects", {})
_ITEM_DIM = int(_A.get("item_model_dim", _DIM))
_ITEM_WEIGHTS = _A.get("item_model_weights", [])
_ROUND_CACHE = {}
_SUBJECT_EFFECT_CACHE = {}
_SUBJECT_KEY_CACHE = {}


def _norm(value) -> str:
    return " ".join(str(value or "").lower().split())


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
    return int.from_bytes(digest, "little") % _DIM


def _hash_item(feature: str) -> int:
    digest = hashlib.blake2b(feature.encode("utf-8", "ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % _ITEM_DIM


def _compact(value) -> str:
    return re.sub(r"[^a-z0-9]+", "", _norm(value))


def _tokens(text: str) -> list[str]:
    toks = _TOKEN_RE.findall(_norm(text)[:_MAX_TEXT_CHARS])
    return toks[:700]


def _benchmark_keys(benchmark: str) -> list[str]:
    b = _norm(benchmark)
    c = _compact(benchmark)
    keys = [b, c]
    mapped = _BENCH_ALIASES.get(b) or _BENCH_ALIASES.get(c)
    if mapped:
        keys.extend([_norm(mapped), _compact(mapped)])
    out = []
    for key in keys:
        if key and key not in out:
            out.append(key)
    return out


def _features(input: dict) -> dict[int, float]:
    subject = _norm(input.get("subject_content", ""))
    benchmark = _norm(input.get("benchmark", ""))
    condition = _norm(input.get("condition", "")) or "none"
    item_toks = _tokens(input.get("item_content", ""))
    subject_toks = _tokens(subject)

    feats = {}

    def add(name: str, value: float = 1.0) -> None:
        i = _hash(name)
        feats[i] = feats.get(i, 0.0) + value

    add("bias")
    add(f"subject_exact={subject}")
    add(f"benchmark={benchmark}")
    add(f"condition={condition}")
    add(f"bench_condition={benchmark}|{condition}")
    for tok in subject_toks[:80]:
        add(f"subject_tok={tok}")
    for tok in item_toks:
        add(f"item_tok={tok}")
    for a, b in zip(item_toks, item_toks[1:]):
        add(f"item_bigram={a}_{b}")

    scale = math.sqrt(sum(v * v for v in feats.values())) or 1.0
    return {i: v / scale for i, v in feats.items()}


def _base_logit(input: dict) -> float:
    feats = _features(input)
    z = sum(float(_WEIGHTS[i]) * v for i, v in feats.items())

    subject = _norm(input.get("subject_content", ""))
    benchmark = _norm(input.get("benchmark", ""))
    condition = _norm(input.get("condition", "")) or "none"
    subject_effect, subject_key = _subject_match(subject)
    bench_effect, bench_key = _benchmark_effect(benchmark)
    z = 0.58 * z
    z += 0.18 * subject_effect
    z += 0.08 * bench_effect
    z += 0.02 * float(_COND.get(condition, 0.0))
    z += 0.12 * _interaction_effect(_SB, subject_key, bench_key)
    z += 0.04 * _interaction_effect(_SC, subject_key, condition)
    z += 0.04 * _interaction_effect(_BC, bench_key, condition)
    z += 0.22 * _item_logit(input)
    return z


def _subject_effect(subject: str) -> float:
    return _subject_match(subject)[0]


def _subject_match(subject: str) -> tuple[float, str]:
    cached = _SUBJECT_EFFECT_CACHE.get(subject)
    if cached is not None:
        return cached, _SUBJECT_KEY_CACHE.get(subject, subject)
    direct = _SUBJECT.get(subject)
    if direct is not None:
        out = float(direct)
        _SUBJECT_EFFECT_CACHE[subject] = out
        _SUBJECT_KEY_CACHE[subject] = subject
        return out, subject
    alias = _SUBJECT_ALIASES.get(subject)
    if alias is not None:
        out = float(alias)
        _SUBJECT_EFFECT_CACHE[subject] = out
        _SUBJECT_KEY_CACHE[subject] = subject
        return out, subject

    # Hidden subject descriptions may wrap the model name in prose. A short
    # substring scan over the compact alias table recovers much of the ability
    # prior without needing third-party fuzzy-matching libraries.
    best_len = 0
    best = 0.0
    best_key = subject
    padded = f" {subject} "
    candidates = (subject, _compact(subject))
    for name, effect in _SUBJECT_ALIASES.items():
        if len(name) <= best_len or len(name) < 5:
            continue
        if any(name and (name in cand or f" {name} " in padded) for cand in candidates):
            best_len = len(name)
            best = float(effect)
            best_key = name
    _SUBJECT_EFFECT_CACHE[subject] = best
    _SUBJECT_KEY_CACHE[subject] = best_key
    return best, best_key


def _benchmark_effect(benchmark: str) -> tuple[float, str]:
    for key in _benchmark_keys(benchmark):
        if key in _BENCH:
            return float(_BENCH[key]), key
    return 0.0, _benchmark_keys(benchmark)[0] if _benchmark_keys(benchmark) else ""


def _interaction_effect(table: dict, left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    keys = [f"{left}||{right}", f"{_compact(left)}||{right}", f"{left}||{_compact(right)}", f"{_compact(left)}||{_compact(right)}"]
    for key in keys:
        val = table.get(key)
        if val is not None:
            return float(val)
    return 0.0


def _item_features(input: dict) -> dict[int, float]:
    benchmark = _norm(input.get("benchmark", ""))
    condition = _norm(input.get("condition", "")) or "none"
    toks = _tokens(input.get("item_content", ""))[:500]
    feats = {}

    def add(name: str) -> None:
        i = _hash_item(name)
        feats[i] = feats.get(i, 0.0) + 1.0

    add("bias")
    add(f"benchmark={benchmark}")
    add(f"bench_compact={_compact(benchmark)}")
    add(f"condition={condition}")
    add(f"bench_condition={benchmark}|{condition}")
    for tok in toks:
        add(f"tok={tok}")
    for a, b in zip(toks, toks[1:]):
        add(f"bi={a}_{b}")
    scale = math.sqrt(sum(v * v for v in feats.values())) or 1.0
    return {i: v / scale for i, v in feats.items()}


def _item_logit(input: dict) -> float:
    if not _ITEM_WEIGHTS:
        return 0.0
    return sum(float(_ITEM_WEIGHTS[i]) * v for i, v in _item_features(input).items())


def _calibration(labeled: list[dict] | None) -> tuple[float, dict[str, float], dict[str, float]]:
    """Return global, benchmark, and condition calibration shifts."""
    if not labeled:
        return 0.0, {}, {}
    key = id(labeled)
    cached = _ROUND_CACHE.get(key)
    if cached is not None:
        return cached

    residuals = []
    by_benchmark = {}
    by_condition = {}
    for ex in labeled:
        try:
            y = 1.0 if int(ex.get("label", 0)) == 1 else 0.0
            p = _sigmoid(_base_logit(ex))
            r = _logit((y + 0.2) / 1.4) - _logit(p)
            residuals.append(r)
            b = _norm(ex.get("benchmark", ""))
            c = _norm(ex.get("condition", "")) or "none"
            by_benchmark.setdefault(b, []).append(r)
            by_condition.setdefault(c, []).append(r)
        except Exception:
            continue
    if not residuals:
        out = (0.0, {}, {})
    else:
        # Conservative shrinkage keeps a tiny label budget from overfitting.
        mean = sum(residuals) / len(residuals)
        global_shift = max(-1.0, min(1.0, mean * min(1.0, len(residuals) / 16.0)))
        bench_shift = {}
        for key_, vals in by_benchmark.items():
            if len(vals) >= 2:
                local = sum(vals) / len(vals)
                bench_shift[key_] = max(-0.9, min(0.9, local * min(0.8, len(vals) / 8.0)))
        cond_shift = {}
        for key_, vals in by_condition.items():
            if len(vals) >= 2:
                local = sum(vals) / len(vals)
                cond_shift[key_] = max(-0.4, min(0.4, local * min(0.5, len(vals) / 10.0)))
        out = (global_shift, bench_shift, cond_shift)
    _ROUND_CACHE[key] = out
    return out


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """Return P(subject answers item correctly) as a native float in [0, 1]."""
    try:
        z = _base_logit(input)
        global_shift, bench_shift, cond_shift = _calibration(labeled)
        benchmark = _norm(input.get("benchmark", ""))
        condition = _norm(input.get("condition", "")) or "none"
        z += global_shift
        z += bench_shift.get(benchmark, 0.0)
        z += cond_shift.get(condition, 0.0)
        p = _sigmoid(z)
        # The private slice can differ sharply from public validation; mild
        # shrinkage improves leaderboard robustness when string matches fail.
        p = 0.92 * p + 0.08 * 0.5
        return float(min(0.975, max(0.025, p)))
    except Exception:
        return 0.5
