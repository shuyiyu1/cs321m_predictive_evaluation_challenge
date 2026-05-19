"""Adaptive-label acquisition function.

Scores favor long, information-rich item prompts from under-sampled text
regions using only deterministic standard-library features.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

try:
    from model import predict as _predict
except Exception:
    _predict = None


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+./-]*")
_SEEN_BUCKETS = Counter()


def _norm(value) -> str:
    return " ".join(str(value or "").lower().split())


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(_norm(text)[:4000])[:500]


def _bucket(input: dict) -> str:
    text = "\n".join(
        [
            _norm(input.get("benchmark", "")),
            _norm(input.get("condition", "")),
            " ".join(_tokens(input.get("item_content", ""))[:120]),
        ]
    )
    return hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=2).hexdigest()


def acquisition_function(input: dict) -> float:
    toks = _tokens(input.get("item_content", ""))
    bucket = _bucket(input)
    _SEEN_BUCKETS[bucket] += 1

    length_score = min(1.0, math.log1p(len(toks)) / math.log(220.0))
    diversity_score = 1.0 / _SEEN_BUCKETS[bucket]
    uncertainty_score = 0.0
    if _predict is not None:
        try:
            p = _predict(input, labeled=None)
            uncertainty_score = 1.0 - min(1.0, abs(p - 0.5) * 2.0)
        except Exception:
            uncertainty_score = 0.0
    condition_bonus = 0.05 if _norm(input.get("condition", "")) not in {"", "none"} else 0.0
    return float(1.6 * diversity_score + 1.1 * uncertainty_score + 0.5 * length_score + condition_bonus)
