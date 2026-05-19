"""Uncertainty-focused acquisition for the IRT-style model."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from model import predict


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+./-]*")
SEEN = Counter()


def _norm(x) -> str:
    return " ".join(str(x or "").lower().split())


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(_norm(text)[:3000])[:450]


def _bucket(input: dict) -> str:
    text = "\n".join(
        [
            _norm(input.get("benchmark", "")),
            _norm(input.get("condition", "")),
            " ".join(_tokens(input.get("item_content", ""))[:100]),
        ]
    )
    return hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=2).hexdigest()


def acquisition_function(input: dict) -> float:
    bucket = _bucket(input)
    SEEN[bucket] += 1
    p = predict(input, labeled=None)
    uncertainty = 1.0 - min(1.0, abs(p - 0.5) * 2.0)
    diversity = 1.0 / SEEN[bucket]
    length = min(1.0, math.log1p(len(_tokens(input.get("item_content", "")))) / math.log(220.0))
    return float(1.4 * uncertainty + 1.0 * diversity + 0.25 * length)
