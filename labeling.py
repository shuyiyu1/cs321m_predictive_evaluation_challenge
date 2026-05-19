"""Acquisition for the IRT+MLP ensemble."""

from __future__ import annotations

import math
import re

from model import predict


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+./-]*")


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(" ".join(str(text or "").lower().split())[:3000])


def acquisition_function(input: dict) -> float:
    p = predict(input, labeled=None)
    uncertainty = 1.0 - min(1.0, abs(p - 0.5) * 2.0)
    length = min(1.0, math.log1p(len(_tokens(input.get("item_content", "")))) / math.log(220.0))
    condition_bonus = 0.05 if str(input.get("condition", "")).lower() not in {"", "none"} else 0.0
    return float(1.5 * uncertainty + 0.35 * length + condition_bonus)
