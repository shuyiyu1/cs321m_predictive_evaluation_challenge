from __future__ import annotations

import math
import re

from model import predict

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+./-]*")


def acquisition_function(input: dict) -> float:
    p = predict(input, labeled=None)
    toks = TOKEN_RE.findall(str(input.get("item_content", "")).lower()[:3000])
    uncertainty = 1.0 - min(1.0, abs(p - 0.5) * 2.0)
    length = min(1.0, math.log1p(len(toks)) / math.log(220.0))
    return float(1.5 * uncertainty + 0.3 * length)
