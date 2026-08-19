"""
Measured N_EFF and ECE, replacing the hardcoded placeholders.
============================================================
Prompt 3 sections 11 and 12. v11 has carried two constants since it was written:

    N_EFF = 1000     # "effective sample size"
    ECE   = 0.02     # "calibration error"

Neither was measured. Both feed decisions directly, and they push in opposite directions, so
being wrong about them is not neutral:

  * N_EFF sets the model's blend weight. blend_weight() shrinks the weight when
    n_eff < the band's min_n (500/800/1500). A flat 1000 therefore claims MORE evidence than
    exists in a thin segment — so a segment with 40 settled bets was blended as if it had 1000,
    which is precisely the "unconstrained tiny-sample correction" section 18 forbids.
  * ECE enters the uncertainty lower bound directly: p_lb = p_blend - (ece + z*se + ...). A flat
    0.02 understates uncertainty for a badly-calibrated segment, so EV_lb comes out too high
    and a bet clears a floor it should not.

Both are now derived from settled outcomes, with a documented conservative fallback when a
segment has too little history to say anything. The hierarchy is section 11's:

    model+market+league  ->  model+market  ->  model  ->  global fallback

and shrinkage is applied by sample size rather than a hard minimum, so there is no cliff.

Deliberately conservative in both directions: when evidence is thin, N_EFF goes DOWN (less
model weight) and ECE goes UP (wider uncertainty). Being unsure should cost the model
confidence, not grant it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

# With no evidence at all, assume very little and be pessimistic about calibration. 1000/0.02
# was optimistic on both counts.
FALLBACK_N_EFF = 100
FALLBACK_ECE = 0.08

# Pseudo-count for shrinking a segment's measured ECE toward the global one.
TAU = 200.0

# Below this, a segment is not allowed to speak for itself at all.
MIN_ROWS_FOR_SEGMENT = 30


def expected_calibration_error(y, p, bins: int = 10) -> float:
    """Population-weighted mean |predicted - observed| across probability bins."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    keep = ~(np.isnan(y) | np.isnan(p))
    y, p = y[keep], p[keep]
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            total += (m.sum() / len(y)) * abs(p[m].mean() - y[m].mean())
    return float(total)


@dataclass
class Evidence:
    scope: str            # "global" | "model_type" | "model_type|league"
    n: int
    n_eff: int
    ece: float
    brier: float | None = None
    logloss: float | None = None
    source: str = "measured"      # measured | shrunk | fallback

    def as_row(self) -> dict:
        return asdict(self)


class EvidenceStore:
    """Measured evidence per segment, with a lookup that degrades gracefully."""

    def __init__(self) -> None:
        self.global_ev: Evidence = Evidence("global", 0, FALLBACK_N_EFF, FALLBACK_ECE,
                                            source="fallback")
        self.segments: dict[str, Evidence] = {}

    # ── fitting ──────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame, *, y_col: str, p_col: str,
            model_type_col: str = "model_type", league_col: str = "league") -> "EvidenceStore":
        d = df.copy()
        d["_y"] = pd.to_numeric(d[y_col], errors="coerce")
        d["_p"] = pd.to_numeric(d[p_col], errors="coerce")
        d = d[d["_y"].notna() & d["_p"].notna()]
        if d.empty:
            return self

        g_ece = expected_calibration_error(d["_y"], d["_p"])
        self.global_ev = Evidence("global", len(d), int(len(d)),
                                  round(float(g_ece), 6), source="measured")

        def _add(key: str, sub: pd.DataFrame) -> None:
            n = len(sub)
            if n < MIN_ROWS_FOR_SEGMENT:
                return
            raw = expected_calibration_error(sub["_y"], sub["_p"])
            if np.isnan(raw):
                return
            # Shrink the segment's ECE toward global by sample size, then take the WORSE of the
            # two. A segment that looks better calibrated than global on 40 rows has probably
            # just been lucky, and believing it would narrow the uncertainty band on the
            # thinnest evidence.
            w = n / (n + TAU)
            shrunk = w * raw + (1 - w) * self.global_ev.ece
            self.segments[key] = Evidence(
                key, n, n_eff=n, ece=round(float(max(shrunk, self.global_ev.ece)), 6),
                source="measured" if w >= 0.5 else "shrunk")

        if model_type_col in d.columns:
            for mt, sub in d.groupby(d[model_type_col].astype(str)):
                _add(mt, sub)
                if league_col in d.columns:
                    for lg, s2 in sub.groupby(sub[league_col].astype(str)):
                        _add(f"{mt}|{lg}", s2)
        return self

    # ── lookup ───────────────────────────────────────────────────────────────
    def lookup(self, model_type: str | None = None,
               league: str | None = None) -> Evidence:
        """Most specific available, per section 11's hierarchy. Never raises."""
        if model_type and league:
            hit = self.segments.get(f"{model_type}|{league}")
            if hit:
                return hit
        if model_type:
            hit = self.segments.get(str(model_type))
            if hit:
                return hit
        return self.global_ev

    # ── persistence ──────────────────────────────────────────────────────────
    def to_json(self, path: Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "global": asdict(self.global_ev),
            "fallback": {"n_eff": FALLBACK_N_EFF, "ece": FALLBACK_ECE, "tau": TAU,
                         "min_rows_for_segment": MIN_ROWS_FOR_SEGMENT},
            "segments": {k: asdict(v) for k, v in self.segments.items()},
        }, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "EvidenceStore":
        s = cls()
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            return s                       # no evidence file yet -> conservative fallback
        if d.get("global"):
            s.global_ev = Evidence(**d["global"])
        for k, v in (d.get("segments") or {}).items():
            s.segments[k] = Evidence(**v)
        return s

    def table(self) -> pd.DataFrame:
        rows = [self.global_ev.as_row()] + [v.as_row() for v in self.segments.values()]
        return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
