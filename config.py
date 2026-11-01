"""
wowza-v11 config — minimal. V11 is a SEPARATE, data-collection-only sibling of v9.
It reads v9's PUBLIC committed data from GitHub and logs what the market-first engine would
decide. No Telegram, no dashboard, no live model. v9 is never touched.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# v9 (frozen live system) public raw data — the source of truth V11 reads.
V9_RAW_BASE = "https://raw.githubusercontent.com/NevixAA/wowza-betting/main/output"

# Performance counting cutoff (mirrors v9: pre-fix new-format tips excluded from counts).
PERFORMANCE_CUTOFF_DATE = "2026-08-10"

# ── V11 discipline gates (from the external review) ───────────────────────────────
# CLV can't be a hard BET gate until the clean sample is big enough. Below this per-segment
# clean-CLV count, a candidate that clears the economics is PAPER, never BET (solves the
# bootstrap trap: you need CLV history to bet, but can't get it without paper-executing first).
MIN_CLV_N = 150                       # clean CLV observations required per segment before BET
# Odds-integrity bounds (data-validation-FIRST — reject contaminated prices before any decision).
OVERROUND_MIN = 1.00                  # 1/over + 1/under must exceed 1.0 (a real book has vig)
OVERROUND_MAX = 1.25                  # ...and not be absurd (mislabelled / non-goal market leak)

# League classification lives in v9's config/leagues.json, read by src/league_registry.py
# (model_type per league). The two hand-typed league sets that used to sit here, and the
# 'unknown' placeholder model type, were removed in the Data Truth pass (2026-11-01).
