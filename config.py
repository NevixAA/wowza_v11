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

# ── League classification (copied verbatim from v9 so tags never diverge) ─────────
STANDARD_FORMAT_LEAGUES = {
    "League One", "League Two",
    "Bundesliga 2", "La Liga 2", "Ligue 2",
    "Championship",
    "Serie B",
    "Greek Super League",
    "National League",
    "Portuguese Primeira Liga",
    "Scottish Championship", "Scottish League One", "Scottish League Two",
    "Belgian First Division A",
    "Dutch Eredivisie",
    "Scottish Premiership",
    "Turkish Super Lig",
}

NEW_FORMAT_LEAGUES = {
    "Denmark Superliga",
    "Austrian Bundesliga",
    "Sweden Allsvenskan",
    "Romanian Superliga",
    "Norway Eliteserien",
    "Finland Veikkausliiga",
    "Ireland Premier Division",
    "Argentina Primera Division",
    "Brazil Serie A",
    "Japan J-League",
    "Mexico Liga MX",
    "China Super League",
    "USA MLS",
    "Saudi Pro League",
    "K-League 1",
}


def model_type_for_league(league) -> str:
    """Canonical model tag: 'standard' | 'new_format' | 'unknown' (identical to v9)."""
    lg = str(league).strip()
    if lg in STANDARD_FORMAT_LEAGUES:
        return "standard"
    if lg in NEW_FORMAT_LEAGUES:
        return "new_format"
    return "unknown"
