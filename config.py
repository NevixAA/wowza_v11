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
