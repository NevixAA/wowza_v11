"""
Canonical league registry for v11 (R2 findings 28 + 29, 2026-11-01).
=====================================================================
v11's result mapping had DRIFTED from v9: it was a hand copy of v9 `FOOTBALL_DATA_LEAGUES`, so the
two API-Football-only new-format leagues (Saudi Pro League, K-League 1 -- football-data.co.uk
publishes neither) had no result source at all and could never be graded, and the old
"Austria Bundesliga" spelling sat in the table as a SECOND KEY for one league instead of as an
alias. The NF season filter was the literal ("2026", "2027").

One table now says, per canonical league: model type, where its results come from, its aliases,
and its season calendar. Everything else in v11 derives from it:

    STD_RESULT_CODES   football-data mmz4281 code          (standard leagues)
    NEW_RESULT_CODES   football-data /new/ code            (new-format leagues it publishes)
    AF_RESULT_LEAGUES  v9's committed af_history.parquet   (API-Football-only leagues)

scripts/v11_tests.py checks it for EQUALITY against v9 (config.py STANDARD/NEW_FORMAT_LEAGUES and
FOOTBALL_DATA_LEAGUES), so the next drift fails a test instead of silently leaving leagues
ungraded.

SEASONS. `src/league_season_calendar.json` is a byte copy of Pro's
`registry/league_season_calendar.json` (season-calendar/1.0). TODO(R2 registry): switch both to
v9 `config/leagues.json` once branch r2/registry-season-failclosed lands.
"""
from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd

CALENDAR_PATH = Path(__file__).resolve().parent / "league_season_calendar.json"
REGISTRY_VERSION = "v11-league-registry/1.0"

# canonical -> (model_type, result_source, code)
LEAGUES: dict[str, tuple[str, str, str | None]] = {
    "Championship": ("standard", "fd_std", "E1"),
    "League One": ("standard", "fd_std", "E2"),
    "League Two": ("standard", "fd_std", "E3"),
    "National League": ("standard", "fd_std", "EC"),
    "Bundesliga 2": ("standard", "fd_std", "D2"),
    "Ligue 2": ("standard", "fd_std", "F2"),
    "La Liga 2": ("standard", "fd_std", "SP2"),
    "Serie B": ("standard", "fd_std", "I2"),
    "Greek Super League": ("standard", "fd_std", "G1"),
    "Portuguese Primeira Liga": ("standard", "fd_std", "P1"),
    "Scottish Premiership": ("standard", "fd_std", "SC0"),
    "Scottish Championship": ("standard", "fd_std", "SC1"),
    "Scottish League One": ("standard", "fd_std", "SC2"),
    "Scottish League Two": ("standard", "fd_std", "SC3"),
    "Belgian First Division A": ("standard", "fd_std", "B1"),
    "Dutch Eredivisie": ("standard", "fd_std", "N1"),
    "Turkish Super Lig": ("standard", "fd_std", "T1"),
    "Denmark Superliga": ("new_format", "fd_new", "DNK"),
    "Austrian Bundesliga": ("new_format", "fd_new", "AUT"),
    "Sweden Allsvenskan": ("new_format", "fd_new", "SWE"),
    "Norway Eliteserien": ("new_format", "fd_new", "NOR"),
    "Finland Veikkausliiga": ("new_format", "fd_new", "FIN"),
    "Ireland Premier Division": ("new_format", "fd_new", "IRL"),
    "Argentina Primera Division": ("new_format", "fd_new", "ARG"),
    "Brazil Serie A": ("new_format", "fd_new", "BRA"),
    "Japan J-League": ("new_format", "fd_new", "JPN"),
    "Mexico Liga MX": ("new_format", "fd_new", "MEX"),
    "China Super League": ("new_format", "fd_new", "CHN"),
    "USA MLS": ("new_format", "fd_new", "USA"),
    "Romanian Superliga": ("new_format", "fd_new", "ROM"),
    # API-Football-only: football-data.co.uk has no file. Results come from v9's PUBLIC committed
    # output/af_history.parquet (read, never written -- v11 invariant 4).
    "Saudi Pro League": ("new_format", "v9_af_history", None),
    "K-League 1": ("new_format", "v9_af_history", None),
}

# Alias -> canonical. An alias is NEVER also a key of LEAGUES.
ALIASES: dict[str, str] = {
    "Austria Bundesliga": "Austrian Bundesliga",
    "Danish Superliga": "Denmark Superliga",
}

STD_RESULT_CODES = {lg: c for lg, (_, src, c) in LEAGUES.items() if src == "fd_std"}
NEW_RESULT_CODES = {lg: c for lg, (_, src, c) in LEAGUES.items() if src == "fd_new"}
AF_RESULT_LEAGUES = tuple(lg for lg, (_, src, _c) in LEAGUES.items() if src == "v9_af_history")


def canonical_league(name) -> str | None:
    """Exact canonical name or a registered alias; anything else is None (never guessed)."""
    lg = str(name).strip()
    if lg in LEAGUES:
        return lg
    return ALIASES.get(lg)


def model_type(name) -> str:
    lg = canonical_league(name)
    return LEAGUES[lg][0] if lg else "unknown"


# ── seasons ───────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=2)
def _calendar(path: str) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for lg, e in d["leagues"].items():
        out[lg] = {"base": sorted([r for r in e["rules"] if "to" not in r], key=lambda r: r["from"]),
                   "special": [r for r in e["rules"] if "to" in r]}
    return {"version": d["version"], "leagues": out}


def calendar_version() -> str:
    return _calendar(str(CALENDAR_PATH))["version"]


def season_for(league, when) -> dict | None:
    """{'type', 'start_year', 'fd_label'} for this league on this date, or None (unknown).

    fd_label is football-data.co.uk's `Season` value in its /new/ files: 'YYYY' for a calendar
    campaign, 'YYYY/YYYY' for a winter one (e.g. Austria '2025/2026', Japan '2026/2027').
    """
    lg = canonical_league(league)
    cal = _calendar(str(CALENDAR_PATH))["leagues"]
    if lg is None or lg not in cal:
        return None
    d = pd.Timestamp(when).date() if not isinstance(when, date) else when
    iso = d.isoformat()
    for r in cal[lg]["special"]:
        if r["from"] <= iso <= r["to"]:
            sy = int(r["label"][:4])
            lab = r["label"]
            fd = lab if "/" not in lab else f"{sy}/{sy + 1}"
            return {"type": "special", "start_year": sy, "fd_label": fd}
    rule = None
    for r in cal[lg]["base"]:
        if r["from"] <= iso:
            rule = r
    if rule is None:
        return None
    if rule["type"] == "calendar":
        return {"type": "calendar", "start_year": d.year, "fd_label": str(d.year)}
    sy = d.year if d.month >= 7 else d.year - 1
    return {"type": "winter", "start_year": sy, "fd_label": f"{sy}/{sy + 1}"}


def nf_result_seasons(league, today=None) -> tuple[str, ...]:
    """football-data `Season` labels to keep for a new-format league: the CURRENT campaign and the
    one before it (a fixture settled just after a rollover still needs its result). Derived from
    the league's season type and the date -- never hardcoded. Unknown league -> () (keep nothing).
    """
    t = pd.Timestamp(today) if today is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    cur = season_for(league, t)
    if cur is None:
        return ()
    labels = [cur["fd_label"]]
    # The previous campaign: step back to just before the current one began.
    probe = t
    for _ in range(800):                                 # at most ~2 years back, day by day
        probe = probe - pd.Timedelta(days=1)
        prev = season_for(league, probe)
        if prev is None:
            break
        if prev["fd_label"] != cur["fd_label"]:
            labels.append(prev["fd_label"])
            break
    return tuple(labels)
