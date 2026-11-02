"""
League registry for v11 -- READ from v9's config/leagues.json, never typed here.
================================================================================
v11's result mapping used to be a hand table in this file (canonical name -> model type,
football-data code, aliases) plus a byte copy of Pro's season calendar. Both duplicated v9's
canonical registry and could drift: the hand table once lacked Saudi Pro League and K-League 1, and
kept "Austria Bundesliga" as a second key instead of an alias. Data Truth 2026-11-01: there is now
ONE source, wowza-betting `config/leagues.json`, read from a v9 checkout exactly as Pro does:

    V9_LOCAL (CI checks wowza-betting out and points this at it)
    -> a sibling ../v9 or ../wowza-betting clone
    -> the same file over public raw HTTP

If none of them is readable this raises RegistryUnavailable. There is no built-in fallback table:
grading against a guessed league list is worse than not grading.

Derived per canonical league that has a model (model_type standard / new_format):

    STD_RESULT_CODES   football-data mmz4281 code          (standard leagues)
    NEW_RESULT_CODES   football-data /new/ code            (new-format leagues it publishes)
    AF_RESULT_LEAGUES  v9's committed af_history.parquet   (API-Football-only leagues)

SEASONS come from each league's `season_rule` (start_year / calendar_year / end_year / fixed /
eras), with the league's own `start_month` -- never an assumed July rollover.

Module attributes (LEAGUES, ALIASES, ...) are loaded lazily on first use, so importing this module
never needs the network; only reading a league fact does.
"""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
REGISTRY_REL = "config/leagues.json"
RAW_URL = f"https://raw.githubusercontent.com/NevixAA/wowza-betting/main/{REGISTRY_REL}"
MODEL_TYPES = ("standard", "new_format")


class RegistryUnavailable(RuntimeError):
    """v9's config/leagues.json could not be read from any source."""


_STATE: dict = {}


def _candidates() -> list[Path]:
    out = []
    env = os.getenv("V9_LOCAL", "")
    if env:
        out.append(Path(env) / REGISTRY_REL)
    out += [PROJ.parent / "v9" / REGISTRY_REL, PROJ.parent / "wowza-betting" / REGISTRY_REL]
    return out


def _read_registry() -> tuple[dict, str]:
    for p in _candidates():
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8")), str(p)
    try:
        import requests
        r = requests.get(RAW_URL, timeout=30)
        if r.status_code == 200:
            return r.json(), RAW_URL
        why = f"HTTP {r.status_code}"
    except Exception as e:                                              # noqa: BLE001
        why = f"{type(e).__name__}: {e}"
    raise RegistryUnavailable(
        f"cannot read v9 {REGISTRY_REL} (tried {[str(p) for p in _candidates()]} and {RAW_URL}: "
        f"{why}). Set V9_LOCAL to a wowza-betting checkout.")


def _build(reg: dict, source: str) -> dict:
    leagues: dict[str, tuple[str, str | None, str | None]] = {}
    aliases: dict[str, str] = {}
    entries: dict[str, dict] = {}
    for e in reg["leagues"]:
        name = e["name"]
        entries[name] = e
        for a in e.get("aliases") or []:
            aliases[a] = name
        mt = e.get("model_type")
        if mt not in MODEL_TYPES:
            continue                                   # no model -> v11 never shadows it
        code = e.get("football_data_code") or None
        if code and not e.get("api_football_only"):
            src = "fd_std" if mt == "standard" else "fd_new"
        elif e.get("api_football_only"):
            # football-data.co.uk has no file; results come from v9's PUBLIC committed
            # output/af_history.parquet (read, never written -- v11 invariant 4).
            src, code = "v9_af_history", None
        else:
            src = None                                 # no known result source: stays ungraded
        leagues[name] = (mt, src, code)
    return {"source": source, "version": str(reg.get("registry_version", "")),
            "LEAGUES": leagues, "ALIASES": aliases, "ENTRIES": entries,
            "STD_RESULT_CODES": {lg: c for lg, (_, s, c) in leagues.items() if s == "fd_std"},
            "NEW_RESULT_CODES": {lg: c for lg, (_, s, c) in leagues.items() if s == "fd_new"},
            "AF_RESULT_LEAGUES": tuple(lg for lg, (_, s, _c) in leagues.items()
                                       if s == "v9_af_history")}


def _state() -> dict:
    if not _STATE:
        reg, src = _read_registry()
        _STATE.update(_build(reg, src))
    return _STATE


def reset_cache() -> None:
    """Forget the loaded registry (tests point V9_LOCAL elsewhere and reload)."""
    _STATE.clear()


def __getattr__(name: str):
    # PEP 562: these are read from the registry on first access, not typed at import time.
    if name in ("LEAGUES", "ALIASES", "STD_RESULT_CODES", "NEW_RESULT_CODES", "AF_RESULT_LEAGUES"):
        return _state()[name]
    if name == "REGISTRY_VERSION":
        return _state()["version"]
    raise AttributeError(name)


def registry_source() -> str:
    return _state()["source"]


def canonical_league(name) -> str | None:
    """Exact canonical name or a registered alias (after trimming whitespace); anything else is
    None. Same rule as v9 src/league_registry.canonical_name: no case folding, no fuzzy match."""
    if name is None or (isinstance(name, float) and name != name):
        return None
    lg = str(name).strip()
    if not lg:
        return None
    st = _state()
    if lg in st["ENTRIES"]:
        return lg
    return st["ALIASES"].get(lg)


def model_type(name) -> str | None:
    """'standard' | 'new_format' from the registry, or None (unknown league, or no model).
    Never a placeholder such as 'unknown'."""
    lg = canonical_league(name)
    leagues = _state()["LEAGUES"]
    return leagues[lg][0] if lg in leagues else None


def is_paper(name) -> bool | None:
    """v9's registry `paper` flag (e.g. USA MLS, owner 2026-10-06/07: "predicted and collected,
    never sent, counted or bet"). None for a league the registry does not know."""
    lg = canonical_league(name)
    if lg is None:
        return None
    return bool(_state()["ENTRIES"][lg].get("paper"))


def calendar_version() -> str:
    return _state()["version"]


# ── seasons ─────────────────────────────────────────────────────────────────────────────────
def _rule_for(rule: dict, d: date) -> dict | None:
    if "eras" in rule:
        for er in rule["eras"]:
            lo = date.fromisoformat(er["from"]) if er.get("from") else date.min
            hi = date.fromisoformat(er["until"]) if er.get("until") else date.max
            if lo <= d < hi:
                return {k: v for k, v in er.items() if k not in ("from", "until")}
        return None
    return rule


def season_for(league, when) -> dict | None:
    """{'type', 'start_year', 'fd_label'} for this league on this date, or None (unknown).

    fd_label is football-data.co.uk's `Season` value in its /new/ files: 'YYYY' for a calendar
    campaign, 'YYYY/YYYY' for a split one (e.g. Austria '2025/2026', Japan '2026/2027'). The
    rollover month is the league's own `start_month` from the registry.
    """
    lg = canonical_league(league)
    if lg is None:
        return None
    rule = _state()["ENTRIES"][lg].get("season_rule")
    if not isinstance(rule, dict):
        return None
    try:
        ts = pd.Timestamp(when)
    except (TypeError, ValueError):
        return None
    if pd.isna(ts):
        return None
    d = ts.date()
    r = _rule_for(rule, d)
    if r is None:
        return None
    t = r.get("type")
    if t == "calendar_year":
        return {"type": "calendar", "start_year": d.year, "fd_label": str(d.year)}
    if t == "start_year":
        sy = d.year if d.month >= int(r["start_month"]) else d.year - 1
        return {"type": "winter", "start_year": sy, "fd_label": f"{sy}/{sy + 1}"}
    if t == "end_year":
        ey = d.year + 1 if d.month >= int(r["start_month"]) else d.year
        return {"type": "winter", "start_year": ey - 1, "fd_label": f"{ey - 1}/{ey}"}
    # 'fixed' (a one-off campaign such as Brazil 2020 or Argentina's Copa Maradona): the registry
    # holds the API season, not football-data's label for it, so the label is unknown -> None.
    return None


def nf_result_seasons(league, today=None) -> tuple[str, ...]:
    """football-data `Season` labels to keep for a new-format league: the CURRENT campaign and the
    one before it (a fixture settled just after a rollover still needs its result). Derived from
    the league's season rule and the date -- never hardcoded. Unknown league -> () (keep nothing).
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
