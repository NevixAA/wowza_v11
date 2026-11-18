"""v11 writer of the prediction ledger (src/prediction_ledger.py, shared verbatim with v9 and Pro) - LEARNING ONLY.
================================================================================================================

    python -m src.prediction_ledger_v11 settle     # settled view (v11_collect.yml, flag-gated)
    python -m src.prediction_ledger_v11 registry   # output/prediction_ledger/registry.json (v11 rows)
    python -m src.prediction_ledger_v11 validate   # the same contract v9 gates as Data truth 12

OWNER RULE (2026-11-17, reconfirmed 2026-11-18 "make it work"): every model writes every prediction it
makes - every match of every tracked league, every market it predicts - into the prediction ledger. The
v11 market-first shadow is one of those models (contract entry repo 'v11', model_id 'v11_shadow', market
'ou25' in config/prediction_ledger_markets.json).

WHAT v11 PREDICTS. One market: O/U 2.5 full time. For every fixture of v9's board, scripts/v11_shadow.py
decides OVER and UNDER separately (`_decide`), and each side's decision carries v11's own forecast

    p_blend(side) = w * p_model(side) + (1 - w) * p_market(side)

with w <= the segment cap (new_format 0.45 / standard 0.40), shrunk by the measured N_EFF and the side's
odds band. That per-side p_blend is the probability stored here, for EVERY fixture v11 evaluates (not only
its picks); OVER and UNDER are separate decisions, so the two need not sum to 1. A side that cannot be
blended (no measured evidence, no model type, odds rejected, above the longshot cap, one-sided, ...) is
stored unscored with the engine's reason. Nothing is filled or defaulted.

WHAT IS NEVER STORED. The inputs: p_model is v9's p_over25 (already in v9's own ledger) and p_market is
the de-vigged price consensus - a price-derived value, the kind the shared core refuses as a column
(PRICE_LIKE: odds / price / implied / overround / edge / ev / kelly / stake). No odds, no EV, no edge, no
overround - not even inside a not_scored_reason (`clean_reason`). The decision tier / side is not a
probability and is not stored either (the ledger is not a tip list).

THE JOIN. fixture_ref is 'oa:<OddsAPI event id>' from v9's board (`_oa_event_id`), else v9's natural key
'nk:<league_slug>|<date>|<home>|<away>' with the board's own names - exactly the reference v9's team rows
and Pro's rows carry, so all three ledgers join 1:1 on it. predicted_at is the v11 snapshot timestamp,
so a ledger row also joins to output/v11_shadow_snapshots.csv on (snapshot_ts, fixture).

v9 is read, never written. The registry (league slugs, seasons) is v9's config/leagues.json through
src/league_registry (V9_LOCAL in CI). OFF unless PREDICTION_LEDGER_ENABLED == 'true'; nothing here can raise
into the collector.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import prediction_ledger as PL  # noqa: E402

REPO = "v11"
MODEL_ID = "v11_shadow"
MARKET = "ou25"
WRITER = "v11_shadow"
#: The code that turns v9's probability + the price consensus into v11's forecast. Its content hash is
#: the model version: a change to any of these files is a new version (new rows, never an overwrite).
VERSION_FILES = ("src/edge_engine.py", "src/book_consensus.py", "src/evidence.py", "scripts/v11_shadow.py")
#: How long after kickoff a fixture with no result stays PENDING before it is reported NO_RESULT.
NO_RESULT_AFTER_H = 72


def _now() -> datetime:
    return datetime.now(timezone.utc)


def model_version() -> tuple[str, str]:
    """(model_version, model_sha). sha256 over VERSION_FILES with line endings normalised, so a Windows
    checkout (autocrlf) and CI give the same version."""
    h = hashlib.sha256()
    for rel in VERSION_FILES:
        p = ROOT / rel
        h.update(rel.encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n") if p.is_file() else b"<missing>")
    sha = h.hexdigest()[:12]
    return f"{MODEL_ID}_{sha[:8]}", sha


def league_slugs() -> PL.LeagueSlugs:
    """Slugs + season rules from v9's registry, read the way every v11 module reads it."""
    from src import league_registry as LR
    return PL.LeagueSlugs(list(LR._state()["ENTRIES"].values()))


def _hash_values(vals) -> str:
    h = hashlib.sha1()
    for v in vals:
        try:
            f = float(v)
            h.update(b"nan|" if f != f else f"{f:.10g}|".encode())
        except (TypeError, ValueError):
            h.update(f"{v}|".encode("utf-8", "replace"))
    return h.hexdigest()[:12]


def text_id(v) -> str:
    """An id cell as text. A value pandas parsed as a NUMBER is refused (''), never stringified: an OddsAPI
    id like '8e12...' parsed as a float is a different id. The writer then falls back to the natural key."""
    if isinstance(v, str):
        return PL._s(v)
    return ""


_DEC = re.compile(r"[-+]?\d+\.\d+")


def clean_reason(why) -> str:
    """The engine's reason without any number a price could hide in (an overround, an EV, an edge)."""
    s = PL._s(why)
    if not s:
        return ""
    if "overround" in s:
        return "odds rejected: overround outside the sanity bounds"
    s = _DEC.sub("", s)
    return re.sub(r"\(\s*[,\]\)]*\s*\)|\(\s*\)", "", s).strip() or "not scored"


class Recorder:
    """One v11 collector run. `add()` per evaluated fixture, `flush()` once. Never raises into the collector
    (the caller wraps both), and holds nothing unless the flag is on (`recorder()`)."""

    def __init__(self, now: datetime | None = None, *, slugs: PL.LeagueSlugs | None = None,
                 markets: dict | None = None):
        self.now = now or _now()
        self.slugs = slugs or league_slugs()
        self.markets = markets or PL.load_markets()
        e = PL.entry_index(self.markets).get((REPO, MODEL_ID, MARKET))
        if e is None or e.get("status") != "active":
            raise PL.LedgerError(f"{REPO}/{MODEL_ID}/{MARKET} is not an active entry of "
                                 f"config/prediction_ledger_markets.json")
        self.entry = e
        self.version, self.sha = model_version()
        from src.book_consensus import ESTIMATOR
        self.payload = ESTIMATOR
        self.rows: list = []
        self.counts = {"fixtures": 0, "no_kickoff": 0, "not_pre_kickoff": 0, "numeric_oa_id": 0}

    def add(self, *, league, home, away, date, kickoff_utc, oa_event_id, probs: dict, reasons: dict,
            inputs=()) -> None:
        """One board fixture. probs/reasons are keyed by selection (OVER / UNDER); a None probability is an
        unscored row carrying its (cleaned) reason."""
        self.counts["fixtures"] += 1
        ko = PL.utc_ts(kickoff_utc)
        if ko is None:
            self.counts["no_kickoff"] += 1            # cannot be proven pre-kickoff: never written
            return
        now = pd.Timestamp(self.now)
        mtk = (ko - now).total_seconds() / 60.0
        due = PL.snapshots_due(mtk)
        if not due:
            self.counts["not_pre_kickoff"] += 1
            return
        if oa_event_id is not None and not isinstance(oa_event_id, str) and not pd.isna(oa_event_id):
            self.counts["numeric_oa_id"] += 1
        oa = text_id(oa_event_id)
        lentry = self.slugs.entry(league)
        d = PL._s(date)[:10] or ko.strftime("%Y-%m-%d")
        fref = PL.fixture_ref(None, oa, (lentry["id"] if lentry else "", d, home, away))
        fhash = _hash_values(inputs)
        for sel in self.entry["selections"]:
            p = probs.get(sel)
            ok = p is not None and not (isinstance(p, float) and p != p)
            why = "" if ok else (clean_reason(reasons.get(sel)) or "v11 produced no blend for this side")
            for snap in due:
                self.rows.append(PL.make_row(
                    repo=REPO, model_id=MODEL_ID, model_version=self.version, model_sha=self.sha,
                    payload_version=self.payload, feature_hash=fhash, track=self.entry.get("track"),
                    market=MARKET, line=self.entry.get("line"), selection=sel, fixture_ref=fref,
                    fixture_id="", oddsapi_event_id=oa, league=self.slugs.canonical(league) or PL._s(league),
                    season=PL.season_of(lentry, ko), kickoff_utc=PL.iso(ko), home_team=home, away_team=away,
                    probability=float(p) if ok else None, snapshot=snap, predicted_at=PL.iso(now),
                    minutes_to_kickoff=round(mtk, 1), not_scored_reason=why,
                    git_sha=(os.getenv("GITHUB_SHA") or "")[:12], run_id=os.getenv("GITHUB_RUN_ID", "")))

    def flush(self, root: Path | None = None) -> dict:
        st = PL.append(self.rows, repo=REPO, writer=WRITER, run_id=os.getenv("GITHUB_RUN_ID", ""), root=root,
                       slugs=self.slugs, markets=self.markets, now=self.now)
        st.update({f"run_{k}": v for k, v in self.counts.items()})
        self.rows = []
        return st


def recorder(now: datetime | None = None) -> Recorder | None:
    """The run's recorder, or None when the flag is off or it cannot load (never raises)."""
    if not PL.enabled():
        return None
    try:
        return Recorder(now=now)
    except Exception as ex:                                            # noqa: BLE001
        print(f"[prediction_ledger] recorder unavailable ({type(ex).__name__}: {ex}); nothing recorded")
        return None


# ── SETTLE ───────────────────────────────────────────────────────────────────────────────────────

def _fixture_key(league, home, away, ko) -> tuple:
    return (PL._s(league), PL._s(home), PL._s(away), ko.strftime("%Y-%m-%d") if ko is not None else "")


def graded_scores(raw: pd.DataFrame, results: pd.DataFrame | None = None) -> dict:
    """{fixture key: (home_goals, away_goals, source)} for the raw rows' kicked-off fixtures, through v11's
    own results module (football-data.co.uk / v9's af_history; league-scoped, +/-1 day, ambiguous names
    refused). `results` is injectable for tests; None fetches."""
    from src import results as rs
    if raw.empty:
        return {}
    fx = raw[["league", "home_team", "away_team", "kickoff_utc"]].drop_duplicates().copy()
    ko = pd.to_datetime(fx["kickoff_utc"], errors="coerce", utc=True)
    fx = fx[ko.notna() & (ko <= pd.Timestamp(_now()))]
    if fx.empty:
        return {}
    ko = pd.to_datetime(fx["kickoff_utc"], utc=True)
    fx = fx.assign(match=fx["home_team"] + " vs " + fx["away_team"], date=ko.dt.strftime("%Y-%m-%d"))
    if results is None:
        results = rs.fetch_results(sorted(fx["league"].unique()))
    g = rs.grade_fixtures(fx[["league", "match", "date", "home_team", "away_team", "kickoff_utc"]]
                          .reset_index(drop=True), results)
    out = {}
    for r in g.to_dict("records"):
        if pd.notna(r.get("home_goals")) and pd.notna(r.get("away_goals")):
            out[_fixture_key(r["league"], r["home_team"], r["away_team"], PL.utc_ts(r["kickoff_utc"]))] = (
                int(r["home_goals"]), int(r["away_goals"]), PL._s(r.get("grade_source")))
    return out


def outcome_fn(scores: dict, *, now: datetime | None = None, markets: dict | None = None):
    idx = PL.entry_index(markets)
    now_ts = pd.Timestamp(now or _now())
    stamp = PL.iso(now_ts)

    def outcome_of(r: dict) -> dict:
        e = idx.get((r["repo"], r["model_id"], r["market"]))
        ko = PL.utc_ts(r.get("kickoff_utc"))
        if e is None:
            return {"outcome_status": "UNREGISTERED"}
        if ko is None or ko > now_ts:
            return {"outcome_status": "PENDING"}
        m = scores.get(_fixture_key(r["league"], r["home_team"], r["away_team"], ko))
        if m is None:
            if (now_ts - ko).total_seconds() / 3600 >= NO_RESULT_AFTER_H:
                return {"outcome_status": "NO_RESULT",
                        "outcome_value": "no result from v11's results sources (football-data / v9 af_history)"}
            return {"outcome_status": "PENDING"}
        y, val = PL.match_outcome(e["settle"], r["selection"], r.get("line"), m[0], m[1])
        if y is None:
            return {"outcome_status": "NO_RESULT", "outcome_value": val}
        return {"outcome_status": "SETTLED", "y": y, "home_goals": m[0], "away_goals": m[1],
                "outcome_value": val, "result_source": m[2], "settled_at": stamp}
    return outcome_of


def _raw(root: Path | None = None) -> pd.DataFrame:
    root = Path(root or PL.LEDGER_DIR)
    parts = [PL.read_partition(p) for p in PL.partitions(root) if PL.parse_partition(root, p)["repo"] == REPO]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=PL.COLUMNS)


def settle(*, root: Path | None = None, settled_root: Path | None = None, now: datetime | None = None,
           results: pd.DataFrame | None = None, force: bool = False) -> dict:
    if not (force or PL.enabled()):
        return {"skipped": f"{PL.FLAG_ENV} is not 'true'"}
    scores = graded_scores(_raw(root), results)
    return PL.write_settled(outcome_fn(scores, now=now), repo=REPO, root=root, settled_root=settled_root)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="v11 prediction ledger (learning only)")
    ap.add_argument("cmd", choices=["settle", "registry", "validate"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--root", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root) if a.root else None
    slugs = league_slugs()
    if a.cmd == "validate":
        issues = PL.validate(root, slugs=slugs)
        for i in issues:
            print(f"PREDICTION LEDGER VIOLATION: {i}")
        print(f"prediction ledger: {len(PL.partitions(root))} partition(s), {len(issues)} violation(s)")
        return 1 if issues else 0
    if not (a.force or PL.enabled()):
        print(f"[prediction_ledger] {PL.FLAG_ENV} is not 'true' - nothing written")
        return 0
    if a.cmd == "settle":
        print(f"[prediction_ledger] settle: {settle(root=root, force=True)}")
    else:
        reg = PL.write_registry(root=root, slugs=slugs, repos=(REPO,))
        print(f"[prediction_ledger] registry: {len(reg['models'])} model x market entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
