"""
Prediction ledger (core) - every model x market prediction, stored before kickoff, append-only.
================================================================================================

    python -m src.prediction_ledger --validate            # exit 1 on any violation (Data truth 12)
    python -m src.prediction_ledger --registry            # rewrite output/prediction_ledger/registry.json

OWNER RULE (Nevo, 2026-11-17): "the models need to predict every match from the tracked leagues - not
as a bet (that's the TOP-3 job) ... we need more predictions of the model to learn from."

This is RESEARCH / LEARNING STORAGE ONLY. Nothing here reads or writes a tip, tier, gate, stake,
threshold, Telegram file or any existing output. It stores statistics only: NO bookmaker price is ever
written (prices live in the private store). The writers that feed it are switched on by the repo
variable PREDICTION_LEDGER_ENABLED (default OFF, see `enabled()`).

LAYOUT (one file per repo x model x market x league x season - never a mixed stream):

    output/prediction_ledger/<repo>/<model_id>/<market>/<league_slug>/<season>.csv   raw, append-only
    output/prediction_ledger_settled/<repo>/<model_id>/<market>/<league_slug>/<season>.csv   settled VIEW
    output/prediction_ledger/<repo>/_runs/<writer>/<YYYY-MM>.csv   one line per writer run (attempts counted)
    output/prediction_ledger/registry.json                     every model x market (x league) + counts

`league_slug` is the league's `id` in config/leagues.json (stable, registry-owned). A league the registry
does not know has no slug: its rows are refused (validator: unknown league).

UNIQUE KEY (first write wins, idempotent):
    (model_id, model_version, market, line, selection, fixture_ref, player_id, snapshot)
A re-run that offers an existing key appends NOTHING; the attempt is counted in the run log. A new model
version writes new rows under its own model_version and never touches the old ones.

SNAPSHOTS (a run fills every snapshot that is due and not yet written):
    first_sight   the first run that saw the fixture - the lead at which TOP-3 freezes a candidate
                  (src/research/league_weekly: every fixture x market is frozen at first sight)
    final         the first run inside the final 120 min before kickoff: the final pre-kickoff opinion
                  the run cadence can freeze under first-write-wins (the real lead is minutes_to_kickoff;
                  a fixture no run saw inside the window simply has no `final` row - never back-filled)
Every snapshot is PRE-KICKOFF by construction: a row with predicted_at >= kickoff_utc is never written
and the validator fails on one.

MISSING STAYS MISSING. A fixture/player the model cannot score is stored with scored=0, an empty
probability and `not_scored_reason`. Nothing is filled, defaulted or invented.

THIS FILE IS SHARED VERBATIM WITH NevixAA/wowzaV9-Pro (src/prediction_ledger.py there): it imports
pandas and the standard library only, so one schema and one key serve both repos and the two ledgers
join on `fixture_id` (API-Football fixture id). Change it in both places or not at all
(tests/test_prediction_ledger.py pins SCHEMA_VERSION and the column list).
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEMA_VERSION = "pl/1.0"

ROOT = Path(__file__).resolve().parents[1]
LEDGER_DIR = ROOT / "output" / "prediction_ledger"
SETTLED_DIR = ROOT / "output" / "prediction_ledger_settled"
MARKETS_FILE = ROOT / "config" / "prediction_ledger_markets.json"
LEAGUES_FILE = ROOT / "config" / "leagues.json"
REGISTRY_FILE = LEDGER_DIR / "registry.json"
FLAG_ENV = "PREDICTION_LEDGER_ENABLED"

#: The raw ledger columns, in order. Statistics only: there is deliberately no odds/price column.
COLUMNS = [
    "schema_version", "repo", "model_id", "model_version", "model_sha", "payload_version",
    "feature_hash", "track", "market", "line", "selection",
    "fixture_ref", "fixture_id", "oddsapi_event_id", "league", "league_slug", "season",
    "kickoff_utc", "home_team", "away_team", "player_id", "player_name", "team",
    "probability", "snapshot", "predicted_at", "minutes_to_kickoff",
    "scored", "not_scored_reason", "git_sha", "run_id",
]
KEY_COLS = ["model_id", "model_version", "market", "line", "selection", "fixture_ref", "player_id",
            "snapshot"]
#: Columns a price could hide in. A row carrying any of them is refused (prices are private).
PRICE_LIKE = re.compile(r"(^|_)(odds|price|implied|overround|edge|kelly|stake|ev)(_|$)", re.I)

#: (label, max minutes to kickoff or None = any lead)
SNAPSHOTS = (("first_sight", None), ("final", 120))
SNAPSHOT_LABELS = tuple(s for s, _ in SNAPSHOTS)

_ID = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_SEASON = re.compile(r"^[0-9]{4}(-[0-9]{2,4})?$")


class LedgerError(ValueError):
    """A row or a partition that breaks the ledger contract."""


# ── switches & config ──────────────────────────────────────────────────────────────────────────────

def enabled() -> bool:
    """True only when the repo variable PREDICTION_LEDGER_ENABLED is exactly 'true' (case-insensitive).
    Anything else - unset, empty, '0', 'false', a typo - is OFF."""
    return os.getenv(FLAG_ENV, "").strip().lower() == "true"


def load_markets(path: Path | None = None) -> dict:
    """config/prediction_ledger_markets.json: every (repo, model_id, market) the ledger knows, with its
    status ('active' writes rows, 'pending_model' is registered with no rows ever written)."""
    return json.loads(Path(path or MARKETS_FILE).read_text(encoding="utf-8"))


def entries(markets: dict | None = None, repo: str | None = None, status: str | None = None) -> list:
    out = []
    for e in (markets or load_markets())["entries"]:
        if repo and e["repo"] != repo:
            continue
        if status and e.get("status") != status:
            continue
        out.append(e)
    return out


def entry_index(markets: dict | None = None) -> dict:
    return {(e["repo"], e["model_id"], e["market"]): e for e in (markets or load_markets())["entries"]}


# ── league slug ────────────────────────────────────────────────────────────────────────────────────

def load_leagues(path: Path | None = None) -> list:
    """The league registry (v9 config/leagues.json). Pro has no registry of its own: its workflows check
    v9 out read-only and point PREDICTION_LEDGER_LEAGUES at v9's file (never a copy that can drift)."""
    path = path or os.getenv("PREDICTION_LEDGER_LEAGUES") or LEAGUES_FILE
    return json.loads(Path(path).read_text(encoding="utf-8"))["leagues"]


class LeagueSlugs:
    """League name (canonical or registered alias, exact after trimming) -> registry `id`.
    No fuzzy match: a near-name is unknown, never 'close enough'."""

    def __init__(self, leagues: list | None = None):
        self.leagues = leagues if leagues is not None else load_leagues()
        self._by_name: dict = {}
        self.by_slug: dict = {}
        for e in self.leagues:
            sid = str(e["id"])
            if not _ID.match(sid):
                raise LedgerError(f"registry id {sid!r} is not a valid slug")
            self.by_slug[sid] = e
            for n in [e["name"], *(e.get("aliases") or [])]:
                self._by_name[str(n).strip()] = sid

    def slug(self, league) -> str | None:
        return self._by_name.get("" if league is None else str(league).strip())

    def entry(self, league) -> dict | None:
        s = self.slug(league)
        return self.by_slug.get(s) if s else None

    def canonical(self, league) -> str | None:
        e = self.entry(league)
        return e["name"] if e else None


def _apply_season_rule(rule: dict, d) -> str:
    """The registry's season convention (config/leagues.json season_rule; the same rules as v9
    src/season_resolver.py) -> season id, or '' when no rule covers the date (missing stays missing)."""
    if not isinstance(rule, dict):
        return ""
    if "eras" in rule:
        for er in rule["eras"]:
            lo = pd.Timestamp(er["from"]).date() if er.get("from") else None
            hi = pd.Timestamp(er["until"]).date() if er.get("until") else None
            if (lo is None or lo <= d) and (hi is None or d < hi):
                return _apply_season_rule({k: v for k, v in er.items() if k not in ("from", "until")}, d)
        return ""
    t = rule.get("type")
    if t == "start_year":
        m = int(rule["start_month"])
        return str(d.year if d.month >= m else d.year - 1)
    if t == "calendar_year":
        return str(d.year)
    if t == "end_year":
        m = int(rule["start_month"])
        return str(d.year + 1 if d.month >= m else d.year)
    if t == "fixed":
        return str(rule["season"])
    return ""


def season_of(entry: dict | None, kickoff) -> str:
    """Season id of a fixture from its registry entry and kickoff ('' if unknown)."""
    t = utc_ts(kickoff)
    if entry is None or t is None:
        return ""
    try:
        return _apply_season_rule(entry.get("season_rule"), t.date())
    except Exception:                                                  # noqa: BLE001
        return ""


def slugify(name: str) -> str:
    """The registry's id convention: lower-case, every run of non-alphanumerics -> '_'.
    tests/test_prediction_ledger.py asserts slugify(name) == id for every registry league, so the
    partition slug is stable and derivable from the canonical name."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def tracked(e: dict) -> bool:
    """A league the production models serve (the 'tracked leagues' of the owner rule)."""
    return e.get("production_state") in ("ENABLED", "ENABLED_PROVIDER_UNSUPPORTED", "PAPER")


# ── snapshots ──────────────────────────────────────────────────────────────────────────────────────

def snapshots_due(minutes_to_kickoff: float) -> list:
    """Every snapshot label whose window contains this lead time (pre-kickoff only)."""
    if minutes_to_kickoff is None or not (minutes_to_kickoff > 0):
        return []
    return [lab for lab, mx in SNAPSHOTS if mx is None or minutes_to_kickoff <= mx]


def utc_ts(v) -> pd.Timestamp | None:
    """Timezone-aware UTC timestamp, or None for a missing / unparseable value."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    t = pd.to_datetime(v, errors="coerce", utc=True)
    return None if pd.isna(t) else t


def iso(t: pd.Timestamp | None) -> str:
    return "" if t is None else t.strftime("%Y-%m-%dT%H:%M:%SZ")


# ── paths ──────────────────────────────────────────────────────────────────────────────────────────

def partition_path(root: Path, repo: str, model_id: str, market: str, league_slug: str,
                   season: str) -> Path:
    for what, v in (("repo", repo), ("model_id", model_id), ("market", market), ("league_slug", league_slug)):
        if not _ID.match(str(v)):
            raise LedgerError(f"{what} {v!r} is not a valid path component")
    if not _SEASON.match(str(season)):
        raise LedgerError(f"season {season!r} is not a valid season")
    return Path(root) / repo / model_id / market / league_slug / f"{season}.csv"


def parse_partition(root: Path, path: Path) -> dict | None:
    """The partition fields a file path encodes, or None for a path outside the layout."""
    try:
        parts = Path(path).relative_to(Path(root)).parts
    except ValueError:
        return None
    if len(parts) != 5 or not parts[4].endswith(".csv") or parts[1].startswith("_"):
        return None
    return {"repo": parts[0], "model_id": parts[1], "market": parts[2], "league_slug": parts[3],
            "season": parts[4][:-4]}


def partitions(root: Path | None = None) -> list:
    root = Path(root or LEDGER_DIR)
    if not root.exists():
        return []
    return sorted(p for p in root.glob("*/*/*/*/*.csv") if parse_partition(root, p))


# ── rows ───────────────────────────────────────────────────────────────────────────────────────────

def _s(v) -> str:
    """Canonical text form of a cell for the key ('' for missing)."""
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        if math.isinf(v) or v != int(v) or abs(v) >= 1e15:
            return repr(v)
        return str(int(v))
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none", "<na>", "nat") else s


def key_of(row: dict) -> tuple:
    return tuple(_s(row.get(c)) for c in KEY_COLS)


def fixture_ref(fixture_id=None, oddsapi_event_id=None, natural: tuple | None = None) -> str:
    """The fixture part of the key, from the provider id the writer's source carries:
        'af:<API-Football fixture id>'   props (player_model works on API-Football fixtures)
        'oa:<OddsAPI event id>'          the v9 team board (output/predictions.csv `_oa_event_id`) and
                                         every Pro row (Pro's board IS v9's predictions.csv, so v9 team
                                         and Pro rows join 1:1 on it)
        'nk:<league_slug>|<date>|<home>|<away>'  a board row with no provider id (natural key, never
                                         guessed into an id)
    '' when nothing identifies the fixture (such a row is refused)."""
    f = _s(fixture_id)
    if f:
        try:
            return f"af:{int(float(f))}"
        except ValueError:
            return ""
    o = _s(oddsapi_event_id)
    if o:
        return f"oa:{o}"
    if natural and all(_s(x) for x in natural):
        return "nk:" + "|".join(_s(x) for x in natural)
    return ""


def _line(v) -> str:
    s = _s(v)
    if not s:
        return ""
    try:
        return f"{float(s):g}"
    except ValueError:
        return s


def make_row(**kw) -> dict:
    """One ledger row with every column present (missing -> ''). Unknown columns are an error and a
    price-like column is refused."""
    extra = set(kw) - set(COLUMNS)
    if extra:
        raise LedgerError(f"unknown ledger column(s) {sorted(extra)}")
    row = {c: "" for c in COLUMNS}
    row.update({k: ("" if v is None else v) for k, v in kw.items()})
    row["schema_version"] = SCHEMA_VERSION
    # ids as plain integer text (12345.0 -> '12345'), so the stored key reads back identically
    for c in ("fixture_id", "player_id"):
        s = _s(row[c])
        try:
            row[c] = str(int(float(s))) if s else ""
        except ValueError:
            row[c] = s
    for c in KEY_COLS:
        if c != "line":
            row[c] = _s(row[c])
    row["line"] = _line(row["line"])
    p = row["probability"]
    if _s(p) != "":
        pf = float(p)
        if not (0.0 <= pf <= 1.0):
            raise LedgerError(f"probability {pf} outside [0, 1]")
        row["probability"] = round(pf, 6)
    row["scored"] = 1 if _s(row["probability"]) != "" else 0
    if row["scored"] == 0 and not _s(row["not_scored_reason"]):
        raise LedgerError("an unscored row needs not_scored_reason (missing stays missing, with a reason)")
    if row["scored"] == 1:
        row["not_scored_reason"] = ""
    return row


# ── append (first write wins) ──────────────────────────────────────────────────────────────────────

def _read_keys(path: Path) -> set:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    d = pd.read_csv(path, usecols=KEY_COLS, dtype=str, keep_default_na=False)
    return {tuple(_s(v) for v in r) for r in d[KEY_COLS].itertuples(index=False, name=None)}


def append(rows: list, *, repo: str, writer: str, run_id: str = "", root: Path | None = None,
           slugs: LeagueSlugs | None = None, markets: dict | None = None,
           now: datetime | None = None) -> dict:
    """Append rows to their partitions, first write wins. Returns counts; never rewrites a file.

    Refused before writing (counted, never stored): a (repo, model_id, market) that is not an ACTIVE
    entry of config/prediction_ledger_markets.json, an unknown league, a row without a fixture
    reference, a snapshot written at/after kickoff, a key already stored or offered twice in this
    batch."""
    root = Path(root or LEDGER_DIR)
    slugs = slugs or LeagueSlugs()
    idx = entry_index(markets)
    stats = {"offered": len(rows), "written": 0, "duplicate_skipped": 0, "late_skipped": 0,
             "refused": 0, "refused_reasons": {}, "partitions": 0}
    touched: list = []

    def refuse(why: str):
        stats["refused"] += 1
        stats["refused_reasons"][why] = stats["refused_reasons"].get(why, 0) + 1

    by_part: dict = {}
    for r in rows:
        e = idx.get((repo, r.get("model_id"), r.get("market")))
        if e is None or e.get("status") != "active":
            refuse("model x market not an active ledger entry")
            continue
        slug = slugs.slug(r.get("league"))
        if not slug:
            refuse("league not in the registry")
            continue
        r["league_slug"], r["repo"] = slug, repo
        if not _s(r.get("fixture_ref")):
            refuse("no fixture reference")
            continue
        if not _SEASON.match(_s(r.get("season"))):
            refuse("season unknown")
            continue
        ko, at = utc_ts(r.get("kickoff_utc")), utc_ts(r.get("predicted_at"))
        if ko is None or at is None or at >= ko:
            stats["late_skipped"] += 1
            continue
        p = partition_path(root, repo, r["model_id"], r["market"], slug, r["season"])
        by_part.setdefault(p, []).append(r)

    for p, part in by_part.items():
        seen = _read_keys(p)
        new = []
        for r in part:
            k = key_of(r)
            if k in seen:
                stats["duplicate_skipped"] += 1
                continue
            seen.add(k)
            new.append(r)
        if not new:
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        head = not p.exists() or p.stat().st_size == 0
        with p.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="raise", lineterminator="\n")
            if head:
                w.writeheader()
            for r in new:
                w.writerow({c: r.get(c, "") for c in COLUMNS})
        stats["written"] += len(new)
        stats["partitions"] += 1
        touched.append(p)

    _run_log(root, repo, writer, run_id, stats, now)
    # Data truth 12 on exactly the partitions this call wrote (cheap; the full check runs in the gate).
    # A violation is reported loudly and returned; it never raises into the pipeline that called us.
    try:
        bad = validate(root, slugs=slugs, markets=markets, only=set(touched)) if touched else []
    except Exception as e:                                             # noqa: BLE001
        bad = [f"validator failed: {type(e).__name__}: {e}"]
    for b in bad:
        print(f"::error title=Prediction ledger (Data truth 12)::{b}")
    stats["violations"] = len(bad)
    return stats


RUN_COLS = ["at", "repo", "writer", "run_id", "offered", "written", "duplicate_skipped", "late_skipped",
            "refused", "refused_reasons", "partitions"]


def _run_log(root: Path, repo: str, writer: str, run_id: str, stats: dict, now=None) -> None:
    now = now or datetime.now(timezone.utc)
    # one run log per writer (predict and player_props run concurrently; neither ever touches the other's)
    p = Path(root) / repo / "_runs" / writer / f"{now:%Y-%m}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    head = not p.exists()
    with p.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=RUN_COLS, lineterminator="\n")
        if head:
            w.writeheader()
        w.writerow({"at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "repo": repo, "writer": writer,
                    "run_id": run_id, **{k: stats[k] for k in RUN_COLS[4:] if k != "refused_reasons"},
                    "refused_reasons": json.dumps(stats["refused_reasons"], sort_keys=True)})


# ── validator (Data truth 12) ──────────────────────────────────────────────────────────────────────

def read_partition(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def validate(root: Path | None = None, slugs: LeagueSlugs | None = None,
             markets: dict | None = None, only: set | None = None) -> list:
    """Every violation of the ledger contract, as text. Empty list = clean.

    Fails on: a duplicate key; a row whose repo / model_id / market / league_slug / season does not match
    its partition; an unknown league (or a league whose slug is not its partition's); a model x market not
    registered (or registered 'pending_model' but holding rows); a row written at/after kickoff; a column
    set that is not the schema (a price column included); an unscored row without a reason; a probability
    outside [0, 1]."""
    root = Path(root or LEDGER_DIR)
    slugs = slugs or LeagueSlugs()
    idx = entry_index(markets)
    issues: list = []
    for p in partitions(root):
        if only is not None and p not in only:
            continue
        part = parse_partition(root, p)
        rel = p.relative_to(root).as_posix()
        try:
            d = read_partition(p)
        except Exception as e:                                             # noqa: BLE001
            issues.append(f"{rel}: unreadable ({e})")
            continue
        if list(d.columns) != COLUMNS:
            issues.append(f"{rel}: columns are not schema {SCHEMA_VERSION}")
            if any(PRICE_LIKE.search(c) for c in d.columns):
                issues.append(f"{rel}: carries a price-like column")
            continue
        e = idx.get((part["repo"], part["model_id"], part["market"]))
        if e is None:
            issues.append(f"{rel}: model x market not registered in config/prediction_ledger_markets.json")
        elif e.get("status") != "active" and len(d):
            issues.append(f"{rel}: entry status {e.get('status')!r} but {len(d)} row(s) stored")
        if part["league_slug"] not in slugs.by_slug:
            issues.append(f"{rel}: unknown league slug {part['league_slug']!r}")
        for col in ("repo", "model_id", "market", "league_slug", "season"):
            bad = d[col] != part[col]
            if bad.any():
                issues.append(f"{rel}: {int(bad.sum())} row(s) with {col} != partition {part[col]!r}")
        lg_slug = d["league"].map(slugs.slug)
        if lg_slug.isna().any():
            issues.append(f"{rel}: {int(lg_slug.isna().sum())} row(s) with a league the registry does not know")
        elif (lg_slug != part["league_slug"]).any():
            issues.append(f"{rel}: {int((lg_slug != part['league_slug']).sum())} row(s) whose league is not "
                          f"the partition league")
        k = d[KEY_COLS].apply(lambda s: s.map(_s))
        dup = k.duplicated(keep=False)
        if dup.any():
            issues.append(f"{rel}: {int(k.duplicated().sum())} duplicate key(s)")
        ko = pd.to_datetime(d["kickoff_utc"], errors="coerce", utc=True)
        at = pd.to_datetime(d["predicted_at"], errors="coerce", utc=True)
        late = ko.isna() | at.isna() | (at >= ko)
        if late.any():
            issues.append(f"{rel}: {int(late.sum())} row(s) not strictly before kickoff "
                          f"(pre-kickoff snapshot)")
        if e is not None and isinstance(e.get("selections"), list) and len(d):
            badsel = ~d["selection"].isin([str(x) for x in e["selections"]])
            if badsel.any():
                issues.append(f"{rel}: {int(badsel.sum())} row(s) with a selection the entry does not register")
        badsnap = ~d["snapshot"].isin(SNAPSHOT_LABELS)
        if badsnap.any():
            issues.append(f"{rel}: {int(badsnap.sum())} row(s) with an unknown snapshot label")
        prob = pd.to_numeric(d["probability"].replace("", None), errors="coerce")
        has_p = d["probability"].astype(str).str.strip() != ""
        if (has_p & (prob.isna() | (prob < 0) | (prob > 1))).any():
            issues.append(f"{rel}: probability outside [0, 1]")
        noreason = ~has_p & (d["not_scored_reason"].astype(str).str.strip() == "")
        if noreason.any():
            issues.append(f"{rel}: {int(noreason.sum())} unscored row(s) without not_scored_reason")
        if (d["schema_version"] != SCHEMA_VERSION).any():
            issues.append(f"{rel}: schema_version is not {SCHEMA_VERSION}")
    return issues


# ── settled view ───────────────────────────────────────────────────────────────────────────────────

SETTLED_EXTRA = ["outcome_status", "y", "home_goals", "away_goals", "outcome_value", "log_loss", "brier",
                 "rps", "log_score", "result_source", "settled_at"]
EPS = 1e-15


def log_loss(p: float, y: int) -> float:
    p = min(max(float(p), EPS), 1 - EPS)
    return -math.log(p) if y == 1 else -math.log(1 - p)


def brier(p: float, y: int) -> float:
    return (float(p) - y) ** 2


def rps(probs: list, outcome_index: int) -> float:
    """Ranked probability score of an ordered outcome set (1X2 = home, draw, away)."""
    cum_p = cum_o = 0.0
    tot = 0.0
    for i, p in enumerate(probs[:-1]):
        cum_p += p
        cum_o += 1.0 if i == outcome_index else 0.0
        tot += (cum_p - cum_o) ** 2
    return tot / (len(probs) - 1)


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def stat_outcome(settle: dict, selection: str, line, hs, as_):
    """(y, outcome_value) of a corners / cards selection from the match STATISTIC (home count hs, away
    count as_), or (None, reason) when a count is missing - never inferred from the score.
        total_stat_over   OVER / UNDER on hs + as_ vs line
        team_stat_over    HOME_OVER / HOME_UNDER / AWAY_OVER / AWAY_UNDER on that side's count vs line"""
    rule, stat = settle.get("rule"), settle.get("stat", "stat")
    if _missing(hs) or _missing(as_):
        return None, f"no {stat} statistics"
    if line is None or str(line).strip() == "":
        return None, "no line"
    h, a, ln, sel = int(hs), int(as_), float(line), str(selection)
    if rule == "total_stat_over":
        if sel not in ("OVER", "UNDER"):
            return None, f"selection {sel!r} is not OVER/UNDER"
        over = h + a > ln
        return int(over if sel == "OVER" else not over), f"{stat} {h}-{a}"
    if rule == "team_stat_over":
        side, _, ou = sel.partition("_")
        if side not in ("HOME", "AWAY") or ou not in ("OVER", "UNDER"):
            return None, f"selection {sel!r} is not HOME_/AWAY_ OVER/UNDER"
        over = (h if side == "HOME" else a) > ln
        return int(over if ou == "OVER" else not over), f"{stat} {h}-{a}"
    return None, f"rule {rule!r} is not a statistic rule"


def match_outcome(settle: dict, selection: str, line, hg, ag, hth=None, hta=None, hs=None, as_=None):
    """(y, outcome_value) of a TEAM / specials selection from the score, or (None, reason) when the
    score the rule needs is missing (HT rules need the HT score; never inferred from FT).
    90-minute settlement (owner decision): hg/ag are the regular-time goals.
    Corners / cards rules (total_stat_over, team_stat_over) settle from the match statistic hs / as_
    (stat_outcome), never from the score."""
    rule, period = settle.get("rule"), settle.get("period", "ft")
    if rule in ("total_stat_over", "team_stat_over"):
        return stat_outcome(settle, selection, line, hs, as_)
    h, a = (hth, hta) if period == "ht" else (hg, ag)
    if h is None or a is None or (isinstance(h, float) and math.isnan(h)) or             (isinstance(a, float) and math.isnan(a)):
        return None, f"no {period.upper()} score"
    h, a = int(h), int(a)
    tot, sel = h + a, str(selection)
    if rule == "total_goals_over":
        over = tot > float(line)
        return int(over if sel == "OVER" else not over), f"{h}-{a}"
    if rule == "btts":
        yes = h > 0 and a > 0
        return int(yes if sel == "YES" else not yes), f"{h}-{a}"
    if rule == "result_1x2":
        res = "H" if h > a else "A" if a > h else "D"
        return int(sel == res), res
    if rule == "correct_score":
        real = f"{h}-{a}" if h <= 6 and a <= 6 else "other"
        return int(sel == real), real
    if rule == "total_goals_eq0":
        return int(tot == 0), str(tot)
    if rule == "total_goals_le1":
        return int(tot <= 1), str(tot)
    m = re.fullmatch(r"total_goals_ge(\d+)", str(rule))
    if m:
        return int(tot >= int(m.group(1))), str(tot)
    if rule == "team_goals_ge4":
        g = h if sel == "HOME" else a
        return int(g >= 4), f"{h}-{a}"
    return None, f"rule {rule!r} is not settled from the score"


def settled_view(raw: pd.DataFrame, outcome_of) -> pd.DataFrame:
    """raw rows + one outcome per row, 1:1 (asserted, no fan-out). `outcome_of(row) -> dict` returns
    {'outcome_status': 'SETTLED'|'PENDING'|'VOID'|..., 'y': 0/1 or None, ...}. Scores are filled only on
    SETTLED scored rows. RPS is set on every 1X2 selection row of a fixture whose 3 selections settled."""
    out = raw.copy()
    rec = [outcome_of(r) for r in raw.to_dict("records")]
    assert len(rec) == len(raw), "settled view must be 1:1 with the raw ledger"
    for c in SETTLED_EXTRA:
        out[c] = pd.Series([("" if x.get(c) is None else x.get(c, "")) for x in rec], index=out.index,
                           dtype=object)
    p = pd.to_numeric(out["probability"].replace("", None), errors="coerce")
    y = pd.to_numeric(out["y"].replace("", None), errors="coerce")
    ok = (out["outcome_status"] == "SETTLED") & p.notna() & y.notna()
    out.loc[ok, "log_loss"] = [round(log_loss(a, int(b)), 6) for a, b in zip(p[ok], y[ok])]
    out.loc[ok, "brier"] = [round(brier(a, int(b)), 6) for a, b in zip(p[ok], y[ok])]
    # correct score: the log score of the realised scoreline sits on its (y = 1) row
    mcs = ok & (out["market"] == "correct_score") & (y == 1)
    out.loc[mcs, "log_score"] = [round(-math.log(min(max(float(a), EPS), 1.0)), 6) for a in p[mcs]]
    m1x2 = ok & (out["market"] == "1x2")
    if m1x2.any():
        order = {"H": 0, "D": 1, "A": 2}
        grp = KEY_COLS[:3] + ["fixture_ref", "player_id", "snapshot"]
        for _, g in out[m1x2].groupby(grp):
            if len(g) == 3 and set(g["selection"]) == set(order):
                g = g.sort_values("selection", key=lambda s: s.map(order))
                probs = pd.to_numeric(g["probability"]).tolist()
                hit = [i for i, v in enumerate(pd.to_numeric(g["y"]).tolist()) if v == 1]
                if len(hit) == 1:
                    out.loc[g.index, "rps"] = round(rps(probs, hit[0]), 6)
    assert len(out) == len(raw)
    return out


def settled_path(settled_root: Path, root: Path, raw_path: Path) -> Path:
    return Path(settled_root) / Path(raw_path).relative_to(Path(root))


def write_settled(outcome_of, *, repo: str, root: Path | None = None, settled_root: Path | None = None,
                  models: tuple | None = None) -> dict:
    """Rebuild the settled VIEW of every raw partition of `repo` (optionally only some model_ids).
    The raw ledger is only READ. Each view is asserted 1:1 with its raw file (same row count, same keys,
    same order) and written only when its content changed."""
    root = Path(root or LEDGER_DIR)
    settled_root = Path(settled_root or SETTLED_DIR)
    stats = {"partitions": 0, "rows": 0, "written": 0, "settled": 0}
    for p in partitions(root):
        part = parse_partition(root, p)
        if part["repo"] != repo or (models and part["model_id"] not in models):
            continue
        raw = read_partition(p)
        view = settled_view(raw, outcome_of)
        assert len(view) == len(raw), f"{p}: settled view is not 1:1 with the raw ledger"
        assert view[KEY_COLS].equals(raw[KEY_COLS]), f"{p}: settled view keys differ from the raw ledger"
        sp = settled_path(settled_root, root, p)
        text = view.to_csv(index=False, lineterminator="\n")
        stats["partitions"] += 1
        stats["rows"] += len(view)
        stats["settled"] += int((view["outcome_status"] == "SETTLED").sum())
        if sp.exists() and sp.read_text(encoding="utf-8") == text:
            continue
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(text, encoding="utf-8", newline="")
        stats["written"] += 1
    return stats


# ── registry ───────────────────────────────────────────────────────────────────────────────────────

#: Why a registry league is outside the ledger's scope (recorded, never silently dropped).
NOT_SCORED_REASON = {
    "COLLECTION_ONLY": "collection only - no production team model serves it (props rows appear only where the "
                       "props pipeline serves the league)",
    "TRAINING_ONLY": "training data only - no production model serves it",
}
PROVIDER_GAP = ("tracked, but the OddsAPI board has no events for it (ENABLED_PROVIDER_UNSUPPORTED): v9 builds no "
                "board row, so no team model scores it; props rows appear where the props pipeline serves it")


def league_scope(slugs: "LeagueSlugs | None" = None) -> list:
    """Every registry league with its ledger scope: tracked (scored) or not scored, with the reason."""
    slugs = slugs or LeagueSlugs()
    out = []
    for e in slugs.leagues:
        st = e.get("production_state")
        if st == "ENABLED_PROVIDER_UNSUPPORTED":
            scope, why = "tracked_not_on_board", PROVIDER_GAP
        elif tracked(e):
            scope, why = "tracked", ""
        else:
            scope, why = "not_scored", NOT_SCORED_REASON.get(st, f"production_state {st!r}: not served")
        out.append({"league_slug": e["id"], "league": e["name"], "production_state": st, "scope": scope,
                    "not_scored_reason": why})
    return out


def build_registry(root: Path | None = None, settled_root: Path | None = None,
                   markets: dict | None = None, slugs: LeagueSlugs | None = None,
                   repos: tuple | None = None, now: datetime | None = None) -> dict:
    """Every model x market of config/prediction_ledger_markets.json (pending ones included, with no rows)
    with model versions, first/last prediction date, row and settled counts, and the same per league."""
    root = Path(root or LEDGER_DIR)
    settled_root = Path(settled_root or SETTLED_DIR)
    slugs = slugs or LeagueSlugs()
    markets = markets or load_markets()
    agg: dict = {}
    for p in partitions(root):
        part = parse_partition(root, p)
        if repos and part["repo"] not in repos:
            continue
        d = read_partition(p)
        sp = settled_path(settled_root, root, p)
        n_settled = 0
        if sp.exists():
            s = pd.read_csv(sp, dtype=str, keep_default_na=False, usecols=["outcome_status", "scored"])
            n_settled = int(((s["outcome_status"] == "SETTLED") & (s["scored"] == "1")).sum())
        k = (part["repo"], part["model_id"], part["market"])
        a = agg.setdefault(k, {"versions": set(), "rows": 0, "scored": 0, "settled": 0, "first": None,
                               "last": None, "leagues": {}})
        at = d["predicted_at"][d["predicted_at"] != ""]
        first, last = (at.min(), at.max()) if len(at) else (None, None)
        a["versions"].update(v for v in d["model_version"].unique() if v)
        a["rows"] += len(d)
        a["scored"] += int((d["scored"] == "1").sum())
        a["settled"] += n_settled
        a["first"] = min(x for x in (a["first"], first) if x) if (a["first"] or first) else None
        a["last"] = max(x for x in (a["last"], last) if x) if (a["last"] or last) else None
        lg = a["leagues"].setdefault(part["league_slug"], {"rows": 0, "predicted": 0, "settled": 0,
                                                            "first": None, "last": None, "seasons": []})
        lg["rows"] += len(d)
        lg["predicted"] += int((d["scored"] == "1").sum())
        lg["settled"] += n_settled
        lg["first"] = min(x for x in (lg["first"], first) if x) if (lg["first"] or first) else None
        lg["last"] = max(x for x in (lg["last"], last) if x) if (lg["last"] or last) else None
        lg["seasons"] = sorted(set(lg["seasons"]) | {part["season"]})
    models = []
    for e in markets["entries"]:
        if repos and e["repo"] not in repos:
            continue
        a = agg.get((e["repo"], e["model_id"], e["market"]))
        models.append({
            "repo": e["repo"], "model_id": e["model_id"], "market": e["market"],
            "status": e.get("status"), "family": e.get("family", ""),
            "selections": e.get("selections", []), "lines": e.get("lines", []),
            "model_versions": sorted(a["versions"]) if a else [],
            "first_prediction": a["first"] if a else None, "last_prediction": a["last"] if a else None,
            "rows": a["rows"] if a else 0, "predicted": a["scored"] if a else 0,
            "settled": a["settled"] if a else 0,
            "leagues": {s: a["leagues"][s] for s in sorted(a["leagues"])} if a else {},
        })
    return {"schema_version": SCHEMA_VERSION,
            "generated_at": (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "layout": "output/prediction_ledger/<repo>/<model_id>/<market>/<league_slug>/<season>.csv",
            "key": KEY_COLS, "snapshots": [{"label": s, "max_minutes_to_kickoff": m} for s, m in SNAPSHOTS],
            "models": models, "leagues": league_scope(slugs)}


def write_registry(path: Path | None = None, **kw) -> dict:
    reg = build_registry(**kw)
    path = Path(path or REGISTRY_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reg, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    return reg


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--registry", action="store_true")
    ap.add_argument("--root", default=None)
    a = ap.parse_args(argv)
    rc = 0
    if a.validate:
        issues = validate(Path(a.root) if a.root else None)
        for i in issues:
            print(f"PREDICTION LEDGER VIOLATION: {i}")
        print(f"prediction ledger: {len(partitions(Path(a.root) if a.root else None))} partition(s), "
              f"{len(issues)} violation(s)")
        rc = 1 if issues else 0
    if a.registry:
        reg = write_registry()
        print(f"registry: {len(reg['models'])} model x market entr(ies) -> {REGISTRY_FILE}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
