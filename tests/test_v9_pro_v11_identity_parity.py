"""v9 / Pro / v11 identity parity (deep-repair brief 2026-11-22, P1 item 10).

The three repos write one prediction ledger contract (src/prediction_ledger.py, vendored) and join on
`fixture_ref`. This pins what the join depends on, against REAL committed data where it is public:

  * the shared files are the same bytes in all three repos (ledger core, market contract, team names);
  * v11's writer builds `fixture_ref` with the shared core's rule, from the same OddsAPI event id v9 uses;
  * v11's O/U 2.5 entry uses v9's market vocabulary (market, line, selections, settle rule);
  * on the committed v9 and Pro ledgers: every `oa:` fixture_ref carries ONE identity (event id, league,
    season, kickoff, home, away) inside each repo AND across the two; no duplicate keys; every row was
    written before kickoff and its minutes_to_kickoff matches its own timestamps.

v11's own ledger rows are price-derived and live only in the private store (wowza-exec,
prediction-ledger-v11), so they are reconciled by the P1 report, not here; this file never reads them.

Peers: V9_LOCAL (CI checks wowza-betting out at _v9) and PRO_LOCAL (CI checks wowzaV9-Pro out at _pro).
A missing peer SKIPS its checks rather than passing them.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src import prediction_ledger as PL  # noqa: E402

ID_COLS = ["oddsapi_event_id", "league_slug", "season", "kickoff_utc", "home_team", "away_team"]
SHARED = ("src/prediction_ledger.py", "config/prediction_ledger_markets.json")


def _peer(env: str, *names: str) -> Path | None:
    for c in (os.getenv(env, ""), *(str(PROJ.parent / n) for n in names), *(str(PROJ / n) for n in names)):
        if c and (Path(c) / "src" / "prediction_ledger.py").is_file():
            return Path(c)
    return None


V9 = _peer("V9_LOCAL", "wowza-betting", "_v9")
PRO = _peer("PRO_LOCAL", "wowzaV9-Pro", "_pro")


def _lf_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _ledger(root: Path, repo: str) -> pd.DataFrame:
    base = root / "output" / "prediction_ledger" / repo
    files = [f for f in base.rglob("*.csv") if "_runs" not in f.parts and "_coverage" not in f.parts]
    if not files:
        return pd.DataFrame(columns=PL.COLUMNS)
    return pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in files], ignore_index=True)


@pytest.fixture(scope="module")
def v9_rows():
    if V9 is None:
        pytest.skip("no v9 checkout (set V9_LOCAL)")
    return _ledger(V9, "v9")


@pytest.fixture(scope="module")
def pro_rows():
    if PRO is None:
        pytest.skip("no Pro checkout (set PRO_LOCAL)")
    return _ledger(PRO, "pro")


# ── shared bytes ────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rel", SHARED)
def test_shared_ledger_files_identical_to_v9(rel):
    if V9 is None:
        pytest.skip("no v9 checkout (set V9_LOCAL)")
    assert _lf_sha(PROJ / rel) == _lf_sha(V9 / rel), f"{rel} differs from v9 - re-vendor it"


@pytest.mark.parametrize("rel", SHARED)
def test_shared_ledger_files_identical_to_pro(rel):
    if PRO is None:
        pytest.skip("no Pro checkout (set PRO_LOCAL)")
    assert _lf_sha(PROJ / rel) == _lf_sha(PRO / rel), f"{rel} differs from Pro"


def test_team_names_identical_to_v9():
    """v11 settles its ledger through these aliases; a stale copy refuses fixtures v9 already resolves
    (2026-11-22: v9 added 'hertha berlin' for the Standard ledger settlement; v11's copy lacked it)."""
    if V9 is None:
        pytest.skip("no v9 checkout (set V9_LOCAL)")
    assert _lf_sha(PROJ / "src" / "team_names.py") == _lf_sha(V9 / "src" / "team_names.py")


# ── fixture identity rule ───────────────────────────────────────────────────────────────────────

def test_v11_fixture_ref_is_the_shared_rule():
    from src import prediction_ledger_v11 as P11
    import inspect
    src = inspect.getsource(P11)
    assert "PL.fixture_ref(" in src, "v11's writer must build fixture_ref with the shared core"
    # The same OddsAPI event id yields the same key in every repo, and a numeric id is never mistaken
    # for an API-Football one.
    assert PL.fixture_ref(oddsapi_event_id="8e1abc") == "oa:8e1abc"
    assert PL.fixture_ref(fixture_id="1234", oddsapi_event_id="8e1abc") == "af:1234"
    assert PL.fixture_ref() == ""


def test_v11_market_vocabulary_matches_v9_ou25():
    m = PL.load_markets()
    by = {(e["repo"], e["model_id"], e["market"]): e for e in m["entries"]}
    v11 = by[("v11", "v11_shadow", "ou25")]
    for mid in ("v9_standard", "v9_new_format"):
        v9 = by[("v9", mid, "ou25")]
        assert (v11["line"], v11["selections"], v11["settle"]) == (v9["line"], v9["selections"], v9["settle"]), mid


# ── committed v9 / Pro ledgers (real rows) ──────────────────────────────────────────────────────

@pytest.mark.parametrize("which", ["v9", "pro"])
def test_ledger_rows_are_unique_pre_kickoff_and_self_consistent(which, request):
    d = request.getfixturevalue(f"{which}_rows")
    if d.empty:
        pytest.skip(f"no committed {which} ledger rows")
    assert not d.duplicated(PL.KEY_COLS).any(), "duplicate ledger keys"
    assert (d["fixture_ref"] != "").all(), "a row without fixture_ref"
    ko = pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce")
    pa = pd.to_datetime(d["predicted_at"], utc=True, errors="coerce")
    assert ko.notna().all() and pa.notna().all(), "unparseable kickoff_utc / predicted_at"
    assert (pa < ko).all(), "a row written at or after kickoff"
    mtk = pd.to_numeric(d["minutes_to_kickoff"], errors="coerce")
    assert ((mtk - (ko - pa).dt.total_seconds() / 60).abs() <= 1.01).all(), "minutes_to_kickoff disagrees"


@pytest.mark.parametrize("which", ["v9", "pro"])
def test_one_identity_per_fixture_ref_within_repo(which, request):
    d = request.getfixturevalue(f"{which}_rows")
    oa = d[d["fixture_ref"].str.startswith("oa:")]
    if oa.empty:
        pytest.skip(f"no oa: rows in the {which} ledger")
    n = oa.groupby("fixture_ref")[ID_COLS].nunique()
    bad = n[(n > 1).any(axis=1)]
    assert bad.empty, f"{len(bad)} fixture_ref(s) with more than one identity, e.g. {list(bad.index[:3])}"
    assert (oa["fixture_ref"] == "oa:" + oa["oddsapi_event_id"]).all()


def test_v9_and_pro_agree_on_every_shared_fixture(v9_rows, pro_rows):
    a = v9_rows[v9_rows["fixture_ref"].str.startswith("oa:")].drop_duplicates("fixture_ref")
    b = pro_rows[pro_rows["fixture_ref"].str.startswith("oa:")].drop_duplicates("fixture_ref")
    j = a[["fixture_ref", *ID_COLS]].merge(b[["fixture_ref", *ID_COLS]], on="fixture_ref", how="inner",
                                           suffixes=("", "_pro"))
    if j.empty:
        pytest.skip("no fixture shared by the committed v9 and Pro ledgers")
    for c in ID_COLS:
        diff = j[j[c] != j[f"{c}_pro"]]
        assert diff.empty, f"{c} differs on {len(diff)} shared fixture(s), e.g. {list(diff.fixture_ref[:3])}"


def test_league_slugs_come_from_v9s_registry(v9_rows, pro_rows):
    if V9 is None:
        pytest.skip("no v9 checkout (set V9_LOCAL)")
    reg = json.loads((V9 / "config" / "leagues.json").read_text(encoding="utf-8"))
    ids = {e["id"] for e in reg["leagues"]}
    for name, d in (("v9", v9_rows), ("pro", pro_rows)):
        stray = sorted(set(d["league_slug"]) - ids - {""})
        assert not stray, f"{name} ledger slugs not in v9's registry: {stray}"
