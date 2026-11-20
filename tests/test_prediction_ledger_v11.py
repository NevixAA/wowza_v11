"""v11 prediction ledger (src/prediction_ledger_v11.py) - owner rule 2026-11-17/18. Learning only.

The shared core src/prediction_ledger.py is VENDORED verbatim from NevixAA/wowza-betting (the way Pro does
it) and pinned here: same bytes as v9's when a v9 checkout carries it, same schema, same key, no price
column. This file pins the v11 writer: every evaluated fixture, both sides, first write wins, the snapshot
rules, nothing price-derived, the flag, the collector hook and settlement on v11's results.

    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/test_prediction_ledger_v11.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "scripts"))

from src import prediction_ledger as PL  # noqa: E402
from src import prediction_ledger_v11 as P11  # noqa: E402

NOW = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)
KO = "2026-11-21T15:00:00Z"                     # 180 min after NOW: first_sight only
KO_SOON = "2026-11-21T13:30:00Z"                # 90 min after NOW: first_sight + final
#: sha256 of src/prediction_ledger.py (LF) on wowza-betting main after #56 (corners / cards stat_outcome). Change it only by
#: re-vendoring v9's file, never by editing the copy here.
CORE_SHA256 = "7a6ecd8a2a975e38ad0bb7cd1d2d0f677648717b1b27208b980d20c070ff494f"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


@pytest.fixture(autouse=True)
def registry(tmp_path, monkeypatch):
    reg = {"registry_version": "test", "leagues": [
        {"id": "league_one", "name": "League One", "aliases": ["English League One"], "model_type": "standard",
         "paper": False, "football_data_code": "E2", "production_state": "ENABLED",
         "season_rule": {"type": "start_year", "start_month": 7}},
        {"id": "usa_mls", "name": "USA MLS", "aliases": [], "model_type": "new_format", "paper": True,
         "football_data_code": "USA", "production_state": "PAPER",
         "season_rule": {"type": "calendar_year"}}]}
    (tmp_path / "v9" / "config").mkdir(parents=True)
    (tmp_path / "v9" / "config" / "leagues.json").write_text(json.dumps(reg), encoding="utf-8")
    monkeypatch.setenv("V9_LOCAL", str(tmp_path / "v9"))
    monkeypatch.setattr(PL, "LEDGER_DIR", tmp_path / "L")
    monkeypatch.setattr(PL, "SETTLED_DIR", tmp_path / "S")
    monkeypatch.setattr(PL, "REGISTRY_FILE", tmp_path / "L" / "registry.json")
    from src import league_registry as LR
    LR.reset_cache()
    yield
    LR.reset_cache()


def _rec(now=NOW):
    return P11.Recorder(now=now)


def _add(rec, **kw):
    a = dict(league="League One", home="Barnsley", away="Wigan Athletic", date="2026-11-21", kickoff_utc=KO,
             oa_event_id="8e1abc", probs={"OVER": 0.5412, "UNDER": 0.4791}, reasons={},
             inputs=(0.6, 900, 0.03, 0.4))
    a.update(kw)
    rec.add(**a)


def _read(tmp_path, league="league_one", season="2026"):
    return pd.read_csv(tmp_path / "L" / "v11" / "v11_shadow" / "ou25" / league / f"{season}.csv", dtype=str,
                       keep_default_na=False)


# ── shared core + contract ───────────────────────────────────────────────────────────────────────

def _v9_dir() -> Path | None:
    for c in (os.getenv("V9_LOCAL_REAL", ""), str(PROJ.parent / "wowza-betting"), str(PROJ / "_v9")):
        if c and (Path(c) / "src" / "prediction_ledger.py").is_file():
            return Path(c)
    return None


def test_core_is_v9s_file_verbatim():
    assert PL.SCHEMA_VERSION == "pl/1.0"
    assert PL.KEY_COLS == ["model_id", "model_version", "market", "line", "selection", "fixture_ref",
                           "player_id", "snapshot"]
    assert [s for s, _ in PL.SNAPSHOTS] == ["first_sight", "final"] and dict(PL.SNAPSHOTS)["final"] == 120
    assert _sha(PROJ / "src" / "prediction_ledger.py") == CORE_SHA256
    v9 = _v9_dir()
    if v9 is None:
        pytest.skip("no v9 checkout carrying src/prediction_ledger.py")
    assert _sha(v9 / "src" / "prediction_ledger.py") == _sha(PROJ / "src" / "prediction_ledger.py")


def test_contract_registers_v11_shadow_active_and_matches_v9():
    m = PL.load_markets()
    assert m["schema_version"] == PL.SCHEMA_VERSION
    mine = [e for e in m["entries"] if e["repo"] == "v11"]
    assert [(e["model_id"], e["market"], e["status"]) for e in mine] == [("v11_shadow", "ou25", "active")]
    assert mine[0]["selections"] == ["OVER", "UNDER"] and mine[0]["settle"]["rule"] == "total_goals_over"
    v9 = _v9_dir()
    if v9 is None or not (v9 / "config" / "prediction_ledger_markets.json").is_file():
        pytest.skip("no v9 checkout carrying config/prediction_ledger_markets.json")
    theirs = json.loads((v9 / "config" / "prediction_ledger_markets.json").read_text(encoding="utf-8"))
    t11 = [e for e in theirs["entries"] if e["repo"] == "v11"]
    if t11 and t11[0].get("status") == "pending_writer":
        pytest.skip("v9's contract has not taken the v11_shadow activation yet (merge the contract PR first)")
    assert t11 == mine and theirs["schema_version"] == m["schema_version"]


def test_no_price_column_and_reasons_carry_no_number():
    assert not [c for c in PL.COLUMNS if PL.PRICE_LIKE.search(c)]
    assert P11.clean_reason("odds rejected: overround 1.312 outside (1.0,1.25]") == \
        "odds rejected: overround outside the sanity bounds"
    assert P11.clean_reason("odds_above_cap(6.0)") == "odds_above_cap"
    assert not any(ch.isdigit() for ch in P11.clean_reason("below_floors(ev_lb -0.1958 vs 0.01, edge -0.01 vs 0.03)"))
    assert P11.clean_reason("MISSING_EVIDENCE") == "MISSING_EVIDENCE"


# ── writer ───────────────────────────────────────────────────────────────────────────────────────

def test_every_fixture_both_sides_first_sight(tmp_path):
    rec = _rec()
    _add(rec)
    _add(rec, home="Exeter City", away="Bristol Rovers", oa_event_id=None,
         probs={"OVER": None, "UNDER": None}, reasons={"OVER": "MISSING_EVIDENCE", "UNDER": "MISSING_EVIDENCE"})
    st = rec.flush()
    assert st["written"] == 4 and st["refused"] == 0 and st["violations"] == 0
    d = _read(tmp_path)
    assert d["selection"].tolist() == ["OVER", "UNDER", "OVER", "UNDER"]
    assert set(d["snapshot"]) == {"first_sight"}                        # 180 min out: no final yet
    assert d["probability"].tolist()[:2] == ["0.5412", "0.4791"]        # sides need not sum to 1
    assert d["fixture_ref"].tolist()[:2] == ["oa:8e1abc"] * 2
    assert d["fixture_ref"].iloc[2] == "nk:league_one|2026-11-21|Exeter City|Bristol Rovers"
    assert d["scored"].tolist() == ["1", "1", "0", "0"]
    assert d["not_scored_reason"].iloc[2] == "MISSING_EVIDENCE"
    assert (d["model_version"] == P11.model_version()[0]).all() and (d["predicted_at"] == "2026-11-21T12:00:00Z").all()
    assert (d["season"] == "2026").all() and (d["line"] == "2.5").all() and (d["repo"] == "v11").all()
    assert PL.validate(tmp_path / "L", slugs=rec.slugs) == []


def test_first_write_wins_and_final_inside_120_minutes(tmp_path):
    rec = _rec()
    _add(rec, kickoff_utc=KO_SOON)
    assert rec.flush()["written"] == 4                                  # first_sight + final, both sides
    rec = _rec(datetime(2026, 11, 21, 12, 30, tzinfo=timezone.utc))
    _add(rec, kickoff_utc=KO_SOON, probs={"OVER": 0.9, "UNDER": 0.1})
    st = rec.flush()
    assert st["written"] == 0 and st["duplicate_skipped"] == 4          # the first opinion is kept
    d = _read(tmp_path)
    assert sorted(d["snapshot"].unique()) == ["final", "first_sight"]
    assert d.loc[d["selection"] == "OVER", "probability"].tolist() == ["0.5412", "0.5412"]
    k = d[PL.KEY_COLS]
    assert not k.duplicated().any()


def test_post_kickoff_and_unknown_kickoff_are_never_written(tmp_path):
    rec = _rec()
    _add(rec, kickoff_utc="2026-11-21T11:59:00Z")
    _add(rec, kickoff_utc=None, home="X", away="Y")
    st = rec.flush()
    assert st["offered"] == 0 and st["run_not_pre_kickoff"] == 1 and st["run_no_kickoff"] == 1
    assert not (tmp_path / "L" / "v11" / "v11_shadow").exists()


def test_numeric_oa_id_falls_back_to_the_natural_key(tmp_path):
    rec = _rec()
    _add(rec, oa_event_id=8e12)
    st = rec.flush()
    assert st["run_numeric_oa_id"] == 1
    assert _read(tmp_path)["fixture_ref"].iloc[0] == "nk:league_one|2026-11-21|Barnsley|Wigan Athletic"


def test_unknown_league_is_refused(tmp_path):
    rec = _rec()
    _add(rec, league="Atlantis League")
    st = rec.flush()
    assert st["written"] == 0 and st["refused_reasons"] == {"league not in the registry": 2}


def test_flag_off_records_nothing(monkeypatch):
    monkeypatch.delenv(PL.FLAG_ENV, raising=False)
    assert P11.recorder(NOW) is None
    monkeypatch.setenv(PL.FLAG_ENV, "1")
    assert P11.recorder(NOW) is None
    monkeypatch.setenv(PL.FLAG_ENV, "true")
    assert isinstance(P11.recorder(NOW), P11.Recorder)


def test_inactive_entry_refuses_to_record(monkeypatch):
    m = PL.load_markets()
    for e in m["entries"]:
        if e["repo"] == "v11":
            e["status"] = "pending_writer"
    with pytest.raises(PL.LedgerError):
        P11.Recorder(now=NOW, markets=m)


def test_version_ignores_line_endings_and_names_the_model():
    v, sha = P11.model_version()
    assert v.startswith("v11_shadow_") and len(sha) == 12 and v.endswith(sha[:8])


# ── collector hook (scripts/v11_shadow.run) ──────────────────────────────────────────────────────

class _Ev:
    n_eff, ece, scope, source = 900, 0.03, "standard|League One", "test"


class _Store:
    load_error = None

    def lookup(self, mt, lg):
        return _Ev() if lg == "League One" else None


def _board():
    base = {"date": "2026-11-21", "kickoff_utc": KO, "generated_at": "2026-11-21T11:50:00Z",
            "odds_over15": 1.30, "odds_over35": 3.40, "best_side": "", "signal_tier": ""}
    return pd.DataFrame([
        dict(base, league="League One", home_team="Barnsley", away_team="Wigan Athletic", p_over25=0.58,
             odds_over25=1.95, odds_under25=1.90, _oa_event_id="8e1abc"),
        dict(base, league="USA MLS", home_team="LA Galaxy", away_team="Austin FC", p_over25=0.61,
             odds_over25=1.80, odds_under25=2.05, _oa_event_id=None),               # no evidence measured
        dict(base, league="League One", home_team="Exeter City", away_team="Bristol Rovers", p_over25="",
             odds_over25=1.95, odds_under25=1.90, _oa_event_id="9f2def"),           # v9 could not score
        dict(base, league="Atlantis League", home_team="A", away_team="B", p_over25=0.5,
             odds_over25=1.95, odds_under25=1.90, _oa_event_id="x1"),                # not in the registry
    ])


def _run_shadow(tmp_path, monkeypatch, flag: str | None):
    import config
    import v11_shadow as S
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(config, "OUTPUT_DIR", out)
    monkeypatch.setattr(S, "_EVIDENCE", _Store())
    board = _board()

    def loader(name, dtype=None):
        if name == "predictions.csv":
            p = tmp_path / "board.csv"
            board.to_csv(p, index=False)
            return pd.read_csv(p, dtype=dtype)
        raise FileNotFoundError(name)
    monkeypatch.setattr(S, "_load_v9", loader)
    if flag is None:
        monkeypatch.delenv(PL.FLAG_ENV, raising=False)
    else:
        monkeypatch.setenv(PL.FLAG_ENV, flag)

    class _DT(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW
    monkeypatch.setattr(S, "datetime", _DT)
    n = S.run()
    return n, pd.read_csv(out / "v11_shadow_snapshots.csv")


def test_collector_records_every_evaluated_fixture(tmp_path, monkeypatch):
    n, snaps = _run_shadow(tmp_path, monkeypatch, "true")
    assert n == 2                                                       # Barnsley + Galaxy rows, as before
    lo = _read(tmp_path)
    by = {(r["home_team"], r["selection"]): r for r in lo.to_dict("records")}
    assert set(by) == {("Barnsley", "OVER"), ("Barnsley", "UNDER"), ("Exeter City", "OVER"),
                       ("Exeter City", "UNDER")}
    assert by[("Barnsley", "OVER")]["scored"] == "1" and by[("Barnsley", "UNDER")]["scored"] == "1"
    assert by[("Exeter City", "OVER")]["not_scored_reason"] == "v9 board row has no p_over25"
    # each side's probability is v11's own blend from that side's decision (standard cap 0.40, no consensus)
    import v11_shadow as S
    exp_o = S._decide(0.58, 1.95, 1.90, 900, 0.03, None, None, 0.40, cons=None, side="OVER")["p_blend"]
    exp_u = S._decide(0.42, 1.90, 1.95, 900, 0.03, None, None, 0.40, cons=None, side="UNDER")["p_blend"]
    assert float(by[("Barnsley", "OVER")]["probability"]) == pytest.approx(exp_o)
    assert float(by[("Barnsley", "UNDER")]["probability"]) == pytest.approx(exp_u)
    assert len(snaps) == 2 and "Exeter City vs Bristol Rovers" not in set(snaps["match"])
    mls = pd.read_csv(tmp_path / "L" / "v11" / "v11_shadow" / "ou25" / "usa_mls" / "2026.csv", dtype=str,
                      keep_default_na=False)
    assert (mls["scored"] == "0").all() and (mls["not_scored_reason"] == "MISSING_EVIDENCE").all()
    assert mls["fixture_ref"].iloc[0] == "nk:usa_mls|2026-11-21|LA Galaxy|Austin FC"
    assert PL.validate(tmp_path / "L", slugs=P11.league_slugs()) == []
    for p in (tmp_path / "L").rglob("*.csv"):
        cols = pd.read_csv(p, nrows=0).columns
        assert not [c for c in cols if PL.PRICE_LIKE.search(c)]


def test_collector_output_is_unchanged_by_the_flag(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _, on = _run_shadow(tmp_path / "a", monkeypatch, "true")
    monkeypatch.setattr(PL, "LEDGER_DIR", tmp_path / "L_off")
    _, off = _run_shadow(tmp_path / "b", monkeypatch, None)
    pd.testing.assert_frame_equal(on, off)
    assert not (tmp_path / "L_off").exists()


# ── settlement ───────────────────────────────────────────────────────────────────────────────────

def test_settled_view_is_one_to_one_and_grades_on_v11_results(tmp_path, monkeypatch):
    rec = _rec()
    _add(rec)
    _add(rec, home="Exeter City", away="Bristol Rovers", oa_event_id="9f2def")
    rec.flush()
    results = pd.DataFrame([{"date": pd.Timestamp("2026-11-21"), "home_team": "Barnsley",
                             "away_team": "Wigan Athletic", "home_goals": 2, "away_goals": 1,
                             "league": "League One"}])
    monkeypatch.setattr(P11, "_now", lambda: datetime(2026, 11, 22, tzinfo=timezone.utc))
    st = P11.settle(results=results, now=datetime(2026, 11, 22, tzinfo=timezone.utc), force=True)
    assert st["rows"] == 4 and st["settled"] == 2
    v = pd.read_csv(tmp_path / "S" / "v11" / "v11_shadow" / "ou25" / "league_one" / "2026.csv", dtype=str,
                    keep_default_na=False)
    raw = _read(tmp_path)
    assert len(v) == len(raw) and v[PL.KEY_COLS].equals(raw[PL.KEY_COLS])
    b = v[v["home_team"] == "Barnsley"].set_index("selection")
    assert b.loc["OVER", "y"] == "1" and b.loc["UNDER", "y"] == "0" and b.loc["OVER", "outcome_value"] == "2-1"
    assert b.loc["OVER", "result_source"] == "football-data.co.uk" and b.loc["OVER", "brier"] != ""
    assert (v[v["home_team"] == "Exeter City"]["outcome_status"] == "PENDING").all()
    late = P11.outcome_fn({}, now=datetime(2026, 11, 25, tzinfo=timezone.utc))(raw.iloc[2].to_dict())
    assert late["outcome_status"] == "NO_RESULT"


def test_settle_and_registry_are_flag_gated(tmp_path, monkeypatch):
    monkeypatch.delenv(PL.FLAG_ENV, raising=False)
    assert "skipped" in P11.settle()
    assert P11.main(["registry"]) == 0 and not (tmp_path / "L" / "registry.json").exists()
    monkeypatch.setenv(PL.FLAG_ENV, "true")
    rec = _rec()
    _add(rec)
    rec.flush()
    assert P11.main(["registry"]) == 0
    reg = json.loads((tmp_path / "L" / "registry.json").read_text(encoding="utf-8"))
    m = [x for x in reg["models"] if x["model_id"] == "v11_shadow"]
    assert m and m[0]["rows"] == 2 and m[0]["leagues"]["league_one"]["predicted"] == 2
    assert P11.main(["validate"]) == 0


# ── private store: nothing price-derived in the public tree ─────────────────────────────────────

def _git(*args) -> "subprocess.CompletedProcess":
    import subprocess
    return subprocess.run(["git", *args], cwd=PROJ, capture_output=True, text=True)


def test_the_ledger_is_never_tracked_in_this_public_repo():
    r = _git("ls-files", "--", "output/prediction_ledger", "output/prediction_ledger_settled")
    if r.returncode != 0:
        pytest.skip("not a git checkout")
    assert r.stdout.strip() == "", "the price-derived v11 ledger is tracked in the PUBLIC repo"
    for probe in ("output/prediction_ledger/v11/v11_shadow/ou25/league_one/2026.csv",
                  "output/prediction_ledger_settled/v11/v11_shadow/ou25/league_one/2026.csv",
                  "output/prediction_ledger/registry.json"):
        assert _git("check-ignore", "-q", "--no-index", probe).returncode == 0, f"{probe} is not git-ignored"


def test_collect_workflow_writes_the_ledger_to_the_private_store_only():
    wf = (PROJ / ".github" / "workflows" / "v11_collect.yml").read_text(encoding="utf-8")
    staged = [ln for ln in wf.splitlines() if "git add" in ln and "prediction_ledger" in ln]
    assert staged == [], f"v11_collect.yml stages the ledger publicly: {staged}"
    assert "bash scripts/prediction_ledger_store.sh open" in wf and "bash scripts/prediction_ledger_store.sh push" in wf
    assert "secrets.PRICE_STORE_KEY" in wf
    # the collector records rows only when the private store opened
    assert "PREDICTION_LEDGER_ENABLED: ${{ steps.pl_store.outputs.ok == 'true' && 'true' || 'false' }}" in wf
    sh = (PROJ / "scripts" / "prediction_ledger_store.sh").read_text(encoding="utf-8")
    assert 'git@github.com:NevixAA/wowza-exec.git}"' in sh and 'BRANCH="prediction-ledger-v11"' in sh
