"""Data Truth round 5 (2026-11-02) - v11: P(OVER) always in v11_p_market, one-estimator loader in
movement and microstructure, restated sidecars on the full history, MLS tips before the paper switch
count, corrections committed by CI. Every test here failed on origin/data-truth-100 749e796.

    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/test_data_truth_round5_v11.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "scripts"))


@pytest.fixture(autouse=True)
def registry(tmp_path, monkeypatch):
    """A tiny v9 registry: League One (standard, live), USA MLS (new_format, PAPER, no date)."""
    reg = {"registry_version": "test", "leagues": [
        {"name": "League One", "aliases": [], "model_type": "standard", "paper": False,
         "football_data_code": "E2"},
        {"name": "USA MLS", "aliases": [], "model_type": "new_format", "paper": True,
         "football_data_code": "USA"}]}
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "leagues.json").write_text(json.dumps(reg), encoding="utf-8")
    monkeypatch.setenv("V9_LOCAL", str(tmp_path))
    from src import league_registry as LR
    LR.reset_cache()
    yield
    LR.reset_cache()


# ── the shadow records P(OVER) ─────────────────────────────────────────────────────────────────
def test_shadow_records_p_over_even_when_the_under_decision_ranks_higher():
    import v11_shadow as S
    from src.book_consensus import ESTIMATOR
    from src.edge_engine import power_devig
    under_best = {"tier": "OBSERVE", "p_market": 0.4718}          # the UNDER side's own value
    cols = S.market_columns(2.00, 1.80, None, under_best, valid=True)
    assert cols["v11_p_market"] == pytest.approx(round(power_devig(2.00, 1.80), 4))
    assert cols["v11_p_market"] < 0.5                               # P(OVER) at 2.00 / 1.80
    assert cols["v11_p_market_side"] == 0.4718
    assert cols["v11_p_market_basis"] == "over" and cols["p_market_estimator"] == ESTIMATOR


def test_p_over_is_the_consensus_median_and_contaminated_odds_record_nothing():
    import v11_shadow as S
    from src.book_consensus import Consensus
    cons = Consensus(p_consensus=0.52, n_books=3, dispersion=0.01,
                     books={"a": 0.50, "b": 0.52, "c": 0.56}, reason="ok")
    assert S.market_p_over(2.0, 1.8, cons) == pytest.approx(0.52)
    none = S.market_columns(2.0, 1.8, cons, {"p_market": None}, valid=False)
    assert none["v11_p_market"] is None and none["p_market_estimator"] is None


def test_rows_with_the_estimator_but_no_over_basis_take_the_sidecar():
    """Rows written by the round-4 shadow carry p_market_estimator but hold the chosen side's value
    (P(UNDER) when UNDER ranked higher). Only a row that says basis 'over' keeps its own value."""
    from src.book_consensus import ESTIMATOR, one_estimator_p_market
    d = pd.DataFrame([
        {"snapshot_id": "a", "v11_p_market": 0.45, "p_market_estimator": ESTIMATOR},
        {"snapshot_id": "b", "v11_p_market": 0.61, "p_market_estimator": ESTIMATOR,
         "v11_p_market_basis": "over"}])
    side = pd.DataFrame([{"snapshot_id": "a", "p_market_restated": 0.55}])
    r = one_estimator_p_market(d, side).set_index("snapshot_id")
    assert r.loc["a", "v11_p_market"] == 0.55 and r.loc["b", "v11_p_market"] == 0.61


# ── one-estimator loader in movement and microstructure ────────────────────────────────────────
def _write_history(out: Path):
    (out / "corrections").mkdir(parents=True)
    pd.DataFrame([
        {"snapshot_id": "s1", "fixture_id": "f", "snapshot_ts": "2026-10-01T08:00:00Z",
         "v11_p_market": 0.70, "p_model_over": 0.55},          # old exchange value, restated
        {"snapshot_id": "s2", "fixture_id": "f", "snapshot_ts": "2026-10-01T09:00:00Z",
         "v11_p_market": 0.40, "p_model_over": 0.55}]          # never restated -> dropped
    ).to_csv(out / "v11_shadow_snapshots.csv", index=False)
    pd.DataFrame([{"snapshot_id": "s1", "p_market_restated": 0.52}]).to_csv(
        out / "corrections" / "v11_p_market_restated.csv", index=False)


def test_market_movement_reads_the_one_estimator(tmp_path, monkeypatch):
    import config
    import v11_market_movement as MM
    _write_history(tmp_path / "out")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "out")
    s = MM._load_snapshots()
    assert list(s["snapshot_id"]) == ["s1"] and s.iloc[0]["v11_p_market"] == 0.52


def test_microstructure_reads_the_one_estimator(tmp_path, monkeypatch):
    import v11_microstructure as MS
    _write_history(tmp_path / "out")
    monkeypatch.setattr(MS, "SNAPS", tmp_path / "out" / "v11_shadow_snapshots.csv")
    s = MS.load_snapshots()
    assert list(s["snapshot_id"]) == ["s1"] and s.iloc[0]["v11_p_market"] == 0.52


# ── restated sidecars ──────────────────────────────────────────────────────────────────────────
def test_restated_sidecar_covers_every_committed_snapshot():
    from src.book_consensus import ESTIMATOR
    snaps = pd.read_csv(PROJ / "output" / "v11_shadow_snapshots.csv", low_memory=False)
    side = pd.read_csv(PROJ / "output" / "corrections" / "v11_p_market_restated.csv",
                       usecols=["snapshot_id"])
    own = ((snaps.get("p_market_estimator", pd.Series(dtype=object)).astype(str) == ESTIMATOR)
           & (snaps.get("v11_p_market_basis", pd.Series(dtype=object)).astype(str) == "over"))
    need = set(snaps.loc[~own.reindex(snaps.index, fill_value=False), "snapshot_id"])
    assert need <= set(side["snapshot_id"])


def test_movement_and_microstructure_restatements_sit_beside_the_originals():
    c = PROJ / "output" / "corrections"
    for name in ("v11_movement_restated_2026-11-02.csv",
                 "v11_microstructure_restated_2026-11-02.csv"):
        r = pd.read_csv(c / name, low_memory=False)
        assert {"as_published", "original_estimator_recomputed",
                "one_estimator_restated"} <= set(r["variant"])
    assert (PROJ / "output" / "v11_movement_summary.csv").exists()
    assert (PROJ / "output" / "v11_microstructure_segments.csv").exists()


def test_append_mode_keeps_existing_rows_byte_for_byte(tmp_path):
    import v11_restate_p_market as R
    from src.book_consensus import ESTIMATOR
    p = tmp_path / "side.csv"
    p.write_bytes(b"snapshot_id,p_market_restated\nold,0.5100001\n")
    old = pd.read_csv(p)
    snaps = pd.DataFrame([
        {"snapshot_id": "old"}, {"snapshot_id": "new"},
        {"snapshot_id": "own", "p_market_estimator": ESTIMATOR, "v11_p_market_basis": "over"}])
    assert list(R.pending_snapshots(snaps, old)["snapshot_id"]) == ["new"]
    R.append_rows(p, list(old.columns), pd.DataFrame([{"snapshot_id": "new",
                                                       "p_market_restated": 0.48}]))
    assert p.read_bytes() == b"snapshot_id,p_market_restated\nold,0.5100001\nnew,0.48\n"


def test_ci_restates_new_rows_and_commits_the_corrections():
    wf = (PROJ / ".github" / "workflows" / "v11_collect.yml").read_text(encoding="utf-8")
    assert "scripts/v11_restate_p_market.py --v9" in wf and "--append" in wf
    assert "output/corrections/" in wf
    # restated BEFORE the research steps that read it
    assert wf.index("v11_restate_p_market.py") < wf.index("v11_market_movement.py")


# ── v9 scoreboard: MLS before the paper switch ─────────────────────────────────────────────────
def _snap(fid, ts, ko, league="USA MLS"):
    k = pd.Timestamp(ko)
    return {"fixture_id": fid, "snapshot_ts": ts, "date": ko[:10], "league": league,
            "match": f"{fid} H vs {fid} A", "kickoff_ts": ko,
            "minutes_to_kickoff": (k - pd.Timestamp(ts)).total_seconds() / 60,
            "live_side": "OVER", "live_tier": "VALUABLE", "odds_over25": 2.0,
            "odds_under25": 1.8}


def test_mls_tips_sent_live_before_the_paper_switch_count_and_after_are_paper():
    import v11_grade as G
    s = pd.DataFrame([
        _snap("before", "2026-10-04T09:00:00Z", "2026-10-05T00:30:00Z"),
        _snap("sameday", "2026-10-07T10:00:00Z", "2026-10-08T00:30:00Z"),   # before 12:46 UTC
        _snap("after", "2026-10-07T13:00:00Z", "2026-10-08T23:30:00Z")])
    t = G.v9_scoreboard_rows(s, {"before": 1, "sameday": 0, "after": 1}).set_index("fixture_id")
    assert bool(t.loc["before", "counted"]) and t.loc["before", "pnl"] == pytest.approx(1.0)
    assert bool(t.loc["sameday", "counted"])
    assert not bool(t.loc["after", "counted"]) and t.loc["after", "exclude_reason"] == "paper_league"


def test_a_registry_paper_since_field_takes_precedence(tmp_path, monkeypatch):
    import v11_grade as G
    from src import league_registry as LR
    reg = {"registry_version": "t", "leagues": [
        {"name": "USA MLS", "aliases": [], "model_type": "new_format", "paper": True,
         "paper_since": "2026-09-01T00:00:00Z", "football_data_code": "USA"}]}
    d = tmp_path / "v9b" / "config"
    d.mkdir(parents=True)
    (d / "leagues.json").write_text(json.dumps(reg), encoding="utf-8")
    monkeypatch.setenv("V9_LOCAL", str(tmp_path / "v9b"))
    LR.reset_cache()
    s = pd.DataFrame([_snap("x", "2026-09-10T09:00:00Z", "2026-09-11T00:30:00Z")])
    t = G.v9_scoreboard_rows(s, {"x": 1})
    assert t.iloc[0]["exclude_reason"] == "paper_league"
