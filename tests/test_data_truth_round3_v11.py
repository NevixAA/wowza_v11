"""Data Truth round 2/3 (2026-11-02) - v11: v9 scoreboard, exchange gate, one estimator, close
label, dead file. Every test except the 0-BET explanation failed on origin/data-truth-100 23b88ab.

    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/test_data_truth_round3_v11.py
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
    """A tiny v9 registry: League One (standard, live), USA MLS (new_format, PAPER)."""
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


# ── v9 scoreboard ──────────────────────────────────────────────────────────────────────────────
def _snap(fid, ts, tier, side="OVER", o=2.10, u=1.75, league="League One", match="A vs B",
          date="2026-10-03", ko="2026-10-03T15:00:00Z"):
    k = pd.Timestamp(ko)
    return {"fixture_id": fid, "snapshot_ts": ts, "date": date, "league": league, "match": match,
            "kickoff_ts": ko,
            "minutes_to_kickoff": (k - pd.Timestamp(ts)).total_seconds() / 60,
            "live_side": side, "live_tier": tier, "odds_over25": o, "odds_under25": u}


def test_v9_tip_is_priced_at_first_sight_and_kept_when_it_later_turns_avoid():
    import v11_grade as G
    s = pd.DataFrame([_snap("f1", "2026-10-02T09:00:00Z", "VALUABLE", o=2.10),
                      _snap("f1", "2026-10-03T09:00:00Z", "VALUABLE", o=1.80),
                      _snap("f1", "2026-10-03T12:00:00Z", "AVOID", side="-", o=1.70)])
    t = G.v9_scoreboard_rows(s, {"f1": 1})
    r = t.iloc[0]
    assert bool(r["counted"]) and r["odds"] == 2.10 and r["pnl"] == pytest.approx(1.10)


def test_v9_paper_league_tips_are_not_counted():
    import v11_grade as G
    # After the 2026-10-07 paper switch (owner 2026-11-02: tips sent live before it count; see
    # tests/test_data_truth_round5_v11.py).
    s = pd.DataFrame([_snap("m1", "2026-10-09T09:00:00Z", "SNIPER", league="USA MLS",
                            date="2026-10-10", ko="2026-10-10T15:00:00Z")])
    t = G.v9_scoreboard_rows(s, {"m1": 0})
    assert not bool(t.iloc[0]["counted"]) and t.iloc[0]["exclude_reason"] == "paper_league"


def test_redated_fixture_within_48h_counts_once_on_the_actual_match():
    """Viking v Aalesund pattern: 08-29 16:00 re-dated to 08-30 15:00 (23 h), tipped under both
    fixture_ids; the old scoreboard counted it twice."""
    import v11_grade as G
    s = pd.DataFrame([
        _snap("old", "2026-08-28T09:00:00Z", "VALUABLE", side="UNDER", date="2026-08-29",
              ko="2026-08-29T16:00:00Z", u=3.50),
        _snap("old", "2026-08-28T20:00:00Z", "VALUABLE", side="UNDER", date="2026-08-29",
              ko="2026-08-29T16:00:00Z", u=3.50),
        _snap("new", "2026-08-28T21:00:00Z", "VALUABLE", side="UNDER", date="2026-08-30",
              ko="2026-08-30T15:00:00Z", u=3.75)])
    t = G.v9_scoreboard_rows(s, {"old": None, "new": 0})
    c = t[t["counted"]]
    assert len(c) == 1
    assert c.iloc[0]["fixture_id"] == "old" and c.iloc[0]["settles_on"] == "new"
    assert c.iloc[0]["odds"] == 3.50 and c.iloc[0]["pnl"] == pytest.approx(2.50)


def test_redated_more_than_48h_voids_the_original_tip():
    import v11_grade as G
    s = pd.DataFrame([
        _snap("old", "2026-09-04T09:00:00Z", "VALUABLE", date="2026-09-06",
              ko="2026-09-06T20:00:00Z"),
        _snap("new", "2026-09-04T10:00:00Z", "NO_BET", side="-", date="2026-09-11",
              ko="2026-09-11T03:00:00Z")])
    t = G.v9_scoreboard_rows(s, {"old": 1, "new": 1})
    assert t.iloc[0]["exclude_reason"].startswith("rescheduled_void")
    assert not t["counted"].any()


def test_a_genuine_second_meeting_is_not_merged():
    """Same home pairing twice a season (Ireland, MLS): both snapshotted side by side, no handoff."""
    import v11_grade as G
    rows = [_snap("m1", f"2026-08-{d:02d}T09:00:00Z", "VALUABLE", date="2026-08-21",
                  ko="2026-08-21T19:00:00Z") for d in (18, 19, 20, 21)]
    rows += [_snap("m2", f"2026-08-{d:02d}T10:00:00Z", "VALUABLE", date="2026-09-04",
                   ko="2026-09-04T19:00:00Z") for d in (18, 25, 30)]
    t = G.v9_scoreboard_rows(pd.DataFrame(rows), {"m1": 1, "m2": 0})
    assert int(t["counted"].sum()) == 2


def test_missing_result_is_never_graded_as_a_win():
    import v11_grade as G
    s = pd.DataFrame([_snap("f1", "2026-10-02T09:00:00Z", "VALUABLE")])
    t = G.v9_scoreboard_rows(s, {"f1": float("nan")})
    assert not bool(t.iloc[0]["counted"]) and t.iloc[0]["exclude_reason"] == "no_result"


def test_scoreboard_keeps_the_legacy_number_beside_the_corrected_one():
    import v11_grade as G
    graded = pd.DataFrame([{"fixture_id": "f1", "month": "2026-10", "live_tier": "VALUABLE",
                            "live_pnl": 0.8, "v11_tier": "NO_BET", "v11_pnl": None}])
    s = pd.DataFrame([_snap("f1", "2026-10-02T09:00:00Z", "VALUABLE", o=2.10)])
    g2, tips = G.attach_v9_first_sight(graded, s, {"f1": 1})
    sb = G.build_scoreboard(g2, tips)
    ov = sb[sb["scope"] == "overall"].set_index("system")
    assert ov.loc["v9", "pnl"] == pytest.approx(1.10)
    assert ov.loc["v9_latest_view (legacy)", "pnl"] == pytest.approx(0.8)
    assert g2.loc[0, "live_first_odds"] == 2.10 and g2.loc[0, "live_pnl"] == 0.8


# ── exchange gate + one estimator ──────────────────────────────────────────────────────────────
def _quotes(*specs):
    out = []
    for bk, o, u in specs:
        for side, price in (("OVER", o), ("UNDER", u)):
            out.append({"snapshot_ts": "2026-10-01T09:00:00Z", "league": "League One",
                        "match": "A vs B", "bookmaker": bk, "market": "OU25", "side": side,
                        "odds": price})
    return pd.DataFrame(out)


def test_exchange_placeholder_and_wide_pairs_are_refused():
    from src.book_consensus import build_index, load_snapshots, lookup
    q = _quotes(("a", 1.90, 1.95), ("b", 1.92, 1.92), ("c", 1.88, 1.98),
                ("betfair_ex_uk", 1.01, 1.01), ("matchbook", 1.56, 1.22))
    c = lookup(build_index(load_snapshots(lambda n: q)), "League One", "A", "B")
    assert set(c.books) == {"a", "b", "c"}
    assert c.n_rejected == 2 and c.exchange_prob is None
    assert c.best_over_odds == 1.92                     # not the 1.56 of a 46%-margin pair


def test_exchange_no_longer_overrides_the_median_and_estimator_is_recorded():
    from src.book_consensus import ESTIMATOR, Consensus
    import v11_shadow as S
    cons = Consensus(p_consensus=0.53, n_books=4, dispersion=0.01,
                     books={"a": 0.50, "b": 0.52, "c": 0.54, "betfair_ex_uk": 0.70},
                     exchange_prob=0.70, best_over_odds=1.95, best_under_odds=1.95,
                     reason="ok")
    out = S._decide(0.55, 1.90, 1.95, 500, 0.03, None, None, 0.40, cons=cons, side="OVER")
    assert out["p_market"] == pytest.approx(0.53, abs=1e-4)
    assert out["price_source"] == "cross_book_median"
    assert out["p_market_estimator"] == ESTIMATOR


def test_single_book_pair_outside_the_gate_is_not_devigged():
    import v11_shadow as S
    out = S._decide(0.55, 1.56, 1.22, 500, 0.03, None, None, 0.40, cons=None, side="OVER")
    assert out["p_market"] is None and out["reason"] == "devig_failed"


def test_research_reads_one_estimator_and_drops_unknown_rows():
    from src.book_consensus import ESTIMATOR, one_estimator_p_market
    d = pd.DataFrame([
        # round 5: a row keeps its own value only when it also says it is P(OVER)
        {"snapshot_id": "s1", "v11_p_market": 0.50, "p_market_estimator": ESTIMATOR,
         "v11_p_market_basis": "over"},
        {"snapshot_id": "s2", "v11_p_market": 0.70, "p_market_estimator": None},   # old exchange
        {"snapshot_id": "s3", "v11_p_market": 0.40, "p_market_estimator": None}])  # not restated
    side = pd.DataFrame([{"snapshot_id": "s2", "p_market_restated": 0.52}])
    r = one_estimator_p_market(d, side).set_index("snapshot_id")
    assert list(r.index) == ["s1", "s2"]
    assert r.loc["s2", "v11_p_market"] == 0.52 and r.loc["s2", "v11_p_market_original"] == 0.70


def test_momentum_restatement_sidecar_is_committed_beside_the_original():
    c = PROJ / "output" / "corrections"
    r = pd.read_csv(c / "v11_momentum_restated_2026-11-02.csv")
    assert {"as_published", "exchange_gate_rows_removed", "one_estimator_restated"} <= set(
        r["variant"])
    assert (PROJ / "output" / "v11_momentum_control.csv").exists()   # original untouched


# ── close label ────────────────────────────────────────────────────────────────────────────────
def test_close_is_labelled_for_what_it_is():
    from src import movement as mv
    e = pd.DataFrame([{"fixture_id": "f", "snapshot_ts": "2026-10-01T08:00:00Z",
                       "v11_p_market": 0.50, "p_model_over": 0.56, "odds_over25": 2.0,
                       "odds_under25": 1.9, "minutes_to_kickoff": 600.0, "n_books": 5}])
    c = pd.DataFrame([{"fixture_id": "f", "snapshot_ts": "2026-10-01T16:00:00Z",
                       "v11_p_market": 0.53, "odds_over25": 1.9, "odds_under25": 2.0,
                       "minutes_to_kickoff": 126.0}])
    d = mv.compute(e, c)
    assert d.iloc[0]["close_kind"] == "LAST_PRE_KICKOFF_SNAPSHOT"
    s = mv.summarise(d, "x", "fixture")
    assert s.close_definition == "last_pre_kickoff_snapshot"
    assert s.median_close_min_to_kickoff == 126.0 and s.share_close_within_30m == 0.0


# ── dead file ──────────────────────────────────────────────────────────────────────────────────
def test_dead_market_movement_file_is_gone():
    assert not (PROJ / "output" / "v11_market_movement.csv").exists()


# ── 0 BET rows: explained, not a bug ───────────────────────────────────────────────────────────
def test_bet_state_needs_150_clean_clv_observations_with_a_positive_mean():
    """BET = SNIPER/MARKSMAN, which requires clv_ok: rolling clean CLV > 0 on >= MIN_CLV_N (150)
    observations for the league. On 82,831 snapshots no league ever had both (only USA MLS reached
    150, at -0.96%), and only 2 snapshots cleared the EV floors at all. The gate works as written."""
    import config
    import v11_shadow as S
    from src.book_consensus import Consensus
    cons = Consensus(p_consensus=0.40, n_books=3, dispersion=0.0, best_over_odds=2.0,
                     best_under_odds=2.0, books={"a": 0.4, "b": 0.4, "c": 0.4}, reason="ok")
    args = dict(p_model=0.70, o_bet=2.30, o_other=1.70, n_eff=5000, ece=0.0, w_cap=0.40,
                cons=cons, side="OVER")
    short = S._decide(rclv=1.0, rclv_n=config.MIN_CLV_N - 1, **args)
    enough = S._decide(rclv=1.0, rclv_n=config.MIN_CLV_N, **args)
    assert S._tier_to_state(short["tier"]) == "PAPER"
    assert S._tier_to_state(enough["tier"]) == "BET"
