"""KICKOFF_CHANGED quality flag (movement calc 1.1.0, owner approval 2026-11-23).

An entry paired with a close taken under a different kickoff is flagged, kept in the detail file,
and left out of every movement/CLV statistic. The version bump is recorded with old vs new counts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src import microstructure as ms  # noqa: E402
from src import movement as mv  # noqa: E402
from src import research_state as rs  # noqa: E402

ENTRY = {"fixture_id": "F1", "snapshot_ts": "2026-08-20T10:00:00Z", "kickoff_ts": "2026-08-20 20:00:00",
         "p_model_over": 0.60, "v11_p_market": 0.52, "odds_over25": 2.00, "odds_under25": 2.00,
         "minutes_to_kickoff": 600.0, "n_books": 8, "league": "L", "model_type": "standard"}
CLOSE = {"fixture_id": "F1", "snapshot_ts": "2026-08-20T18:00:00Z", "v11_p_market": 0.55,
         "odds_over25": 1.85, "odds_under25": 2.20, "minutes_to_kickoff": 120.0}


def _run(close_mtk: float, kickoff: str | None = "2026-08-20 20:00:00") -> pd.DataFrame:
    e = dict(ENTRY)
    if kickoff is None:
        e.pop("kickoff_ts")
    else:
        e["kickoff_ts"] = kickoff
    return mv.compute(pd.DataFrame([e]), pd.DataFrame([{**CLOSE, "minutes_to_kickoff": close_mtk}]))


@pytest.mark.parametrize("close_mtk,flag", [(120.0, "OK"), (125.0, "OK"), (125.05, "KICKOFF_CHANGED"),
                                            (114.9, "KICKOFF_CHANGED"), (60.0, "KICKOFF_CHANGED"),
                                            (300.0, "KICKOFF_CHANGED")])
def test_flag_uses_a_strict_five_minute_tolerance(close_mtk, flag):
    d = _run(close_mtk)
    assert d["clv_quality"].iloc[0] == flag
    assert bool(d["kickoff_changed"].iloc[0]) == (flag == "KICKOFF_CHANGED")


def test_flagged_row_is_kept_but_not_eligible():
    d = _run(60.0)
    assert len(d) == 1 and d["kickoff_shift_min"].iloc[0] == pytest.approx(-60.0)
    assert mv.eligible(d).empty


def test_close_after_the_rows_own_kickoff_is_flagged():
    d = _run(120.0, kickoff="2026-08-20 17:00:00")
    assert d["clv_quality"].iloc[0] == "KICKOFF_CHANGED"


def test_missing_kickoff_is_not_assessed_never_passed():
    d = _run(60.0, kickoff=None)
    assert pd.isna(d["kickoff_changed"].iloc[0])
    assert "KICKOFF_CHANGED" in d["quality_not_assessed"].iloc[0]


def test_versions_bumped_together():
    assert mv.CALC_VERSION == "1.1.0"
    assert rs.CALC_VERSION == mv.CALC_VERSION          # Pro: V11_CALC_VERSION_MISMATCH otherwise
    assert ms.CALC_VERSION == "1.1.0"
    assert mv.KICKOFF_CHECK is True


def test_restatement_record_has_old_and_new_counts():
    rec = PROJ / "output" / "corrections" / "v11_movement_kickoff_changed_2026-11-23.csv"
    r = pd.read_csv(rec).set_index("metric")
    assert int(r.loc["kickoff_changed_rows", "calc_1.1.0"]) == 1829
    assert int(r.loc["kickoff_changed_fixtures", "calc_1.1.0"]) == 64
    assert int(r.loc["kickoff_changed_rows_ok_under_1.0.0", "calc_1.1.0"]) == 1761
    assert int(r.loc["kickoff_changed_rows_close_after_row_kickoff", "calc_1.1.0"]) == 401
    assert int(r.loc["detail_rows", "calc_1.0.0"]) == int(r.loc["detail_rows", "calc_1.1.0"])
    assert int(r.loc["clv_quality=KICKOFF_CHANGED", "calc_1.0.0"]) == 0
    assert (int(r.loc["eligible_rows", "calc_1.0.0"]) - int(r.loc["eligible_rows", "calc_1.1.0"])
            == 1761)
