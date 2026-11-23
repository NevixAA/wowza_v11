"""Old vs new movement counts for the KICKOFF_CHANGED flag (movement calc 1.0.0 -> 1.1.0).

    python scripts/v11_restate_kickoff_changed.py

Builds the movement detail twice from the same committed snapshots: once with the kickoff check
off (the 1.0.0 rules) and once with it on (1.1.0), and writes the comparison BESIDE the outputs:

    output/corrections/v11_movement_kickoff_changed_2026-11-23.csv

No row is deleted in either build: the detail file keeps every flagged row. Only eligibility for
the movement/CLV statistics changes. RESEARCH ONLY; nothing here selects or stakes anything.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
import config  # noqa: E402
from src import movement as mv  # noqa: E402
from scripts import v11_market_movement as vm  # noqa: E402

OUT = config.OUTPUT_DIR / "corrections" / "v11_movement_kickoff_changed_2026-11-23.csv"


def _metrics(detail: pd.DataFrame, e: pd.DataFrame) -> dict:
    m = {"detail_rows": len(detail), "detail_fixtures": detail["fixture_id"].nunique(),
         "eligible_rows": len(e), "eligible_fixtures": e["fixture_id"].nunique()}
    for k, v in detail["clv_quality"].value_counts().items():
        m[f"clv_quality={k}"] = int(v)
    fl = mv.fixture_level(e)
    s = mv.summarise(fl, "overall (fixture-level, earliest entry)", "fixture")
    for k in ("n_fixtures", "n_moved", "toward_rate", "mean_signed_move_pp", "n_clv",
              "mean_clv_pct", "pct_positive_clv"):
        m[f"headline_{k}"] = getattr(s, k, None)
    s2 = mv.summarise(e, "overall (snapshot-level, cluster CI)", "snapshot")
    for k in ("n_obs", "toward_rate", "n_clv", "mean_clv_pct"):
        m[f"snapshot_{k}"] = getattr(s2, k, None)
    return m


def main() -> int:
    mv.KICKOFF_CHECK = False
    old_d, old_e = vm.build()
    mv.KICKOFF_CHECK = True
    new_d, new_e = vm.build()
    if old_d.empty or new_d.empty:
        print("no movement detail could be built")
        return 1
    old, new = _metrics(old_d, old_e), _metrics(new_d, new_e)
    kc = new_d[new_d["kickoff_changed"].fillna(False).astype(bool)]
    ko = pd.to_datetime(kc["kickoff_ts"], utc=True, errors="coerce")
    flagged = {
        "kickoff_changed_rows": len(kc),
        "kickoff_changed_fixtures": kc["fixture_id"].nunique(),
        "kickoff_changed_rows_ok_under_1.0.0": int(
            old_d.loc[kc.index.intersection(old_d.index), "clv_quality"].eq(mv.FLAG_OK).sum()),
        "kickoff_changed_rows_close_after_row_kickoff": int(
            (pd.to_datetime(kc["close_ts"], utc=True, errors="coerce") > ko).sum()),
        "kickoff_shift_min_abs_median": float(kc["kickoff_shift_min"].abs().median()) if len(kc) else None,
        "kickoff_not_assessed_rows": int(new_d["kickoff_changed"].isna().sum()),
    }
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for k in sorted(set(old) | set(new), key=lambda x: (x.startswith("headline") or x.startswith("snapshot"), x)):
        rows.append({"metric": k, "calc_1.0.0": old.get(k, 0 if k.startswith("clv_quality") else None),
                     "calc_1.1.0": new.get(k, 0 if k.startswith("clv_quality") else None)})
    for k, v in flagged.items():
        rows.append({"metric": k, "calc_1.0.0": None, "calc_1.1.0": v})
    df = pd.DataFrame(rows)
    df["generated_at"] = now
    df["source"] = "output/v11_shadow_snapshots.csv (one-estimator research snapshots)"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, lineterminator="\n")
    print(df.to_string(index=False))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
