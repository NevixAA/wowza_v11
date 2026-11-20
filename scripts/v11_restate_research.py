"""
Restate the movement and microstructure tables with the ONE p_market estimator, beside the originals
====================================================================================================
Data Truth round 5 (2026-11-02). v11_market_movement.py and v11_microstructure.py now read the
one-estimator P(OVER) (src/book_consensus.research_snapshots), as v11_momentum_control.py already
did. Their committed tables were built on the published v11_p_market, which switched estimator
between snapshots of one fixture and on some rows held P(UNDER). This script writes, beside the
originals and without touching them:

    output/corrections/v11_movement_restated_<date>.csv
    output/corrections/v11_microstructure_restated_<date>.csv

Each stacks three variants of every table, so the estimator effect is isolated from code changes:

    as_published                   the committed output/*.csv as it stands (not recomputed)
    original_estimator_recomputed  this code, on the published v11_p_market
    one_estimator_restated         this code, on the one-estimator P(OVER)

    python scripts/v11_restate_research.py [--date YYYY-MM-DD]

Reads v11 output files only; no network, no v9 access, writes only under output/corrections/.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
import config  # noqa: E402
from scripts import v11_market_movement as MM  # noqa: E402
from scripts import v11_microstructure as MS  # noqa: E402

MOVEMENT_TABLES = ("v11_movement_summary.csv", "v11_movement_by_residual.csv",
                   "v11_movement_by_model.csv", "v11_movement_by_time.csv",
                   "v11_movement_by_league.csv")
MICRO_TABLES = ("v11_microstructure_segments.csv",)


def _stack(frames: list[tuple[str, str, pd.DataFrame]]) -> pd.DataFrame:
    out = []
    for variant, table, df in frames:
        if df is None or df.empty:
            continue
        d = df.copy()
        d.insert(0, "table", table)
        d.insert(0, "variant", variant)
        out.append(d)
    return pd.concat(out, ignore_index=True, sort=False) if out else pd.DataFrame()


def recompute(one_estimator: bool) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    """(movement tables, microstructure segments, residual-information fit, counts)."""
    with contextlib.redirect_stdout(io.StringIO()):
        detail, e = MM.build(one_estimator)
        mv_tables = MM.report(detail, e) if not detail.empty and not e.empty else {}
        snaps = MS.load_snapshots(one_estimator)
        det = detail.loc[:, ~detail.columns.duplicated()] if not detail.empty else detail
        j = MS.build(snaps=snaps, det=det)
        seg = MS.segments(j)
        reg = MS.residual_information(j)
    counts = {"snapshots_used": len(snaps), "detail_rows": len(detail),
              "eligible_rows": len(e),
              "eligible_fixtures": int(e["fixture_id"].nunique()) if len(e) else 0}
    return mv_tables, seg, reg, counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
    a = ap.parse_args()
    out_dir = config.OUTPUT_DIR / "corrections"
    out_dir.mkdir(parents=True, exist_ok=True)

    published = {t: pd.read_csv(config.OUTPUT_DIR / t, low_memory=False)
                 for t in MOVEMENT_TABLES + MICRO_TABLES if (config.OUTPUT_DIR / t).exists()}
    mv_rows, ms_rows, counts = [], [], []
    for t in MOVEMENT_TABLES:
        mv_rows.append(("as_published", t, published.get(t)))
    for t in MICRO_TABLES:
        ms_rows.append(("as_published", t, published.get(t)))

    for variant, one in (("original_estimator_recomputed", False),
                         ("one_estimator_restated", True)):
        mv_tables, seg, reg, n = recompute(one)
        counts.append({"variant": variant, **n})
        for t in MOVEMENT_TABLES:
            mv_rows.append((variant, t, mv_tables.get(t)))
        ms_rows.append((variant, "v11_microstructure_segments.csv", seg))
        ms_rows.append((variant, "residual_information (printed, not published)", reg))

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    mv_df = _stack(mv_rows).assign(generated_at=stamp)
    ms_df = _stack(ms_rows).assign(generated_at=stamp)
    pm = out_dir / f"v11_movement_restated_{a.date}.csv"
    pmi = out_dir / f"v11_microstructure_restated_{a.date}.csv"
    mv_df.to_csv(pm, index=False)
    ms_df.to_csv(pmi, index=False)
    print(pd.DataFrame(counts).to_string(index=False))
    print(f"[restate-research] wrote {pm.name} ({len(mv_df):,} rows), "
          f"{pmi.name} ({len(ms_df):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
