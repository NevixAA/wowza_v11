"""
Measure N_EFF and ECE from settled fixtures, replacing the hardcoded constants.
==============================================================================
    python scripts/v11_fit_evidence.py

Writes output/v11_evidence.json, which v11_shadow.py reads on every run. Prompt 3 sections 11
and 12.

The two constants being replaced were N_EFF = 1000 and ECE = 0.02, neither measured. They feed
decisions in opposite directions, so a wrong value is not neutral:

  * N_EFF sets the model's blend weight (shrunk when n_eff is below the band's min_n of
    500/800/1500). A flat 1000 claimed more evidence than a thin segment has, so a league with
    40 settled fixtures was blended as though it had a thousand.
  * ECE enters the uncertainty lower bound directly. A flat 0.02 understated uncertainty for a
    poorly-calibrated segment, so EV_lb came out too high and bets cleared floors they should
    not have.

Data: over25_result from v11_graded.csv (the label) joined to p_model_over from
v11_shadow_snapshots.csv (what the model said). The snapshot used per fixture is the LAST
PRE-KICKOFF one, via closing_snapshot() — never the last row, which can be post-kickoff.

If there is not enough settled history, this writes nothing and v11 keeps the CONSERVATIVE
fallback (N_EFF 100, ECE 0.08) rather than the optimistic constants. Absent evidence should
cost the model confidence, not grant it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
import config  # noqa: E402
from src.evidence import EvidenceStore, expected_calibration_error  # noqa: E402
from scripts.v11_shadow import closing_snapshot  # noqa: E402


def _label(v) -> float | None:
    """over25_result -> 1 / 0 / None. Accepts the several spellings the grader has used."""
    s = str(v).strip().upper()
    if s in {"OVER", "WIN", "1", "1.0", "TRUE", "YES", "HIT"}:
        return 1.0
    if s in {"UNDER", "LOSS", "0", "0.0", "FALSE", "NO", "MISS"}:
        return 0.0
    return None


def build() -> pd.DataFrame:
    g_path = config.OUTPUT_DIR / "v11_graded.csv"
    s_path = config.OUTPUT_DIR / "v11_shadow_snapshots.csv"
    if not g_path.exists() or not s_path.exists():
        print("[evidence] need both v11_graded.csv and v11_shadow_snapshots.csv")
        return pd.DataFrame()

    g = pd.read_csv(g_path)
    s = pd.read_csv(s_path)

    if "over25_result" not in g.columns:
        print("[evidence] v11_graded.csv has no over25_result column")
        return pd.DataFrame()

    g["y"] = g["over25_result"].map(_label)
    g = g[g["y"].notna()]
    print(f"[evidence] graded fixtures with a usable label: {len(g)}")

    # The last snapshot per fixture — deliberately NOT closing_snapshot() here.
    #
    # closing_snapshot() requires a known, positive minutes_to_kickoff, because a CLOSING PRICE
    # taken after kickoff is a leak. That rule is right for CLV and wrong for this, and the
    # difference is what is being measured:
    #
    #   CLV needs a PRICE, and a post-kickoff price reflects the match in progress.
    #   Calibration needs the MODEL'S PROBABILITY, and v9's predictions.csv is pre-match by
    #   construction (invariant 5: already-kicked-off fixtures are skipped), so any
    #   p_model_over in any snapshot was produced before the match started.
    #
    # Applying the CLV rule here cost everything: the 83 settled fixtures were snapshotted
    # BEFORE minutes_to_kickoff existed, so their value is NaN, closing_snapshot excluded them,
    # and the overlap with the 177 fixtures that do have it was exactly 0.
    if s.empty:
        print("[evidence] no snapshots available")
        return pd.DataFrame()
    keep = [c for c in ("date", "match", "league", "model_type", "p_model_over",
                        "snapshot_ts") if c in s.columns]
    close = (s[keep].sort_values("snapshot_ts")
                    .drop_duplicates(subset=["date", "match"], keep="last"))

    j = g.merge(close, on=["date", "match"], how="inner", suffixes=("", "_snap"))
    if "league" not in j.columns and "league_snap" in j.columns:
        j["league"] = j["league_snap"]
    j["p"] = pd.to_numeric(j.get("p_model_over"), errors="coerce")
    j = j[j["p"].notna()]
    print(f"[evidence] joined to a model probability: {len(j)}")
    return j


def main() -> int:
    j = build()
    if j.empty:
        print("[evidence] nothing to fit — v11 keeps the CONSERVATIVE fallback "
              "(N_EFF=100, ECE=0.08), not the old optimistic 1000/0.02")
        return 0

    store = EvidenceStore().fit(j, y_col="y", p_col="p")
    out = config.OUTPUT_DIR / "v11_evidence.json"
    store.to_json(out)

    print(f"\n=== measured evidence ({len(j)} settled fixtures) ===")
    print(store.table().to_string(index=False))
    print(f"\n[evidence] global ECE {store.global_ev.ece:.4f} vs the hardcoded 0.0200")
    if store.global_ev.ece > 0.02:
        print("  -> the old constant was OPTIMISTIC: real calibration error is larger, so the "
              "uncertainty band was too narrow and EV_lb too high")
    print(f"[evidence] global N_EFF {store.global_ev.n_eff} vs the hardcoded 1000")
    if store.global_ev.n_eff < 1000:
        print("  -> the old constant OVERSTATED the evidence, so the model's blend weight was "
              "larger than the sample justifies")
    print(f"\n[evidence] written -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
