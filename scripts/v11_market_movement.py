"""
Does the market move TOWARD Wowza when Wowza disagrees with it?
==============================================================
    python scripts/v11_market_movement.py

Prompt 3 section 16. This is arguably the most valuable question in the project, and it was
uncomputable until two days ago: it needs snapshot history with horizons, and v11 was deleting
its history on every run (mean 1.0 snapshots per fixture). There are now ~26 per fixture
spanning T-15min to T-7d.

Why it matters more than accuracy. A model that LEADS the market is worth something even if its
standalone hit rate is mediocre — it is seeing something the price has not absorbed yet, which
is exactly what CLV monetises. A model that FOLLOWS the market is worth nothing however
accurate, because by the time it speaks the price already agrees.

The test:

    residual  = p_model(first) - p_market(first)     the initial disagreement
    movement  = p_market(last) - p_market(first)     where the price actually went

If Wowza leads, movement should share the SIGN of residual: when the model says "higher than
the price", the price should subsequently rise. Measured as:

  * agreement rate — share of fixtures where sign(movement) == sign(residual). 50% is chance.
  * mean movement conditioned on residual direction and band.
  * correlation between residual and movement.

Deliberate cautions:
  * `last` is the last PRE-KICKOFF snapshot via closing_snapshot(). Using the last row would
    admit a post-kickoff price, and a price that has seen part of the match is not a forecast.
  * fixtures whose price never moved at all are reported separately rather than counted as
    disagreement — a flat market is an absence of evidence, not evidence against.
  * every segment carries n, and small ones are labelled. A promising bucket at n=12 is an
    observation, not a discovery.
  * no threshold is tuned here. This measures; it does not select.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
import config  # noqa: E402
from scripts.v11_shadow import closing_snapshot  # noqa: E402

# Residual bands in percentage points, per Prompt 3 section 15.
BANDS = [(-100, -10, "<-10"), (-10, -6, "-10:-6"), (-6, -4, "-6:-4"), (-4, -2, "-4:-2"),
         (-2, 2, "-2:+2"), (2, 4, "+2:+4"), (4, 6, "+4:+6"), (6, 10, "+6:+10"),
         (10, 100, ">+10")]

MIN_MOVE = 0.002        # below this the price is treated as unmoved, not as a tiny signal


def _label(n: int) -> str:
    if n < 30:
        return "INSUFFICIENT_SAMPLE"
    if n < 100:
        return "EARLY_SIGNAL"
    if n < 500:
        return "RESEARCH_ONLY"
    return "VALIDATED"


def _band(pp: float) -> str:
    for lo, hi, name in BANDS:
        if lo <= pp < hi:
            return name
    return "?"


def build() -> pd.DataFrame:
    p = config.OUTPUT_DIR / "v11_shadow_snapshots.csv"
    if not p.exists():
        print("[movement] no snapshot history yet")
        return pd.DataFrame()
    s = pd.read_csv(p)
    for c in ("p_model_over", "v11_p_market", "minutes_to_kickoff"):
        s[c] = pd.to_numeric(s.get(c), errors="coerce")
    s = s[s["p_model_over"].notna() & s["v11_p_market"].notna()]
    if s.empty:
        return pd.DataFrame()

    s = s.sort_values("snapshot_ts")
    first = s.drop_duplicates("fixture_id", keep="first")

    # Last PRE-KICKOFF snapshot. A post-kickoff price has seen part of the match.
    close = closing_snapshot(s)
    if close is None or close.empty:
        print("[movement] no pre-kickoff closing snapshots — cannot measure movement")
        return pd.DataFrame()

    f = first[["fixture_id", "league", "model_type", "p_model_over", "v11_p_market",
               "minutes_to_kickoff", "snapshot_ts"]].rename(
        columns={"p_model_over": "p_model_first", "v11_p_market": "p_market_first",
                 "minutes_to_kickoff": "hrs_first_min", "snapshot_ts": "ts_first"})
    l = close[["fixture_id", "v11_p_market", "minutes_to_kickoff", "snapshot_ts"]].rename(
        columns={"v11_p_market": "p_market_last", "minutes_to_kickoff": "hrs_last_min",
                 "snapshot_ts": "ts_last"})
    d = f.merge(l, on="fixture_id", how="inner")
    d = d[d["ts_last"] > d["ts_first"]]          # need a genuine later observation

    d["residual_pp"] = (d["p_model_first"] - d["p_market_first"]) * 100.0
    d["movement_pp"] = (d["p_market_last"] - d["p_market_first"]) * 100.0
    d["residual_band"] = d["residual_pp"].map(_band)
    d["moved"] = d["movement_pp"].abs() >= MIN_MOVE * 100.0
    # Did the price move the way the model pointed? Kept as FLOAT, not bool, so an unmoved
    # market can be NaN — "no evidence" is a third state and must not collapse into False,
    # which would count a flat market as the model being wrong.
    d["toward_model"] = np.where(
        d["moved"],
        (np.sign(d["movement_pp"]) == np.sign(d["residual_pp"])).astype(float),
        np.nan,
    )
    return d


def report(d: pd.DataFrame) -> pd.DataFrame:
    moved = d[d["moved"]]
    print(f"\n=== market movement vs initial disagreement ===")
    print(f"  fixtures with two comparable snapshots : {len(d)}")
    print(f"  of which the price actually moved      : {len(moved)} "
          f"({100 * len(moved) / max(1, len(d)):.0f}%)")
    print(f"  flat markets (no evidence either way)  : {len(d) - len(moved)}")
    if moved.empty:
        print("  nothing moved — no measurement possible")
        return pd.DataFrame()

    agree = float(moved["toward_model"].mean())
    n = len(moved)
    # Binomial standard error against the 50% null.
    se = (0.25 / n) ** 0.5
    z = (agree - 0.5) / se if se else 0.0
    print(f"\n  price moved TOWARD the model : {agree:.1%}  (n={n}, 50% = chance)")
    print(f"  z vs chance                  : {z:+.2f}   [{_label(n)}]")
    print(f"  mean |movement|              : {moved['movement_pp'].abs().mean():.2f}pp")
    corr = float(moved[["residual_pp", "movement_pp"]].corr().iloc[0, 1])
    print(f"  corr(residual, movement)     : {corr:+.3f}")

    # ── PLACEBO CONTROL: is this just mean reversion? ────────────────────────
    #
    # The model's probabilities are systematically MORE CENTRAL than the market's
    # (p_model spans ~0.34-0.68, p_market ~0.27-0.77). So "the price moved toward the model"
    # and "the price moved toward the middle" can be the same sentence, and ordinary mean
    # reversion in a noisy opening price would produce the headline result with no skill
    # whatsoever.
    #
    # The control replaces p_model with a FIXED anchor that contains no information — the
    # median opening market probability. If the market moves toward that at the same rate, the
    # model has added nothing and the 64% is an artifact of where its numbers sit.
    anchor = float(moved["p_market_first"].median())
    placebo_resid = (anchor - moved["p_market_first"]) * 100.0
    placebo_toward = (np.sign(moved["movement_pp"]) == np.sign(placebo_resid)).astype(float)
    p_rate = float(placebo_toward.mean())
    p_z = (p_rate - 0.5) / se if se else 0.0
    print(f"\n  PLACEBO — price moved toward a FIXED anchor ({anchor:.3f}, no model input)")
    print(f"    toward anchor              : {p_rate:.1%}   z {p_z:+.2f}")
    edge_over_placebo = agree - p_rate
    print(f"    model's excess over placebo: {edge_over_placebo:+.1%}")
    if p_rate >= agree - 0.02:
        print("    -> THE MODEL ADDS NOTHING. A constant anchor does as well, so the headline")
        print("       number is mean reversion in a noisy opening price, not foresight.")
    elif p_rate > 0.5:
        print("    -> the market does mean-revert, so part of the headline is that; the model")
        print("       still beats the constant anchor, which is the part that could be skill")
    else:
        print("    -> no mean reversion in the control; the headline is not explained by it")
    if abs(z) < 2:
        print("  -> NOT distinguishable from chance at this sample size")
    elif agree > 0.5:
        print("  -> the market tends to move toward Wowza; this is the LEADING case and is "
              "what CLV would monetise")
    else:
        print("  -> the market moves AWAY from Wowza; the model is following, or wrong")

    rows = []
    print(f"\n=== by residual band (the sign test, per band) ===")
    for _, name in [(b[2], b[2]) for b in BANDS]:
        g = moved[moved["residual_band"] == name]
        if g.empty:
            continue
        a = float(g["toward_model"].mean())
        rows.append({"segment": f"band {name}", "n": len(g), "toward_model_rate": round(a, 4),
                     "mean_movement_pp": round(float(g["movement_pp"].mean()), 3),
                     "label": _label(len(g))})
        print(f"  {name:>8}  n={len(g):>4}  toward {a:>5.1%}  "
              f"mean move {g['movement_pp'].mean():+6.2f}pp   [{_label(len(g))}]")

    print(f"\n=== by model type ===")
    for mt, g in moved.groupby(moved["model_type"].astype(str)):
        a = float(g["toward_model"].mean())
        rows.append({"segment": f"model_type {mt}", "n": len(g),
                     "toward_model_rate": round(a, 4),
                     "mean_movement_pp": round(float(g["movement_pp"].mean()), 3),
                     "label": _label(len(g))})
        print(f"  {mt:<12} n={len(g):>4}  toward {a:>5.1%}   [{_label(len(g))}]")

    rows.insert(0, {"segment": "overall", "n": n, "toward_model_rate": round(agree, 4),
                    "mean_movement_pp": round(float(moved["movement_pp"].mean()), 3),
                    "label": _label(n), "z_vs_chance": round(z, 3),
                    "corr_residual_movement": round(corr, 4),
                    "flat_markets": int(len(d) - len(moved))})
    return pd.DataFrame(rows)


def main() -> int:
    d = build()
    if d.empty:
        print("[movement] not enough history yet — needs >=2 snapshots per fixture with a "
              "known pre-kickoff horizon")
        return 0
    out = report(d)
    if not out.empty:
        p = config.OUTPUT_DIR / "v11_market_movement.csv"
        out.to_csv(p, index=False)
        print(f"\n[movement] written -> {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
