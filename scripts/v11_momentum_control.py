"""
Does the Wowza residual predict market movement AFTER controlling for momentum?
==============================================================================
    python scripts/v11_momentum_control.py [--write]

Prompt 02 section 2. This is the experiment the whole prospective-validation phase exists for,
and it is deliberately a REGRESSION rather than another stratified table, because the tables
cannot answer the one objection that matters.

THE OBJECTION

`v11_market_movement.py` reports a fixture-level toward-Wowza rate of **57.9%** (n=465,
p=0.00225). Read alone that says the market drifts our way. But a market that is ALREADY moving
will keep moving, and our residual is computed against the price at the same moment — so if we
tend to disagree with a price that is mid-drift, we will look predictive while contributing
nothing. Momentum is not a nuisance here; it is the whole alternative explanation.

Note the existing evidence already hints at this: the same statistic measured snapshot-level with
fixture-clustered intervals falls to 54.2% with p=0.095. Fixture-level counts each fixture once
and is the honest unit; snapshot-level with clustering is the conservative one. The truth being
somewhere between them is exactly why a control is needed rather than a louder headline.

THE MODEL

    future_move_pp ~ residual_pp
                   + prev_move_pp        (how far it had already moved)
                   + velocity_pp_h       (how fast, recently)
                   + acceleration_pp_h2  (whether that rate is rising)
                   + minutes_to_kickoff  (moves get bigger near the close)
                   + dispersion          (how much the books disagree)

Ordinary least squares, fitted with `np.linalg.lstsq`. No regularisation, no feature selection, no
ML — section 2 says transparent first, and a coefficient nobody can read cannot be argued with.

The coefficient on `residual_pp` is the answer. If it survives with the momentum terms in the
model, the residual carries information the market had not yet priced. If it collapses toward
zero, the 57.9% was momentum wearing our name.

INFERENCE IS CLUSTERED BY FIXTURE

One fixture contributes many snapshots and they are anything but independent — consecutive
observations of one drifting price are nearly the same observation. Treating them as independent
would shrink every interval by roughly sqrt(snapshots per fixture) and manufacture significance.
So the bootstrap resamples FIXTURES with replacement, refits, and takes percentile intervals.

WHAT IS NOT CLAIMED

An OLS coefficient is not a profit. It says the residual is associated with subsequent movement
once momentum is accounted for; it does not say a price was available, that it could be bet, or
that CLV net of vig is positive. Those are separate questions with their own evidence.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

import config  # noqa: E402
from src import movement as mv  # noqa: E402

CALC_VERSION = "1.0.0"

# Prior windows (minutes) over which "how far had it already moved" is measured. Section 2 asks
# for 15m..24h; the collector runs twice an hour, so anything under 30m is mostly empty and is
# kept only so the emptiness is visible rather than assumed away.
PRIOR_WINDOWS = (15, 30, 60, 180, 360, 720, 1440)

# Future targets (minutes ahead). Section 2's list.
FUTURE_WINDOWS = (30, 60, 180)

# How close a snapshot must be to the requested offset to count as that observation. Without a
# tolerance, merge_asof happily matches a "30 minutes ago" price to one from six hours ago and
# reports a momentum term computed from the wrong window.
TOLERANCE_MIN = {15: 12, 30: 20, 60: 35, 180: 90, 360: 150, 720: 300, 1440: 480}

# Below this, a coefficient is reported but never interpreted. Section 10's discipline applied to
# regression rather than to bet counts.
MIN_FIXTURES = 100

PRED_COLS = ["residual_pp", "prev_move_pp", "velocity_pp_h", "acceleration_pp_h2",
             "hours_to_kickoff", "dispersion"]

# Prompt 2 section 9's fuller control set. Kept as a SEPARATE list rather than replacing the one
# above, so the two models are reported side by side and the marginal effect of each block of
# controls is visible instead of inferred.
PRED_COLS_FULL = PRED_COLS + ["move_30m", "move_3h", "velocity_1h", "direction_streak",
                              "dispersion_change", "n_books", "current_probability"]

# Below this, a residual is smaller than the market's own tick and cannot lead anything. Used to
# separate WOWZA_NEUTRAL from WOWZA_LEADS (section 11) — without it, quiet-market rows with a
# trivial residual land in LEADS and inflate the one bucket the price-discovery claim rests on.
MIN_RESIDUAL_PP = 2.0


def _load() -> pd.DataFrame:
    p = config.OUTPUT_DIR / "v11_shadow_snapshots.csv"
    if not p.exists():
        raise SystemExit(f"no snapshots at {p}")
    d = pd.read_csv(p, low_memory=False)
    d["snapshot_ts"] = pd.to_datetime(d["snapshot_ts"], errors="coerce", utc=True)
    d["kickoff_ts"] = pd.to_datetime(d["kickoff_ts"], errors="coerce", utc=True)
    d = d[d["snapshot_ts"].notna() & d["fixture_id"].notna()]

    # PRE-KICKOFF ONLY. A post-kickoff price is a different market — the in-play one — and mixing
    # it in would let a goal masquerade as price discovery.
    if "is_post_kickoff" in d.columns:
        d = d[~d["is_post_kickoff"].astype(str).str.lower().isin(("true", "1"))]
    if "minutes_to_kickoff" in d.columns:
        d = d[pd.to_numeric(d["minutes_to_kickoff"], errors="coerce") > 0]

    # Only rows where both sides of the comparison exist. A residual needs a model probability
    # AND a market probability; either missing makes the row uninterpretable, not zero.
    for c in ("p_model_over", "v11_p_market"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d["p_model_over"].notna() & d["v11_p_market"].notna()]
    if "valid_odds" in d.columns:
        d = d[d["valid_odds"].astype(str).str.lower().isin(("true", "1"))]
    return d.sort_values(["fixture_id", "snapshot_ts"]).reset_index(drop=True)


def _asof_col(d: pd.DataFrame, col: str, minutes: float) -> pd.Series:
    """Value of `col` `minutes` before each row, per fixture. Used for dispersion change."""
    if col not in d.columns:
        return pd.Series(np.nan, index=d.index)
    tol = pd.Timedelta(minutes=TOLERANCE_MIN.get(int(minutes), max(10, minutes * 0.4)))
    left = d[["fixture_id", "snapshot_ts"]].copy()
    left["target_ts"] = left["snapshot_ts"] - pd.Timedelta(minutes=minutes)
    right = d[["fixture_id", "snapshot_ts", col]].rename(
        columns={"snapshot_ts": "match_ts", col: "_v"})
    right["_v"] = pd.to_numeric(right["_v"], errors="coerce")
    out = pd.merge_asof(left.sort_values("target_ts"), right.sort_values("match_ts"),
                        left_on="target_ts", right_on="match_ts", by="fixture_id",
                        direction="nearest", tolerance=tol)
    return out.sort_index()["_v"]


def _asof(d: pd.DataFrame, minutes: float, direction: str) -> pd.Series:
    """Market probability `minutes` before (direction='backward') or after ('forward') each row,
    matched within tolerance, per fixture."""
    tol = pd.Timedelta(minutes=TOLERANCE_MIN.get(int(minutes), max(10, minutes * 0.4)))
    left = d[["fixture_id", "snapshot_ts"]].copy()
    offset = pd.Timedelta(minutes=minutes)
    left["target_ts"] = (left["snapshot_ts"] - offset if direction == "backward"
                         else left["snapshot_ts"] + offset)
    right = d[["fixture_id", "snapshot_ts", "v11_p_market"]].rename(
        columns={"snapshot_ts": "match_ts", "v11_p_market": "p_at"})
    out = pd.merge_asof(
        left.sort_values("target_ts"), right.sort_values("match_ts"),
        left_on="target_ts", right_on="match_ts", by="fixture_id",
        direction="nearest", tolerance=tol)
    return out.sort_index()["p_at"]


def build(prior_window: int = 60, future_window: int = 60) -> pd.DataFrame:
    d = _load()
    if d.empty:
        return d

    d["residual_pp"] = (d["p_model_over"] - d["v11_p_market"]) * 100.0
    d["hours_to_kickoff"] = pd.to_numeric(d["minutes_to_kickoff"], errors="coerce") / 60.0
    d["dispersion"] = pd.to_numeric(d.get("book_dispersion"), errors="coerce")

    # --- momentum: where the price WAS -------------------------------------------------
    p_prev = _asof(d, prior_window, "backward")
    d["prev_move_pp"] = (d["v11_p_market"] - p_prev) * 100.0

    # VELOCITY MUST USE A SHORTER WINDOW THAN prev_move, or it is not a second variable at all.
    # The first version set velocity = prev_move / (window/60), which is prev_move times a
    # constant — perfectly collinear with it. lstsq does not fail on that, it silently SPLITS the
    # coefficient between the two, and the run printed prev_move_pp and velocity_pp_h with
    # identical values (-0.5197 each), which is how the bug announced itself.
    #
    # Recent velocity is now the rate over the most recent short window, which is genuinely
    # distinct information: "it moved 2pp over the hour" and "most of that was in the last ten
    # minutes" are different states of the market.
    short = max(15, prior_window // 4)
    p_short = _asof(d, short, "backward")
    d["velocity_pp_h"] = ((d["v11_p_market"] - p_short) * 100.0) / (short / 60.0)

    # Acceleration needs a SECOND, earlier window: the rate over [-2w, -w] compared with the rate
    # over [-w, now]. Differencing the velocity column by row index instead would be wrong —
    # rows are unevenly spaced, so consecutive rows are not a fixed time apart.
    p_prev2 = _asof(d, prior_window * 2, "backward")
    recent_rate = d["prev_move_pp"] / (prior_window / 60.0)
    earlier_rate = ((p_prev - p_prev2) * 100.0) / (prior_window / 60.0)
    d["acceleration_pp_h2"] = (recent_rate - earlier_rate) / (prior_window / 60.0)

    # --- the target: where the price WENT ----------------------------------------------
    p_next = _asof(d, future_window, "forward")
    d["future_move_pp"] = (p_next - d["v11_p_market"]) * 100.0

    # SIGNED TOWARD THE RESIDUAL. A raw move is meaningless without a direction: +0.5pp is
    # movement our way when the residual is positive and against us when it is negative.
    d["future_move_toward_pp"] = np.where(d["residual_pp"] >= 0,
                                          d["future_move_pp"], -d["future_move_pp"])

    # --- Prompt 2 section 10: the full prior-movement feature set -----------------------
    # Every one of these uses ONLY information available at t. The `backward` direction in
    # `_asof` is what enforces that; a forward-looking window here would leak the answer into
    # the predictors and produce a spectacular, meaningless result.
    for w, name in ((10, "move_10m"), (30, "move_30m"), (60, "move_1h"), (180, "move_3h")):
        d[name] = (d["v11_p_market"] - _asof(d, w, "backward")) * 100.0
    for w, name in ((30, "velocity_30m"), (60, "velocity_1h"), (180, "velocity_3h")):
        d[name] = d[f"move_{'30m' if w == 30 else ('1h' if w == 60 else '3h')}"] / (w / 60.0)

    # DIRECTION STREAK: how many of the recent windows agree in sign. A market drifting one way
    # for three hours is a different state from one that jittered to the same place, and pooling
    # them is how momentum gets under-controlled.
    signs = np.sign(d[["move_30m", "move_1h", "move_3h"]].fillna(0.0))
    d["direction_streak"] = signs.sum(axis=1)

    # Dispersion CHANGE, not level: books converging and books scattering are opposite states
    # even at identical dispersion.
    d["dispersion_change"] = d["dispersion"] - _asof_col(d, "book_dispersion", prior_window)
    d["n_books"] = pd.to_numeric(d.get("n_books"), errors="coerce")
    d["current_probability"] = d["v11_p_market"]

    # --- section 11's FOUR-way classification ------------------------------------------
    # Four, not three: NEUTRAL was missing and it is not a rounding detail. A residual smaller
    # than the market's own tick size cannot lead anything, so pooling those rows into LEADS
    # inflated the one bucket the whole price-discovery claim rests on. On the 2026-08-30 run
    # WOWZA_LEADS held 414 observations; most of them were quiet-market rows with a residual too
    # small to act on.
    same_sign = np.sign(d["prev_move_pp"]) == np.sign(d["residual_pp"])
    quiet = d["prev_move_pp"].abs() < mv.MIN_MOVE_PP
    small_resid = d["residual_pp"].abs() < MIN_RESIDUAL_PP
    d["wowza_role"] = np.select(
        [d["prev_move_pp"].isna(), small_resid, quiet, same_sign],
        ["UNKNOWN_NO_PRIOR", "WOWZA_NEUTRAL", "WOWZA_LEADS",
         "WOWZA_AGREES_WITH_EXISTING_MOVE"],
        default="WOWZA_OPPOSES_MARKET")

    d["prior_window_min"] = prior_window
    d["future_window_min"] = future_window
    d["calc_version"] = CALC_VERSION
    return d


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def _design(d: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    sub = d.dropna(subset=cols + ["future_move_pp"])
    X = np.column_stack([np.ones(len(sub))] + [sub[c].to_numpy(float) for c in cols])
    return X, sub["future_move_pp"].to_numpy(float), sub["fixture_id"]


def fit(d: pd.DataFrame, cols: list[str], *, n_boot: int = 1000,
        seed: int = 11) -> pd.DataFrame:
    """OLS with fixture-clustered bootstrap intervals."""
    X, y, clusters = _design(d, cols)
    if len(y) < 30:
        return pd.DataFrame()
    beta = _ols(X, y)

    rng = np.random.default_rng(seed)
    uniq = clusters.unique()
    idx_by_fixture = {f: np.flatnonzero((clusters == f).to_numpy()) for f in uniq}
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_fixture[f] for f in pick])
        try:
            boots.append(_ols(X[rows], y[rows]))
        except np.linalg.LinAlgError:
            continue
    B = np.array(boots)

    names = ["intercept"] + cols
    lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
    # Fraction of bootstrap draws on the opposite side of zero, doubled: a two-sided bootstrap
    # p-value that needs no normality assumption.
    p = 2 * np.minimum((B > 0).mean(axis=0), (B < 0).mean(axis=0))
    return pd.DataFrame({
        "term": names, "coef": beta, "ci_lo": lo, "ci_hi": hi, "p_value": p,
        "n_obs": len(y), "n_fixtures": len(uniq),
        "excludes_zero": (lo > 0) | (hi < 0),
        "sample_status": mv.sample_status(len(uniq)),
        "calc_version": CALC_VERSION,
    })


def artifact_diagnostic(d: pd.DataFrame, *, n_boot: int = 400) -> pd.DataFrame:
    """Is the headline an artefact of p_market appearing on both sides of the regression?

    THE PROBLEM, which applies to the uncontrolled AND the controlled model:

        residual_pp    =  p_model  -  p_market(t)
        prev_move_pp   =  p_market(t) - p_market(t-w)
        future_move_pp =  p_market(t+w) - p_market(t)

    `p_market(t)` appears in all three, and with OPPOSITE signs between residual and
    future_move. So if p_market(t) is measured with any noise — and it is, being de-vigged from a
    handful of books that quote at different instants — that noise alone produces a POSITIVE
    residual coefficient and a NEGATIVE prev_move coefficient. Exactly the pattern the first run
    reported (+1.20 and -0.52). Neither number is evidence of anything until this is ruled out.

    THE TEST: a placebo residual. Replace p_model with a value that cannot possibly know anything
    — the model probability from a DIFFERENT, randomly chosen fixture — keeping p_market(t)
    exactly where it is. Any real forecasting power disappears; the shared-price artefact does
    not, because p_market(t) is untouched.

        placebo coefficient ~= 0    -> the artefact is small, the real coefficient means something
        placebo coefficient ~= real -> the headline is arithmetic, not price discovery

    This is the single most important control in the script and it is cheap. Without it, a
    perfectly mechanical result reads as a discovery.
    """
    sub = d.dropna(subset=["residual_pp", "future_move_pp", "v11_p_market", "p_model_over"]).copy()
    if len(sub) < 100:
        return pd.DataFrame()
    rng = np.random.default_rng(7)
    rows = []

    real = fit(sub, ["residual_pp"], n_boot=n_boot)
    if not real.empty:
        r = real[real["term"] == "residual_pp"].iloc[0]
        rows.append({"variant": "real residual", "coef": r["coef"], "ci_lo": r["ci_lo"],
                     "ci_hi": r["ci_hi"], "p_value": r["p_value"], "n_obs": r["n_obs"],
                     "n_fixtures": r["n_fixtures"]})

    # Shuffle the MODEL probability across fixtures, leaving the market price in place.
    shuffled = sub.copy()
    shuffled["p_model_over"] = rng.permutation(sub["p_model_over"].to_numpy())
    shuffled["residual_pp"] = (shuffled["p_model_over"] - shuffled["v11_p_market"]) * 100.0
    plc = fit(shuffled, ["residual_pp"], n_boot=n_boot)
    if not plc.empty:
        r = plc[plc["term"] == "residual_pp"].iloc[0]
        rows.append({"variant": "PLACEBO shuffled model prob", "coef": r["coef"],
                     "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "p_value": r["p_value"],
                     "n_obs": r["n_obs"], "n_fixtures": r["n_fixtures"]})

    # And the purest form: drop the model entirely and regress the future move on the CURRENT
    # PRICE. If this alone is strongly negative, the price mean-reverts against its own
    # measurement error and every coefficient above inherits it.
    sub["neg_p_market_pp"] = -sub["v11_p_market"] * 100.0
    only = fit(sub, ["neg_p_market_pp"], n_boot=n_boot)
    if not only.empty:
        r = only[only["term"] == "neg_p_market_pp"].iloc[0]
        rows.append({"variant": "price only (no model at all)", "coef": r["coef"],
                     "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "p_value": r["p_value"],
                     "n_obs": r["n_obs"], "n_fixtures": r["n_fixtures"]})

    out = pd.DataFrame(rows)
    if not out.empty:
        out["calc_version"] = CALC_VERSION
    return out


def placebo_table(d: pd.DataFrame, *, n_boot: int = 400, seed: int = 5) -> pd.DataFrame:
    """Prompt 2 section 12: score the V9 residual against every alternative predictor.

    THE COMPARISON THAT DECIDES THE PHASE. A toward-rate on its own cannot distinguish skill from
    arithmetic, because several predictors that contain no football information at all point the
    same way as the residual most of the time:

      FIXED ANCHOR is the important one, and v11's own `movement.placebo_toward_rate` docstring
      already explains why: the model's probabilities are systematically more CENTRAL than the
      market's, so "moved toward the model" and "moved toward the middle" are frequently the
      same sentence. Ordinary mean reversion in a noisy price then reproduces the headline with
      no skill whatsoever. Prompt 2 section 8 reports this placebo at ~57.8% against a
      toward-Wowza rate of ~57.1% — the placebo is BETTER.

      MEAN REVERSION (-prev_move) is the same idea expressed as a direction rather than a level.

      SHUFFLED RESIDUAL keeps the market price exactly where it is and attaches a model
      probability drawn from a different fixture, so it isolates the part of the coefficient
      produced by p_market(t) sitting on both sides of the regression.

    Every variant is scored the SAME way on the SAME rows, so the numbers are comparable. A
    residual that cannot beat these is not evidence of price discovery, whatever its p-value.
    """
    need = ["residual_pp", "future_move_pp", "v11_p_market", "prev_move_pp"]
    sub = d.dropna(subset=need).copy()
    if len(sub) < 100:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)

    anchor = float(sub["v11_p_market"].median())
    variants = {
        # The real thing.
        "v9_residual": sub["residual_pp"],
        # Section 12's controls, in order of how much they should worry us.
        "PLACEBO_fixed_anchor": (anchor - sub["v11_p_market"]) * 100.0,
        "PLACEBO_shuffled_residual": (pd.Series(rng.permutation(sub["p_model_over"].to_numpy()),
                                                index=sub.index)
                                      - sub["v11_p_market"]) * 100.0,
        "PLACEBO_mean_reversion": -sub["prev_move_pp"],
        "PLACEBO_momentum_continuation": sub["prev_move_pp"],
        "PLACEBO_random_direction": pd.Series(rng.normal(size=len(sub)), index=sub.index),
        "CONTROL_market_only": -sub["v11_p_market"] * 100.0,
        "CONTROL_dispersion": sub["dispersion"].fillna(0.0),
    }
    if "velocity_3h" in sub.columns:
        variants["PLACEBO_long_momentum"] = sub["velocity_3h"].fillna(0.0)

    rows = []
    for name, pred in variants.items():
        s = sub.assign(_pred=pd.to_numeric(pred, errors="coerce")).dropna(subset=["_pred"])
        if len(s) < 50:
            continue
        # DROP the real residual before renaming the variant onto its name. Renaming straight on
        # top produced TWO columns called residual_pp, so `sub[c]` in the design matrix returned
        # a 2-column frame instead of a Series and np.column_stack built a ragged array.
        s2 = s.drop(columns=["residual_pp"]).rename(columns={"_pred": "residual_pp"})
        f = fit(s2, ["residual_pp"], n_boot=n_boot)
        if f.empty:
            continue
        r = f[f["term"] == "residual_pp"].iloc[0]
        # Toward-rate, scored identically for every variant: did the market move in the direction
        # this predictor pointed?
        toward = float((np.sign(s["future_move_pp"]) == np.sign(s["_pred"])).mean())
        rows.append({
            "variant": name, "coef": r["coef"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
            "p_value": r["p_value"], "toward_rate": round(toward, 4),
            "n_obs": int(len(s)), "n_fixtures": int(s["fixture_id"].nunique()),
            "excludes_zero": bool(r["excludes_zero"]),
            "sample_status": mv.sample_status(int(s["fixture_id"].nunique())),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    real = out[out["variant"] == "v9_residual"]
    if len(real):
        rt = float(real["toward_rate"].iloc[0])
        out["toward_rate_vs_v9_pp"] = ((out["toward_rate"] - rt) * 100).round(2)
        # BEATS_V9 is the column to read first. Any placebo above zero here is a predictor with
        # no football information that points the right way more often than the model does.
        out["beats_v9"] = out["toward_rate"] > rt
    out["calc_version"] = CALC_VERSION
    return out.sort_values("toward_rate", ascending=False).reset_index(drop=True)


def chronological(d: pd.DataFrame, *, folds: int = 4, n_boot: int = 200) -> pd.DataFrame:
    """Prompt 2 section 13: expanding-window evaluation, never a random split.

    Market data is time-dependent and every fixture's snapshots are adjacent in time, so a random
    split trains on the same afternoon it tests on. Each fold here fits on everything before a cut
    and evaluates after it, which is the only split that answers "would this have held going
    forward".
    """
    sub = d.dropna(subset=PRED_COLS + ["future_move_pp"]).sort_values("snapshot_ts")
    if len(sub) < 400:
        return pd.DataFrame()
    cuts = pd.qcut(sub["snapshot_ts"].rank(method="first"), folds + 1,
                   labels=False, duplicates="drop")
    rows = []
    for k in range(1, folds + 1):
        train, test = sub[cuts < k], sub[cuts == k]
        if len(train) < 200 or len(test) < 100:
            continue
        f = fit(test, PRED_COLS, n_boot=n_boot)
        if f.empty:
            continue
        r = f[f["term"] == "residual_pp"].iloc[0]
        rows.append({"fold": k, "train_rows": int(len(train)), "test_rows": int(len(test)),
                     "test_from": str(test["snapshot_ts"].min()),
                     "test_to": str(test["snapshot_ts"].max()),
                     "residual_coef": r["coef"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                     "p_value": r["p_value"], "excludes_zero": bool(r["excludes_zero"]),
                     "n_fixtures": int(test["fixture_id"].nunique())})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["calc_version"] = CALC_VERSION
    return out


def run(prior: int, future: int, *, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = build(prior, future)
    if d.empty:
        return d, pd.DataFrame()

    rows = []
    # UNIVARIATE FIRST, then the controlled model. Reporting only the controlled coefficient
    # would hide how much of the raw association the controls actually absorb, which is the
    # number the reader wants.
    for label, cols in (("residual only", ["residual_pp"]),
                        ("+ momentum", ["residual_pp", "prev_move_pp", "velocity_pp_h",
                                        "acceleration_pp_h2"]),
                        ("+ full controls", PRED_COLS),
                        ("+ section 9 controls", PRED_COLS_FULL)):
        f = fit(d, cols, n_boot=n_boot)
        if f.empty:
            continue
        f.insert(0, "model", label)
        rows.append(f)
    coefs = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not coefs.empty:
        coefs.insert(0, "future_window_min", future)
        coefs.insert(0, "prior_window_min", prior)

    # Role breakdown — leadership vs agreement, never pooled.
    by_role = (d.dropna(subset=["future_move_toward_pp"])
                .groupby("wowza_role")
                .agg(n_obs=("future_move_toward_pp", "size"),
                     n_fixtures=("fixture_id", "nunique"),
                     mean_toward_pp=("future_move_toward_pp", "mean"),
                     median_toward_pp=("future_move_toward_pp", "median"),
                     pct_moved_toward=("future_move_toward_pp", lambda s: float((s > 0).mean())))
                .reset_index())
    if not by_role.empty:
        by_role["sample_status"] = by_role["n_fixtures"].map(mv.sample_status)
        by_role["prior_window_min"] = prior
        by_role["future_window_min"] = future
    return coefs, by_role


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", type=int, default=60)
    ap.add_argument("--future", type=int, default=60)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--all-windows", action="store_true",
                    help="sweep every prior x future pair")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    pairs = ([(p, f) for p in PRIOR_WINDOWS for f in FUTURE_WINDOWS] if a.all_windows
             else [(a.prior, a.future)])

    all_coefs, all_roles = [], []
    for prior, future in pairs:
        coefs, roles = run(prior, future, n_boot=a.boot)
        if coefs.empty:
            print(f"[momentum] prior={prior}m future={future}m — too few usable rows")
            continue
        all_coefs.append(coefs)
        all_roles.append(roles)

        n_fix = int(coefs["n_fixtures"].iloc[0])
        print(f"\n=== prior {prior}m -> future {future}m "
              f"(n={int(coefs['n_obs'].iloc[0]):,} snapshots, {n_fix} fixtures, "
              f"{mv.sample_status(n_fix)}) ===")
        for model in coefs["model"].unique():
            sub = coefs[coefs["model"] == model]
            r = sub[sub["term"] == "residual_pp"].iloc[0]
            star = "*" if r["excludes_zero"] else " "
            print(f"  {model:18} residual coef {r['coef']:+.4f}{star} "
                  f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  p={r['p_value']:.3f}")
        full = coefs[coefs["model"] == "+ full controls"]
        for _, r in full.iterrows():
            if r["term"] in ("intercept", "residual_pp"):
                continue
            star = "*" if r["excludes_zero"] else " "
            print(f"      {r['term']:20} {r['coef']:+.4f}{star} "
                  f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]")

        if not roles.empty:
            print("  role breakdown:")
            for _, r in roles.iterrows():
                print(f"      {r['wowza_role']:32} n={r['n_obs']:>6,} "
                      f"({r['n_fixtures']:>3} fx)  toward {r['pct_moved_toward']:.1%}  "
                      f"mean {r['mean_toward_pp']:+.3f}pp  [{r['sample_status']}]")

    if not all_coefs:
        return 1

    coefs = pd.concat(all_coefs, ignore_index=True)
    roles = pd.concat(all_roles, ignore_index=True)

    # THE HEADLINE, stated once and plainly.
    base = coefs[(coefs["model"] == "residual only") & (coefs["term"] == "residual_pp")]
    ctrl = coefs[(coefs["model"] == "+ full controls") & (coefs["term"] == "residual_pp")]
    if len(base) and len(ctrl):
        b, c = float(base["coef"].mean()), float(ctrl["coef"].mean())
        survives = bool(ctrl["excludes_zero"].all())
        shrink = (1 - c / b) * 100 if b else float("nan")
        print(f"\n{'=' * 78}")
        print(f"residual coefficient  uncontrolled {b:+.4f}  ->  controlled {c:+.4f}  "
              f"({shrink:.0f}% absorbed by momentum)")
        print(f"survives controls at 95%: {'YES' if survives else 'NO'}")
        if int(ctrl["n_fixtures"].min()) < MIN_FIXTURES:
            print(f"NOT INTERPRETABLE YET: {int(ctrl['n_fixtures'].min())} fixtures, "
                  f"minimum {MIN_FIXTURES}. Reported so the number exists, not so it is believed.")
        print("=" * 78)

    # The artefact control, run on the primary window only — it answers whether ANY of the above
    # is interpretable, so it is printed last and read first.
    diag = artifact_diagnostic(build(a.prior, a.future), n_boot=min(a.boot, 400))
    if not diag.empty:
        print("\nARTEFACT CONTROL — is p_market(t) on both sides producing this?")
        for _, r in diag.iterrows():
            print(f"  {r['variant']:30} coef {r['coef']:+.4f} "
                  f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  p={r['p_value']:.3f}")
        real = diag[diag["variant"] == "real residual"]
        plc = diag[diag["variant"] == "PLACEBO shuffled model prob"]
        if len(real) and len(plc):
            rc, pc = float(real["coef"].iloc[0]), float(plc["coef"].iloc[0])
            share = (pc / rc * 100) if rc else float("nan")
            print(f"  -> the placebo reproduces {share:.0f}% of the real coefficient")
            if abs(share) > 70:
                print("  -> READ THE UNCONTROLLED HEADLINE AS ARITHMETIC, NOT AS PRICE "
                      "DISCOVERY: a model probability that cannot know anything scores "
                      "almost the same.")

    # ---- Prompt 2 sections 12 and 13 --------------------------------------------------
    primary = build(a.prior, a.future)
    plc = placebo_table(primary, n_boot=min(a.boot, 400))
    chrono = chronological(primary, n_boot=min(a.boot, 200))

    if not plc.empty:
        print("\nPLACEBO / CONTROL TABLE (section 12) — sorted by toward-rate")
        print(f"  {'variant':32} {'toward':>7} {'coef':>9} {'p':>6} {'vs v9':>7}  n_fx")
        for _, r in plc.iterrows():
            flag = "  <-- BEATS v9" if r.get("beats_v9") and r["variant"] != "v9_residual" else ""
            print(f"  {r['variant']:32} {r['toward_rate']:>7.3f} {r['coef']:>+9.4f} "
                  f"{r['p_value']:>6.3f} {r.get('toward_rate_vs_v9_pp', 0):>+7.2f} "
                  f"{r['n_fixtures']:>5}{flag}")
        beaten = plc[(plc["variant"] != "v9_residual") & (plc.get("beats_v9", False))]
        if len(beaten):
            print(f"  -> {len(beaten)} control(s) with NO football information point the right "
                  f"way more often than the model. The toward-rate is not evidence of price "
                  f"discovery.")

    if not chrono.empty:
        print("\nCHRONOLOGICAL FOLDS (section 13) — expanding window, never a random split")
        for _, r in chrono.iterrows():
            star = "*" if r["excludes_zero"] else " "
            print(f"  fold {int(r['fold'])}  test {str(r['test_from'])[:10]}.."
                  f"{str(r['test_to'])[:10]}  n_fx {int(r['n_fixtures']):>4}  "
                  f"residual {r['residual_coef']:>+.4f}{star} p={r['p_value']:.3f}")
        n_sig = int(chrono["excludes_zero"].sum())
        print(f"  -> significant in {n_sig} of {len(chrono)} folds")

    if a.write:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        coefs.to_csv(config.OUTPUT_DIR / "v11_momentum_control.csv",
                     index=False, encoding="utf-8")
        roles.to_csv(config.OUTPUT_DIR / "v11_momentum_roles.csv",
                     index=False, encoding="utf-8")
        wrote = ["v11_momentum_control.csv", "v11_momentum_roles.csv"]
        # Extending the existing two files with two more rather than one wide one: the grains
        # differ (one row per variant vs one per fold), and section 53 asks not to sprawl.
        if not plc.empty:
            plc.to_csv(config.OUTPUT_DIR / "v11_placebo_table.csv", index=False,
                       encoding="utf-8")
            wrote.append("v11_placebo_table.csv")
        if not chrono.empty:
            chrono.to_csv(config.OUTPUT_DIR / "v11_chronological_folds.csv", index=False,
                          encoding="utf-8")
            wrote.append("v11_chronological_folds.csv")
        print(f"[momentum] wrote {', '.join(wrote)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
