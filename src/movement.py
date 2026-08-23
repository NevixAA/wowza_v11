"""
Market-movement measurement: does Wowza's disagreement predict where the price goes?
====================================================================================
RESEARCH ONLY. Nothing here selects a bet, sizes a stake, or feeds a notifier.

THE HYPOTHESIS. v11's residual test says Wowza does not improve on the market at predicting
match OUTCOMES (Brier 0.2272 vs 0.2273 — a dead heat). That is not the same claim as "Wowza
knows nothing". It may identify information EARLIER than the broad market, in which case its
disagreement predicts where the price moves before kickoff even though the closing price
ultimately absorbs the same information. The value proposition would then be "Wowza sometimes
reaches the future market price first", not "Wowza beats bookmakers".

    residual            = p_model - p_market_entry
    market_move         = p_market_close - p_market_entry
    signed_market_move  = market_move * sign(residual)
    toward_wowza        = signed_market_move > 0

────────────────────────────────────────────────────────────────────────────────
THE STATISTICAL TRAP THAT GOVERNS THIS WHOLE MODULE

There are 9,971 snapshots across 336 fixtures — a mean of 29.7 per fixture. Every snapshot of
one fixture shares the same model opinion, the same closing price and the same market; they are
not 9,971 independent observations of anything. Treating them as independent inflates n by ~30x
and shrinks every confidence interval by ~5.5x, which would turn a null result into an
overwhelming discovery on arithmetic alone.

So this module reports TWO units of analysis and never blurs them:

  * FIXTURE-LEVEL (primary). One observation per fixture. n is small and honest, and it is the
    number that governs any claim about whether the effect is real.
  * SNAPSHOT-LEVEL (secondary). Every eligible snapshot, for the durable dataset and for
    entry-timing questions that only exist within a fixture (does a T-12h residual behave
    differently from a T-30m one?). Confidence intervals here are CLUSTER BOOTSTRAPPED by
    fixture — resampling fixtures, not rows — so the interval reflects 336 independent things.

`unit=` is a required argument on the summary functions rather than a default, so a caller has
to state which one they mean.

────────────────────────────────────────────────────────────────────────────────
AN IDENTITY WORTH KNOWING BEFORE READING THE OUTPUTS

`p_market` here is the de-vigged consensus probability of OVER. If the model prefers OVER
(residual > 0) our side is OVER and the fair probability of our side is p_market; if it prefers
UNDER our side is UNDER and the fair probability of our side is 1 - p_market. So:

    side OVER :  clv_prob = p_close       - p_entry       =  market_move
    side UNDER:  clv_prob = (1-p_close)   - (1-p_entry)   = -market_move

    => clv_prob == market_move * sign(residual) == signed_market_move

**Signed market movement IS closing-line value, measured in fair-probability space.** They are
one measurement, not two independent confirmations, and a report that presents them as two
agreeing results is double-counting. They diverge only when CLV is computed from actual
EXECUTABLE odds, which carry vig and book-specific pricing — that is `clv_pct_price` below, and
it is the one that would decide whether any of this is monetisable.

────────────────────────────────────────────────────────────────────────────────
SIGN HANDLING FOR UNDER — the thing most likely to invalidate the research

Both p_model and p_market are OVER probabilities, so the residual sign encodes which side the
model prefers and the signed-movement algebra is direction-agnostic: model says UNDER
(residual < 0) and the over-price falls (market_move < 0) gives (-)x(-) = + = toward Wowza.
Correct without special-casing.

Executable CLV is NOT direction-agnostic. It must use the odds of the side we would actually
back — odds_under25 when the model prefers UNDER — and using odds_over25 for both sides would
produce a confident, entirely fictitious CLV series. `price_side_odds` is the only place that
selects, and it is unit-tested in both directions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Definition version, archived with every observation Pro ingests (movement brief section 18).
#
# BUMP THIS whenever a definition changes — the residual or signed-move formula, MIN_MOVE_PP, a
# bucket boundary, the eligibility rules, or the side-selection logic. Observations computed under
# different definitions must never be pooled, and without a version stamp they silently would be:
# a later reader would average a v1 and a v2 residual and get a number that means nothing. Pro
# partitions by ingest run, so a bump ADDS a version rather than overwriting the previous one.
CALC_VERSION = "1.0.0"

# Below this the price is treated as UNMOVED rather than as a tiny signal. A flat market is an
# absence of evidence, not evidence against the model, and must not be counted as a miss.
MIN_MOVE_PP = 0.2

# Residual buckets in percentage points (brief section 6).
RESIDUAL_BANDS = [(-1e9, -10, "< -10"), (-10, -6, "-10:-6"), (-6, -4, "-6:-4"),
                  (-4, -2, "-4:-2"), (-2, 2, "-2:+2"), (2, 4, "+2:+4"),
                  (4, 6, "+4:+6"), (6, 10, "+6:+10"), (10, 1e9, "> +10")]

# Absolute-residual buckets (brief section 7).
ABS_BANDS = [(0, 2, "0-2"), (2, 4, "2-4"), (4, 6, "4-6"), (6, 8, "6-8"),
             (8, 10, "8-10"), (10, 1e9, "10+")]

# Time-to-kickoff buckets in MINUTES, ordered far -> near (brief section 8).
TIME_BANDS = [(1440, 1e9, ">24h"), (720, 1440, "12-24h"), (360, 720, "6-12h"),
              (180, 360, "3-6h"), (60, 180, "1-3h"), (30, 60, "30-60m"),
              (10, 30, "10-30m"), (0, 10, "<10m")]

# Sample-discipline labels (brief section 12). Applied to the FIXTURE count, never the snapshot
# count — 24 snapshots of one fixture is one observation of the market, not 24.
def sample_status(n_fixtures: int) -> str:
    if n_fixtures < 50:
        return "INSUFFICIENT_SAMPLE"
    if n_fixtures < 150:
        return "EARLY_SIGNAL"
    if n_fixtures < 500:
        return "RESEARCH"
    return "VALIDATABLE"


# ── quality flags (brief section 19) ────────────────────────────────────────
# Only applied where the data actually supports the judgement. A flag we cannot evaluate is
# recorded in `not_assessed` rather than silently treated as passing, because "we checked and it
# was fine" and "we could not check" are different statements about data quality.
FLAG_OK = "OK"

# Plausible de-vigged overround for a two-way market. Outside this the pair is not a coherent
# O/U 2.5 quote and the mapping is suspect.
_OVERROUND_LO, _OVERROUND_HI = 0.98, 1.25
_MIN_BOOKS = 3


def quality_flags(row: pd.Series) -> tuple[str, list[str]]:
    """(flag, not_assessed). flag is the FIRST disqualifying condition, or OK."""
    flags, not_assessed = [], []

    mtk = row.get("minutes_to_kickoff")
    if pd.isna(mtk):
        flags.append("MISSING_KICKOFF")
    elif float(mtk) <= 0:
        flags.append("POST_KICKOFF_ENTRY")

    if pd.isna(row.get("p_market_close")):
        flags.append("MISSING_CLOSE")
    ctk = row.get("close_minutes_to_kickoff")
    if pd.notna(ctk) and float(ctk) <= 0:
        flags.append("POST_KICKOFF_CLOSE")

    o, u = row.get("odds_over25"), row.get("odds_under25")
    if pd.isna(o) or pd.isna(u):
        flags.append("MISSING_OPPOSITE_SIDE")
    else:
        try:
            s = 1.0 / float(o) + 1.0 / float(u)
            if not (_OVERROUND_LO <= s <= _OVERROUND_HI):
                flags.append("INVALID_MARKET_MAPPING")
        except (TypeError, ValueError, ZeroDivisionError):
            flags.append("INVALID_MARKET_MAPPING")

    nb = row.get("n_books")
    if pd.isna(nb):
        not_assessed.append("INSUFFICIENT_BOOKS")
    elif float(nb) < _MIN_BOOKS:
        flags.append("INSUFFICIENT_BOOKS")

    # STALE_ENTRY_PRICE / STALE_CLOSE_PRICE / SYNTHETIC_ODDS need per-book quote timestamps and
    # a provenance marker that these snapshots do not carry. Inventing a proxy (e.g. calling an
    # unchanged price "stale") would misclassify a genuinely settled market, so they are
    # declared unassessed. The brief is explicit: do not invent quality failures.
    not_assessed += ["STALE_ENTRY_PRICE", "STALE_CLOSE_PRICE", "SYNTHETIC_ODDS"]

    return (flags[0] if flags else FLAG_OK), sorted(set(not_assessed))


# ── core arithmetic ─────────────────────────────────────────────────────────
def band_of(value: float, bands) -> str:
    if pd.isna(value):
        return "unknown"
    for lo, hi, name in bands:
        if lo <= float(value) < hi:
            return name
    return "unknown"


def price_side_odds(residual: float, odds_over: float, odds_under: float) -> tuple[str, float]:
    """(side, odds) for the side the MODEL prefers.

    The only direction-dependent step in the module. residual > 0 means the model thinks OVER is
    underpriced, so the bet — and therefore the price whose CLV we measure — is OVER.
    residual == 0 has no preferred side; returns ("", nan) so it cannot silently become OVER.
    """
    if pd.isna(residual) or residual == 0:
        return "", float("nan")
    if residual > 0:
        return "OVER", (float(odds_over) if pd.notna(odds_over) else float("nan"))
    return "UNDER", (float(odds_under) if pd.notna(odds_under) else float("nan"))


def compute(entry: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """Join each entry snapshot to its fixture's close and derive every movement field.

    `entry` is one row per candidate snapshot; `close` is one row per fixture (the last
    PRE-kickoff snapshot). Rows where entry is not strictly before close are dropped — a
    zero-length window cannot measure movement, and the close row compared with itself would
    contribute a guaranteed movement of exactly 0.
    """
    e = entry.copy()
    c = close[["fixture_id", "snapshot_ts", "v11_p_market", "odds_over25", "odds_under25",
               "minutes_to_kickoff"]].rename(columns={
        "snapshot_ts": "close_ts", "v11_p_market": "p_market_close",
        "odds_over25": "close_odds_over25", "odds_under25": "close_odds_under25",
        "minutes_to_kickoff": "close_minutes_to_kickoff"})
    d = e.merge(c, on="fixture_id", how="left")

    d["entry_ts"] = d["snapshot_ts"]
    d["p_market_entry"] = pd.to_numeric(d["v11_p_market"], errors="coerce")
    d["p_model"] = pd.to_numeric(d["p_model_over"], errors="coerce")
    d["p_market_close"] = pd.to_numeric(d["p_market_close"], errors="coerce")

    d["residual"] = d["p_model"] - d["p_market_entry"]
    d["residual_pp"] = d["residual"] * 100.0
    d["abs_residual_pp"] = d["residual_pp"].abs()

    d["market_move_pp"] = (d["p_market_close"] - d["p_market_entry"]) * 100.0
    d["signed_market_move_pp"] = d["market_move_pp"] * np.sign(d["residual_pp"])

    d["moved"] = d["market_move_pp"].abs() >= MIN_MOVE_PP
    # FLOAT, not bool: an unmoved market is NaN — a third state. Collapsing it to False would
    # score a flat market as the model being wrong.
    d["toward_wowza"] = np.where(d["moved"], (d["signed_market_move_pp"] > 0).astype(float),
                                 np.nan)

    sides = d.apply(lambda r: price_side_odds(r["residual"], r.get("odds_over25"),
                                              r.get("odds_under25")), axis=1)
    d["bet_side"] = [s for s, _ in sides]
    d["entry_odds"] = [o for _, o in sides]
    closes = d.apply(lambda r: price_side_odds(r["residual"], r.get("close_odds_over25"),
                                               r.get("close_odds_under25")), axis=1)
    d["close_odds"] = [o for _, o in closes]

    # Fair probability OF OUR SIDE at entry and close (see the identity in the module docstring).
    is_under = d["bet_side"].eq("UNDER")
    d["entry_fair_probability"] = np.where(is_under, 1 - d["p_market_entry"], d["p_market_entry"])
    d["close_fair_probability"] = np.where(is_under, 1 - d["p_market_close"], d["p_market_close"])

    # Executable CLV from real prices: positive means we took a better price than the close.
    with np.errstate(divide="ignore", invalid="ignore"):
        d["clv_pct"] = (d["entry_odds"] / d["close_odds"] - 1.0) * 100.0
    d.loc[~np.isfinite(d["clv_pct"]), "clv_pct"] = np.nan

    d["minutes_to_kickoff"] = pd.to_numeric(d.get("minutes_to_kickoff"), errors="coerce")
    d["residual_band"] = d["residual_pp"].map(lambda v: band_of(v, RESIDUAL_BANDS))
    d["abs_residual_band"] = d["abs_residual_pp"].map(lambda v: band_of(v, ABS_BANDS))
    d["time_band"] = d["minutes_to_kickoff"].map(lambda v: band_of(v, TIME_BANDS))

    q = d.apply(quality_flags, axis=1)
    d["clv_quality"] = [f for f, _ in q]
    d["quality_not_assessed"] = ["|".join(n) for _, n in q]

    # A genuine later observation is required.
    d = d[d["close_ts"].notna() & (d["close_ts"] > d["entry_ts"])]
    return d


def eligible(d: pd.DataFrame) -> pd.DataFrame:
    """Rows clean enough to carry a movement claim."""
    return d[(d["clv_quality"] == FLAG_OK) & d["signed_market_move_pp"].notna()]


# ── inference ───────────────────────────────────────────────────────────────
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — correct near 0/1 and at small n, unlike the normal approximation."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def cluster_bootstrap_mean(values: pd.Series, clusters: pd.Series, *, n_boot: int = 2000,
                           seed: int = 20260823) -> tuple[float, float]:
    """95% CI for a mean, resampling CLUSTERS (fixtures) rather than rows.

    Snapshots of one fixture are ~30 correlated views of a single market. Resampling rows would
    report the precision of 10,000 independent observations when there are 336.

    `seed` is fixed and passed explicitly because Math.random-style nondeterminism would make
    the committed research outputs unreproducible between runs.
    """
    df = pd.DataFrame({"v": pd.to_numeric(values, errors="coerce"), "c": clusters}).dropna()
    if df.empty:
        return (float("nan"), float("nan"))
    groups = [g["v"].to_numpy() for _, g in df.groupby("c", sort=True)]
    if len(groups) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = np.arange(len(groups))
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(idx, size=len(groups), replace=True)
        means[b] = np.concatenate([groups[i] for i in pick]).mean()
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


@dataclass
class Summary:
    """One segment's evidence. Directional and magnitude results are separate FIELDS because the
    brief's central warning is that a good directional rate can coexist with worthless economics
    — a single verdict string would hide exactly that."""
    segment: str
    unit: str
    n_obs: int
    n_fixtures: int
    n_moved: int
    toward_rate: float
    toward_ci_lo: float
    toward_ci_hi: float
    z_vs_chance: float
    p_value: float
    mean_signed_move_pp: float
    median_signed_move_pp: float
    signed_ci_lo: float
    signed_ci_hi: float
    mean_abs_move_pp: float
    mean_move_when_correct_pp: float
    mean_move_when_wrong_pp: float
    p_move_ge_05pp: float
    p_move_ge_1pp: float
    p_move_ge_2pp: float
    n_clv: int
    mean_clv_pct: float
    median_clv_pct: float
    pct_positive_clv: float
    mean_entry_odds: float
    mean_close_odds: float
    sample_status: str


def _norm_sf(z: float) -> float:
    """Two-sided p-value from a z-score. erfc avoids a scipy dependency."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def summarise(d: pd.DataFrame, segment: str, unit: str) -> Summary:
    """Every statistic the brief asks for, for one segment.

    `unit` must be "fixture" or "snapshot" and is recorded in the output. At snapshot level the
    CI is cluster-bootstrapped by fixture; at fixture level rows are already independent.
    """
    if unit not in ("fixture", "snapshot"):
        raise ValueError(f"unit must be 'fixture' or 'snapshot', got {unit!r}")
    n_fix = int(d["fixture_id"].nunique()) if len(d) else 0
    moved = d[d["moved"].fillna(False)]
    n = len(moved)
    k = int(moved["toward_wowza"].sum()) if n else 0
    rate = (k / n) if n else float("nan")

    # THE DIRECTIONAL INTERVAL MUST BE CLUSTERED TOO.
    #
    # The first version of this function used wilson(k, n) for both units, and the result was a
    # live demonstration of the very trap the module docstring describes: the same data read
    # 54.7% [46.4, 62.8] p=0.27 per fixture and 52.6% [51.3, 53.9] p=0.0001 per snapshot. The
    # second is not a stronger finding on more data, it is the SAME 181 fixtures counted ~30
    # times each, and the interval shrank by the square root of that. Publishing it beside a
    # correctly clustered magnitude CI would have made the direction look established while the
    # magnitude looked uncertain, purely as an artifact of which statistic got the right method.
    #
    # A proportion is the mean of a 0/1 variable, so the same cluster bootstrap applies. z is
    # then derived from the bootstrap interval's own width rather than from sqrt(0.25/n), which
    # assumes independence and is exactly what is not true here.
    if unit == "snapshot" and n:
        lo, hi = cluster_bootstrap_mean(moved["toward_wowza"], moved["fixture_id"])
        se = (hi - lo) / (2 * 1.96) if pd.notna(lo) and hi > lo else float("nan")
    else:
        lo, hi = wilson(k, n)
        se = math.sqrt(0.25 / n) if n else float("nan")
    z = ((rate - 0.5) / se) if (n and pd.notna(se) and se > 0) else float("nan")

    sm = moved["signed_market_move_pp"]
    if unit == "snapshot":
        s_lo, s_hi = cluster_bootstrap_mean(sm, moved["fixture_id"])
    else:
        # Independent rows: ordinary normal interval on the mean.
        if n > 1:
            m, sd = float(sm.mean()), float(sm.std(ddof=1))
            h = 1.96 * sd / math.sqrt(n)
            s_lo, s_hi = m - h, m + h
        else:
            s_lo = s_hi = float("nan")

    correct = sm[sm > 0]
    wrong = sm[sm < 0]
    absmove = moved["market_move_pp"].abs()
    clv = moved["clv_pct"].dropna()

    def _f(x):
        return float(x) if pd.notna(x) else float("nan")

    return Summary(
        segment=segment, unit=unit, n_obs=len(d), n_fixtures=n_fix, n_moved=n,
        toward_rate=round(rate, 4) if n else float("nan"),
        toward_ci_lo=round(lo, 4), toward_ci_hi=round(hi, 4),
        z_vs_chance=round(z, 3) if n else float("nan"),
        p_value=round(_norm_sf(z), 5) if n else float("nan"),
        mean_signed_move_pp=round(_f(sm.mean()), 4) if n else float("nan"),
        median_signed_move_pp=round(_f(sm.median()), 4) if n else float("nan"),
        signed_ci_lo=round(s_lo, 4) if pd.notna(s_lo) else float("nan"),
        signed_ci_hi=round(s_hi, 4) if pd.notna(s_hi) else float("nan"),
        mean_abs_move_pp=round(_f(absmove.mean()), 4) if n else float("nan"),
        mean_move_when_correct_pp=round(_f(correct.mean()), 4) if len(correct) else float("nan"),
        mean_move_when_wrong_pp=round(_f(wrong.mean()), 4) if len(wrong) else float("nan"),
        p_move_ge_05pp=round(float((absmove >= 0.5).mean()), 4) if n else float("nan"),
        p_move_ge_1pp=round(float((absmove >= 1.0).mean()), 4) if n else float("nan"),
        p_move_ge_2pp=round(float((absmove >= 2.0).mean()), 4) if n else float("nan"),
        n_clv=int(len(clv)),
        mean_clv_pct=round(_f(clv.mean()), 4) if len(clv) else float("nan"),
        median_clv_pct=round(_f(clv.median()), 4) if len(clv) else float("nan"),
        pct_positive_clv=round(float((clv > 0).mean()), 4) if len(clv) else float("nan"),
        mean_entry_odds=round(_f(moved["entry_odds"].mean()), 4) if n else float("nan"),
        mean_close_odds=round(_f(moved["close_odds"].mean()), 4) if n else float("nan"),
        sample_status=sample_status(n_fix),
    )


def summarise_by(d: pd.DataFrame, by: str, unit: str, *, prefix: str = "") -> pd.DataFrame:
    """summarise() per group, ordered by the natural band order where one exists."""
    order = None
    if by == "residual_band":
        order = [b[2] for b in RESIDUAL_BANDS]
    elif by == "abs_residual_band":
        order = [b[2] for b in ABS_BANDS]
    elif by == "time_band":
        order = [b[2] for b in TIME_BANDS]
    keys = ([k for k in order if k in set(d[by])] if order
            else sorted(d[by].dropna().astype(str).unique()))
    rows = [summarise(d[d[by].astype(str) == k], f"{prefix}{k}", unit).__dict__ for k in keys]
    return pd.DataFrame(rows)


def fixture_level(d: pd.DataFrame, *, at_minutes: float | None = None) -> pd.DataFrame:
    """One observation per fixture — the primary unit.

    `at_minutes=None` takes the EARLIEST eligible snapshot, which is the honest test of "does an
    early disagreement predict the subsequent path". Passing a value takes the snapshot closest
    to that many minutes before kickoff, for entry-timing comparisons.

    Selecting one row per fixture is what makes the confidence intervals mean anything; see the
    module docstring.
    """
    if d.empty:
        return d
    if at_minutes is None:
        return d.sort_values("entry_ts").drop_duplicates("fixture_id", keep="first")
    t = d.assign(_gap=(pd.to_numeric(d["minutes_to_kickoff"], errors="coerce")
                       - float(at_minutes)).abs())
    return t.sort_values("_gap").drop_duplicates("fixture_id", keep="first").drop(columns="_gap")


def placebo_toward_rate(moved: pd.DataFrame) -> tuple[float, float]:
    """(rate, anchor) for a FIXED anchor carrying no model input.

    Preserved from the original script because it is the control that matters. The model's
    probabilities are systematically more central than the market's, so "moved toward the model"
    and "moved toward the middle" can be the same sentence — ordinary mean reversion in a noisy
    opening price would reproduce the headline with no skill at all. If the market moves toward a
    constant at the same rate, the model has added nothing.
    """
    if moved.empty:
        return (float("nan"), float("nan"))
    anchor = float(moved["p_market_entry"].median())
    resid = anchor - moved["p_market_entry"]
    toward = (np.sign(moved["market_move_pp"]) == np.sign(resid * 100.0)).astype(float)
    return (float(toward.mean()), anchor)
