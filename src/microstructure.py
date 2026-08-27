"""
Market microstructure — backward-looking price context for every snapshot.
=========================================================================

WHAT THIS IS FOR

The prospective result is that when Wowza disagrees with the market, the market subsequently
moves toward Wowza about 58% of the time. The obvious alternative explanation is that Wowza
simply agrees with a move that had ALREADY STARTED, in which case the "signal" is momentum
wearing a model's clothes. Section 20 of the brief is explicit that this must be separated, and
separating it needs the price history that preceded each observation.

So every feature here looks STRICTLY BACKWARD from the snapshot's own timestamp. Nothing reads a
later price, a closing price or a result. That is not a stylistic preference: the whole research
question is whether a quantity known at time T predicts what happens after T, and a single
forward-looking input would make the answer meaningless while still producing a number.

WHAT THE DATA ACTUALLY SUPPORTS — measured, not assumed

Measured over 34,568 snapshots across 521 fixtures:

* Median inter-snapshot gap is **33.6 minutes** (p25 23.9, p75 44.3, p95 64.5).
* Fixtures are observed for a median of **99.9 hours** — about four days before kickoff.
* Only **12.9%** of consecutive snapshot pairs show any price change at all, and the median
  move when one happens is **0.31pp**.

Three consequences, all of which shape this module:

1. **`velocity_30m` is not computable and is not offered.** A 30-minute window is shorter than
   the median sampling gap, so it would be null more often than not and, worse, non-null
   precisely for the unrepresentative fixtures that happened to be sampled densely. The brief
   asks for it; the data cannot support it, so it is reported as unavailable rather than
   silently produced from whatever happened to fall in the window.

2. **Velocity needs a real prior observation inside the window, not an assumed one.** Every
   window returns NULL unless an actual snapshot exists at least `_MIN_SPAN_FRAC` of the way
   back. Without that rule a 6h velocity computed from two points 20 minutes apart would be
   extrapolated by a factor of 18 and would dominate every distribution it entered.

3. **A no-change poll is information, and a missing poll is not.** Because only 13% of pairs
   move, "no price change in the last hour" is the common state and must be distinguished from
   "we did not look". `n_price_changes` and `time_since_last_change_min` are therefore counted
   over observed polls only, and carry the poll count alongside so a low change-count on two
   polls cannot be read as a quiet market.

A NOTE ON WHAT `previous_market_move_pp` ALREADY MEANT

`scripts/v11_market_movement._attach_momentum` defines it as `current - FIRST price of the
fixture`, i.e. move-from-open. That is a legitimate momentum measure and it is kept under the
name it deserves, `move_from_open_pp`. It is NOT the same thing as the brief's `last_move_pp`,
which is the move since the immediately preceding snapshot, and conflating the two would answer
section 20 with the wrong variable. Both are produced here.

DELIBERATELY NOT RE-IMPLEMENTED

Book dispersion (`n_books`, `book_dispersion`) is already computed upstream in
`src/book_consensus.py` and stored on the snapshot rows; it is passed through, not recomputed.
Section 26 of the brief asks specifically that velocity, dispersion and CLV not acquire three
separate definitions across three repos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CALC_VERSION = "1.0.0"

# Lookback windows in minutes. 30m is absent on purpose — see the module docstring.
WINDOWS_MIN = (60, 180, 360)
WINDOW_LABEL = {60: "1h", 180: "3h", 360: "6h"}

# A window is only honoured when a real observation sits at least this far back inside it.
# Without it, two points 20 minutes apart would be extrapolated into a "6h velocity".
_MIN_SPAN_FRAC = 0.5

# A price is treated as changed only beyond this, matching movement.MIN_MOVE_PP so the two
# modules cannot disagree about whether a market moved.
MIN_MOVE_PP = 0.2

# Quality flags (brief section 25). NULL beats a fabricated value; a flag says WHY it is null.
FLAG_OK = "OK"
F_NO_PREV = "MISSING_PREVIOUS_SNAPSHOT"
F_NO_OPEN = "NO_OPENING_PRICE"
F_FEW_BOOKS = "INSUFFICIENT_BOOKS"
F_POST_KO = "POST_KICKOFF"
F_NO_KO = "MISSING_KICKOFF"
F_SPARSE = "SPARSE_HISTORY"
F_BAD_PROB = "INVALID_PROBABILITY"

_MIN_BOOKS = 3


def _need(df: pd.DataFrame, cols: tuple[str, ...]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"microstructure: input is missing {missing}")


def compute(snaps: pd.DataFrame) -> pd.DataFrame:
    """Per-snapshot backward-looking microstructure.

    Expects the v11 shadow snapshot columns: fixture_id, snapshot_ts, v11_p_market, and
    optionally minutes_to_kickoff / n_books / book_dispersion / odds_over25 / odds_under25.

    Returns one row per input snapshot with the derived columns added. Rows are NOT dropped —
    filtering is the caller's decision, and a silently shrinking frame is how a coverage
    problem turns into a confident wrong answer.
    """
    _need(snaps, ("fixture_id", "snapshot_ts", "v11_p_market"))
    d = snaps.copy()
    d["_ts"] = pd.to_datetime(d["snapshot_ts"], errors="coerce", utc=True)
    d["_p"] = pd.to_numeric(d["v11_p_market"], errors="coerce")
    d["_mtk"] = (pd.to_numeric(d["minutes_to_kickoff"], errors="coerce")
                 if "minutes_to_kickoff" in d.columns else np.nan)
    d = d.sort_values(["fixture_id", "_ts"]).reset_index(drop=True)

    g = d.groupby("fixture_id", sort=False)

    # ── Opening and previous price ────────────────────────────────────────────
    d["opening_probability"] = g["_p"].transform("first")
    d["previous_probability"] = g["_p"].shift(1)
    d["_prev_ts"] = g["_ts"].shift(1)
    d["snapshot_index"] = g.cumcount()

    d["move_from_open_pp"] = (d["_p"] - d["opening_probability"]) * 100.0
    d["last_move_pp"] = (d["_p"] - d["previous_probability"]) * 100.0
    d["minutes_since_previous"] = (d["_ts"] - d["_prev_ts"]).dt.total_seconds() / 60.0

    # First snapshot of a fixture: NaN, never 0.0. "The market has not moved" and "we have no
    # earlier observation" are different facts, and coding the second as zero invents a flat
    # market — the same reasoning _attach_momentum already applies.
    first = d["snapshot_index"] == 0
    d.loc[first, ["move_from_open_pp", "last_move_pp"]] = np.nan

    # ── Change counting over OBSERVED polls ──────────────────────────────────
    moved = d["last_move_pp"].abs() >= MIN_MOVE_PP
    d["_moved"] = moved.fillna(False)
    # INCLUSIVE of the move into this snapshot. That move is `p_now - p_previous`, which is
    # fully known at this snapshot's own timestamp, so excluding it does not protect against
    # leakage — it just understates what was observable and puts the count one behind reality.
    # Caught by the unit tests: two supra-threshold moves reported 1, and an up-down-up path
    # reported 1 reversal instead of 2.
    d["n_price_changes"] = g["_moved"].cumsum()
    d["n_polls_seen"] = d["snapshot_index"]

    # Time since the last OBSERVED change. Held at NaN until a change has been seen, so a
    # fixture that has never moved does not masquerade as "changed a long time ago".
    d["_chg_ts"] = d["_ts"].where(d["_moved"])
    d["_last_chg"] = g["_chg_ts"].ffill().shift(0)
    d["_last_chg"] = d.groupby("fixture_id", sort=False)["_chg_ts"].transform(
        lambda s: s.shift(1).ffill())
    d["time_since_last_change_min"] = (d["_ts"] - d["_last_chg"]).dt.total_seconds() / 60.0

    # Direction reversals: how choppy the path has been. Sign of each observed move, ignoring
    # flat polls, compared with the previous non-flat move.
    d["_dir"] = np.sign(d["last_move_pp"].where(d["_moved"]))
    d["_prev_dir"] = g["_dir"].transform(lambda s: s.ffill().shift(1))
    d["_rev"] = ((d["_dir"].notna()) & (d["_prev_dir"].notna())
                 & (d["_dir"] != d["_prev_dir"])).astype(int)
    # Inclusive, for the same reason as n_price_changes above.
    d["reversal_count"] = g["_rev"].cumsum()
    d["direction_changes"] = d["reversal_count"]

    # ── Windowed velocity, largest move, and range ───────────────────────────
    for w in WINDOWS_MIN:
        lbl = WINDOW_LABEL[w]
        vel, largest, rng, used = _window_stats(d, w)
        d[f"velocity_{lbl}"] = vel
        d[f"largest_move_{lbl}_pp"] = largest
        d[f"range_{lbl}_pp"] = rng
        d[f"window_span_{lbl}_min"] = used

    # Acceleration: is the market speeding up? Short-window velocity minus long-window. Needs
    # both, so it is null wherever either is.
    d["market_acceleration"] = d["velocity_1h"] - d["velocity_3h"]

    # 30m is claimed by the brief but cannot be supported (median gap 33.6 min). Present as an
    # explicit all-NaN column so downstream code and the report can see it was considered and
    # found unavailable, rather than quietly missing.
    d["velocity_30m"] = np.nan

    # ── Dispersion: passed through from book_consensus, not recomputed ────────
    d["n_books"] = pd.to_numeric(d.get("n_books"), errors="coerce")
    d["market_prob_std"] = pd.to_numeric(d.get("book_dispersion"), errors="coerce")

    d["quality_status"], d["quality_flags"] = zip(*d.apply(_flags, axis=1))
    d["micro_calc_version"] = CALC_VERSION

    return d.drop(columns=[c for c in d.columns if c.startswith("_")])


def _window_stats(d: pd.DataFrame, window_min: int):
    """(velocity pp/h, largest absolute move, high-low range, realised span) per row.

    Implemented per fixture with a merge_asof-style backward search rather than a rolling
    window, because the snapshots are irregularly spaced — a fixed-row rolling window would
    silently mean different amounts of time on different fixtures.
    """
    vel = np.full(len(d), np.nan)
    largest = np.full(len(d), np.nan)
    rng = np.full(len(d), np.nan)
    span = np.full(len(d), np.nan)
    need = window_min * _MIN_SPAN_FRAC

    for _, idx in d.groupby("fixture_id", sort=False).indices.items():
        ts = d["_ts"].values[idx]
        p = d["_p"].values[idx]
        tmin = ts.astype("datetime64[s]").astype(np.int64) / 60.0
        for j in range(len(idx)):
            lo = tmin[j] - window_min
            k = np.searchsorted(tmin, lo, side="left")
            if k >= j:
                continue                       # no earlier observation inside the window
            realised = tmin[j] - tmin[k]
            span[idx[j]] = realised
            if realised < need or not np.isfinite(p[j]) or not np.isfinite(p[k]):
                continue                       # too short a base to normalise honestly
            vel[idx[j]] = ((p[j] - p[k]) * 100.0) / (realised / 60.0)
            seg = p[k:j + 1]
            seg = seg[np.isfinite(seg)]
            if seg.size >= 2:
                steps = np.abs(np.diff(seg)) * 100.0
                largest[idx[j]] = steps.max() if steps.size else np.nan
                rng[idx[j]] = (seg.max() - seg.min()) * 100.0
    return vel, largest, rng, span


def _flags(r: pd.Series) -> tuple[str, str]:
    flags: list[str] = []
    p = r.get("_p")
    if p is None or not np.isfinite(p) or not (0.0 < float(p) < 1.0):
        flags.append(F_BAD_PROB)
    if r.get("snapshot_index", 0) == 0:
        flags.append(F_NO_PREV)
    if not np.isfinite(r.get("opening_probability", np.nan)):
        flags.append(F_NO_OPEN)
    nb = r.get("n_books")
    if nb is None or not np.isfinite(nb):
        flags.append(F_FEW_BOOKS)          # unknown book count is not a passing grade
    elif float(nb) < _MIN_BOOKS:
        flags.append(F_FEW_BOOKS)
    mtk = r.get("_mtk")
    if mtk is None or not np.isfinite(mtk):
        flags.append(F_NO_KO)
    elif float(mtk) <= 0:
        flags.append(F_POST_KO)
    if float(r.get("n_polls_seen", 0) or 0) < 3:
        flags.append(F_SPARSE)
    return (FLAG_OK if not flags else "FLAGGED", "|".join(flags) if flags else "")


def trend_alignment(residual_pp, prior_move_pp, flat_pp: float = MIN_MOVE_PP) -> pd.Series:
    """Classify each observation against the move that preceded it (brief section 4).

    ALIGNS   — the market was already drifting the way Wowza points (continuation)
    OPPOSES  — the market was drifting away from Wowza (Wowza calls a reversal)
    FLAT     — no meaningful prior move
    UNKNOWN  — no prior observation, so the question cannot be asked of this row

    UNKNOWN is a category rather than a default bucket. Folding no-prior-data into FLAT would
    add rows to the very segment used to argue the effect is not momentum, which is exactly the
    conclusion that must not be reached by accounting choice.
    """
    r = pd.to_numeric(pd.Series(residual_pp), errors="coerce")
    m = pd.to_numeric(pd.Series(prior_move_pp), errors="coerce").reindex(r.index)
    out = pd.Series("UNKNOWN", index=r.index, dtype=object)
    known = r.notna() & m.notna()
    flat = known & (m.abs() < flat_pp)
    same = known & ~flat & (np.sign(m) == np.sign(r))
    opp = known & ~flat & (np.sign(m) != np.sign(r))
    out[flat] = "MARKET_FLAT"
    out[same] = "WOWZA_ALIGNS_WITH_TREND"
    out[opp] = "WOWZA_OPPOSES_TREND"
    return out


def coverage(d: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Per-feature coverage and missingness — the brief asks for this for every new feature."""
    cols = cols or [c for c in d.columns if c.startswith(("velocity_", "largest_move_",
                                                          "range_", "market_", "move_from_open",
                                                          "last_move", "n_price_changes",
                                                          "time_since_last_change",
                                                          "reversal_count", "n_books",
                                                          "window_span_"))]
    rows = []
    n = len(d)
    for c in cols:
        if c not in d.columns:
            continue
        v = pd.to_numeric(d[c], errors="coerce")
        nn = int(v.notna().sum())
        rows.append({"feature": c, "n": n, "non_null": nn,
                     "coverage_pct": round(100.0 * nn / n, 2) if n else 0.0,
                     "missing_pct": round(100.0 * (n - nn) / n, 2) if n else 0.0,
                     "mean": round(float(v.mean()), 4) if nn else None,
                     "p05": round(float(v.quantile(0.05)), 4) if nn else None,
                     "p95": round(float(v.quantile(0.95)), 4) if nn else None})
    return pd.DataFrame(rows)
