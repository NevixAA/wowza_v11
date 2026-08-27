"""
Market-microstructure research pipeline (brief sections 2-8, 18-21).
====================================================================

Produces ONE durable detail dataset — `output/v11_market_microstructure.csv` — plus the segment
tables the research questions need, rather than a scatter of disconnected CSVs.

WHAT IT DOES

1. Derives backward-looking microstructure per snapshot (`src/microstructure.py`).
2. Joins it onto the existing forward outcomes in `v11_market_movement_detail.csv`
   (signed move to close, clean CLV, toward_wowza) by `snapshot_id`.
3. Reduces to ONE OBSERVATION PER FIXTURE at several entry points, because the raw table holds a
   median of 74 snapshots per fixture and treating those as independent inflates n ~74x.
4. Scores every segment against PLACEBO baselines that carry no model input.

WHY THE PLACEBO IS THE POINT

The headline is that the market moves toward Wowza ~58% of the time. `movement.placebo_toward_rate`
asks a deliberately stupider question: did the market move toward a FIXED anchor — the median
market probability — which involves no model at all? On this sample the answer is 58.33% against
Wowza's 58.00%. The model-free control matches the model.

The mechanism is measurable rather than speculative: the model's probabilities have 0.564x the
standard deviation of the market's, so the model is systematically more central, and
corr(residual, distance-to-anchor) is +0.833 with 83% sign agreement. "The market moved toward
Wowza" and "the market moved toward the middle" are largely the same sentence on this data.

Every table therefore reports the placebo beside the headline. A segment is only interesting where
it beats its own placebo, and that is a much higher bar than beating 50%.

WHAT IS NOT ATTEMPTED

No model is trained (brief sections 16 and 30). Fixture-level n is ~300 and the graduation
threshold is 500 clean observations minimum, 1000+ preferred. The logistic here exists only to
ask whether the residual carries information ONCE the mean-reversion term is in the equation; it
is a diagnostic, not a predictor, and it is not persisted or used for any decision.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

import src.microstructure as ms          # noqa: E402
import src.movement as mv                # noqa: E402

OUT = PROJ / "output"
SNAPS = OUT / "v11_shadow_snapshots.csv"
DETAIL = OUT / "v11_market_movement_detail.csv"
MICRO_CSV = OUT / "v11_market_microstructure.csv"
SEG_CSV = OUT / "v11_microstructure_segments.csv"
COV_CSV = OUT / "v11_microstructure_coverage.csv"

SEED = 20260827

# Entry points to evaluate. None = earliest eligible snapshot, which is where the published
# headline comes from and, as it turns out, the only place the effect appears at all.
ENTRIES = [(None, "earliest"), (1440, "~24h"), (720, "~12h"),
           (360, "~6h"), (180, "~3h"), (60, "~1h")]

MICRO_PREFIXES = ("velocity_", "largest_move_", "range_", "window_span_", "move_from_open",
                  "last_move", "n_price_changes", "reversal_count", "direction_changes",
                  "time_since_last_change", "market_acceleration", "opening_probability",
                  "previous_probability", "snapshot_index", "n_polls_seen",
                  "minutes_since_previous", "quality_flags", "quality_status")


def build() -> pd.DataFrame:
    """Snapshot-level microstructure joined to forward outcomes."""
    snaps = pd.read_csv(SNAPS, low_memory=False)
    micro = ms.compute(snaps)
    det = pd.read_csv(DETAIL, low_memory=False)
    # v11_market_movement_detail.csv carries a duplicated `p_market_entry` column (written twice
    # by the detail builder). Dropped here rather than upstream so this script stays read-only
    # with respect to the existing pipeline.
    det = det.loc[:, ~det.columns.duplicated()]

    cols = ["snapshot_id"] + [c for c in micro.columns
                              if c.startswith(MICRO_PREFIXES) and c != "snapshot_id"]
    j = det.merge(micro[cols], on="snapshot_id", how="left")

    j["trend_alignment"] = ms.trend_alignment(j["residual"] * 100.0, j["move_from_open_pp"])
    # Pure mean-reversion term: distance from the crowd centre, carrying NO model input. This is
    # the quantity the placebo uses and the one the residual turns out to be 0.83 correlated with.
    anchor = float(pd.to_numeric(j["p_market_entry"], errors="coerce").median())
    j["anchor_probability"] = anchor
    j["reversion_pp"] = (anchor - pd.to_numeric(j["p_market_entry"], errors="coerce")) * 100.0
    return j


def _placebos(f: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Three null baselines (brief section 21). Deterministic given SEED."""
    move = pd.to_numeric(f["market_move_pp"], errors="coerce").values
    resid = pd.to_numeric(f["residual"], errors="coerce").values
    anchor_rate, _ = mv.placebo_toward_rate(f)
    shuffled = rng.permutation(resid)
    return {
        "placebo_anchor": anchor_rate,
        "placebo_shuffled": float(np.mean(np.sign(move) == np.sign(shuffled))),
        "placebo_flipped": float(np.mean(np.sign(move) == np.sign(-resid))),
    }


def _row(f: pd.DataFrame, segment: str, unit: str, rng) -> dict:
    tw = pd.to_numeric(f["toward_wowza"], errors="coerce").dropna()
    if tw.empty:
        return {}
    lo, hi = mv.wilson(int(tw.sum()), len(tw))
    clv = pd.to_numeric(f["clv_pct"], errors="coerce")
    sm = pd.to_numeric(f["signed_market_move_pp"], errors="coerce")
    clo, chi = mv.cluster_bootstrap_mean(clv, f["fixture_id"], seed=SEED)
    p = _placebos(f, rng)
    return {
        "segment": segment, "unit": unit, "n_fixtures": int(f["fixture_id"].nunique()),
        "n": int(len(tw)),
        "toward_pct": round(100 * float(tw.mean()), 2),
        "toward_ci_lo": round(100 * lo, 2), "toward_ci_hi": round(100 * hi, 2),
        "placebo_anchor_pct": round(100 * p["placebo_anchor"], 2),
        "edge_vs_placebo_pp": round(100 * (float(tw.mean()) - p["placebo_anchor"]), 2),
        "placebo_shuffled_pct": round(100 * p["placebo_shuffled"], 2),
        "placebo_flipped_pct": round(100 * p["placebo_flipped"], 2),
        "mean_signed_move_pp": round(float(sm.mean()), 3),
        "median_signed_move_pp": round(float(sm.median()), 3),
        "mean_clv_pct": round(float(clv.mean()), 3),
        "median_clv_pct": round(float(clv.median()), 3),
        "clv_ci_lo": round(clo, 3), "clv_ci_hi": round(chi, 3),
        "positive_clv_pct": round(100 * float((clv > 0).mean()), 2),
        "median_abs_residual_pp": round(float(pd.to_numeric(
            f["abs_residual_pp"], errors="coerce").median()), 2),
        "sample_status": mv.sample_status(int(f["fixture_id"].nunique())),
    }


def segments(j: pd.DataFrame) -> pd.DataFrame:
    el = mv.eligible(j)
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []

    for at, lbl in ENTRIES:
        f = mv.fixture_level(el, at_minutes=at)
        f = f[pd.to_numeric(f["toward_wowza"], errors="coerce").notna()]
        if len(f) < 20:
            continue
        r = _row(f, f"entry={lbl}", "fixture", rng)
        if r:
            rows.append(r)

    # Everything below is measured at the earliest entry, which is the only point where the
    # headline exists, so the segmentations are asked of the same sample that produced it.
    base = mv.fixture_level(el)
    base = base[pd.to_numeric(base["toward_wowza"], errors="coerce").notna()]

    for col, pretty in (("model_type", "model"), ("trend_alignment", "trend"),
                        ("abs_residual_band", "abs_residual")):
        if col not in base.columns:
            continue
        for val, g in base.groupby(col):
            if len(g) < 20:                     # below this a rate is not worth printing
                continue
            r = _row(g, f"{pretty}={val}", "fixture", rng)
            if r:
                rows.append(r)

    # Dispersion in observed terciles rather than invented cutoffs (brief section 5).
    disp = pd.to_numeric(base.get("market_prob_std"), errors="coerce")
    if disp.notna().sum() >= 60:
        q = disp.quantile([1 / 3, 2 / 3]).values
        band = pd.cut(disp, [-np.inf, q[0], q[1], np.inf],
                      labels=["tight", "mid", "dispersed"])
        for val, g in base.groupby(band, observed=True):
            if len(g) < 20:
                continue
            r = _row(g, f"dispersion={val}", "fixture", rng)
            if r:
                rows.append(r)
    return pd.DataFrame(rows)


def residual_information(j: pd.DataFrame) -> pd.DataFrame:
    """Does the residual say anything ONCE mean reversion is accounted for? (sections 20, 19-Q10)

    A diagnostic, not a model. Reported as coefficients with z-scores so the answer is legible
    either way; nothing here is persisted for prediction or used to select a threshold.
    """
    el = mv.eligible(j)
    f = mv.fixture_level(el)
    f = f[pd.to_numeric(f["toward_wowza"], errors="coerce").notna()].copy()
    y = pd.to_numeric(f["toward_wowza"], errors="coerce").astype(float).values

    def z(v):
        v = pd.to_numeric(v, errors="coerce").astype(float)
        sd = v.std()
        return ((v - v.mean()) / sd).fillna(0).values if sd and np.isfinite(sd) else np.zeros(len(v))

    # Coverage of each candidate term ON THIS SAMPLE, checked before it is allowed into a
    # specification. Mean-filling a column that is 99% missing produces a near-constant
    # regressor, and the fit then returns a huge coefficient with an astronomical standard
    # error — the first run of this reported velocity_3h at beta=+27.07, se=13588, which is
    # not a weak result but an undefined one. velocity_3h is 2/300 at the earliest entry
    # precisely BECAUSE the earliest snapshot has no three hours of history behind it.
    _MIN_TERM_COVERAGE = 0.60
    raw = {
        "residual_pp": f["residual"] * 100.0,
        "reversion_pp": f["reversion_pp"],
        "log_window_h": np.log1p(pd.to_numeric(f["window_min"], errors="coerce") / 60.0),
        "dispersion": f.get("market_prob_std"),
        "velocity_3h": f.get("velocity_3h"),
    }
    cover = {k: float(pd.to_numeric(v, errors="coerce").notna().mean()) for k, v in raw.items()}
    terms = {k: z(v) for k, v in raw.items()}
    out = []
    for spec in (["residual_pp"], ["reversion_pp"], ["residual_pp", "reversion_pp"],
                 ["residual_pp", "reversion_pp", "log_window_h", "dispersion"],
                 ["residual_pp", "reversion_pp", "log_window_h", "dispersion", "velocity_3h"]):
        thin = [c for c in spec if cover.get(c, 0.0) < _MIN_TERM_COVERAGE]
        if thin:
            # COMPLETE CASES instead of mean-filling. Restricting the sample loses rows and says
            # so; filling keeps every row and quietly changes what the coefficient means. The
            # reduced n is reported in the `n` column so the trade is visible.
            mask = np.ones(len(y), dtype=bool)
            for c in spec:
                mask &= pd.to_numeric(raw[c], errors="coerce").notna().values
            if mask.sum() < 60:
                out.append({"spec": " + ".join(spec), "n": int(mask.sum()),
                            "term": "SKIPPED (complete cases < 60: "
                                    + ", ".join(f"{c}={100*cover[c]:.0f}%" for c in thin) + ")",
                            "beta": None, "se": None, "z": None,
                            "significant_95": False, "log_lik": None})
                continue
            sub_y = y[mask]
            A = np.column_stack([np.ones(int(mask.sum()))]
                                + [z(pd.Series(raw[c]).reset_index(drop=True)[mask])
                                   for c in spec])
            out.extend(_fit(A, sub_y, spec, note="complete-cases"))
            continue
        A = np.column_stack([np.ones(len(y))] + [terms[c] for c in spec])
        out.extend(_fit(A, y, spec, note="full"))
    return pd.DataFrame(out)


def _fit(A: np.ndarray, y: np.ndarray, spec: list[str], *, note: str) -> list[dict]:
    """IRLS logistic with Wald standard errors. Shared by both sample paths so a full-sample
    and a complete-case fit can never differ by implementation."""
    b = np.zeros(A.shape[1])
    for _ in range(200):
        p = 1.0 / (1.0 + np.exp(-A @ b))
        W = p * (1 - p) + 1e-9
        try:
            b = b + np.linalg.solve((A * W[:, None]).T @ A + 1e-6 * np.eye(A.shape[1]),
                                    A.T @ (y - p))
        except np.linalg.LinAlgError:
            break
    p = np.clip(1.0 / (1.0 + np.exp(-A @ b)), 1e-9, 1 - 1e-9)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    W = p * (1 - p)
    se = np.sqrt(np.diag(np.linalg.pinv((A * W[:, None]).T @ A)))
    rows = []
    for nm, coef, s in zip(["intercept"] + spec, b, se):
        rows.append({"spec": " + ".join(spec) + (f" [{note}]" if note != "full" else ""),
                     "n": int(len(y)), "term": nm,
                     "beta": round(float(coef), 4), "se": round(float(s), 4),
                     "z": round(float(coef / s) if s else np.nan, 2),
                     "significant_95": bool(abs(coef / s) > 1.96) if s else False,
                     "log_lik": round(ll, 2)})
    return rows


def _stamp(d: pd.DataFrame) -> pd.DataFrame:
    """Stamp `generated_at` as a COLUMN, following scripts/v11_grade.py.

    Not left to the file mtime: `git checkout` resets mtime, so on a fresh CI runner every
    artifact looks seconds old and `research_state`'s staleness check can never fire. The six
    movement outputs still rely on the mtime fallback and are exposed to exactly that — worth
    fixing there too, but it changes six schemas and is left alone here rather than bundled in.
    """
    out = d.copy()
    out["generated_at"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    return out


def main() -> int:
    j = build()
    _stamp(j).to_csv(MICRO_CSV, index=False)
    print(f"[micro] {MICRO_CSV.name}: {len(j):,} rows x {len(j.columns)} cols")

    cov = ms.coverage(j)
    _stamp(cov).to_csv(COV_CSV, index=False)
    print(f"[micro] {COV_CSV.name}: coverage for {len(cov)} features")

    seg = segments(j)
    reg = residual_information(j)
    _stamp(seg).to_csv(SEG_CSV, index=False)
    print(f"[micro] {SEG_CSV.name}: {len(seg)} segments\n")

    show = ["segment", "n", "toward_pct", "toward_ci_lo", "toward_ci_hi",
            "placebo_anchor_pct", "edge_vs_placebo_pp", "mean_clv_pct", "sample_status"]
    print(seg[show].to_string(index=False))
    print("\nDoes the residual add information beyond mean reversion?")
    print(reg[reg["term"] != "intercept"].to_string(index=False))
    print("\nRead `edge_vs_placebo_pp`, not `toward_pct`. The placebo carries no model input; "
          "\nbeating 50% is not evidence, beating the placebo would be.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
