"""
Does the market move TOWARD Wowza when Wowza disagrees with it?
==============================================================
    python scripts/v11_market_movement.py

RESEARCH ONLY. Produces measurements. Selects nothing, sizes nothing, notifies nothing.

THE HYPOTHESIS (movement brief section 24). v11's residual test says Wowza does not beat the
market at predicting OUTCOMES — Brier 0.2272 vs 0.2273, a dead heat. That is not the same as
"Wowza knows nothing". It may reach information EARLIER than the broad market, in which case its
disagreement predicts where the price goes before kickoff even though the closing price
eventually absorbs the same information. The claim would be "Wowza sometimes gets to the future
market price first", not "Wowza beats bookmakers".

The arithmetic and, more importantly, the two statistical traps that govern how these numbers
must be read live in src/movement.py. In brief:

  1. PSEUDO-REPLICATION. 29.7 snapshots per fixture are not 29.7 independent observations.
     Fixture-level results are primary; snapshot-level intervals are cluster-bootstrapped.
  2. SIGNED MOVEMENT *IS* FAIR-PROBABILITY CLV. They are one measurement, not two agreeing
     ones. Only `clv_pct`, computed from executable odds, is independent of it.

Kept from the first version of this script because they are the controls that matter:

  * the PLACEBO — the market moving toward a fixed, information-free anchor. The model's
    probabilities sit more centrally than the market's, so "moved toward the model" and "moved
    toward the middle" can be the same sentence, and plain mean reversion would reproduce the
    headline with no skill at all.
  * flat markets as a THIRD state, never as a miss.
  * every segment carries n, and small ones are labelled from the FIXTURE count.

Outputs (brief section 17):
    output/v11_market_movement_detail.csv   one row per eligible snapshot — the durable record
    output/v11_movement_summary.csv         headline, placebo, chronological stability
    output/v11_movement_by_residual.csv     signed and absolute residual buckets
    output/v11_movement_by_model.csv        standard vs new_format
    output/v11_movement_by_time.csv         time-to-kickoff buckets
    output/v11_movement_by_league.csv       per league, with sample-discipline status
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
import config  # noqa: E402
from src import movement as mv  # noqa: E402
from scripts.v11_shadow import closing_snapshot  # noqa: E402

DETAIL_COLS = [
    "fixture_id", "snapshot_id", "entry_ts", "kickoff_ts", "minutes_to_kickoff",
    "league", "model_type",
    "p_model", "p_market_entry", "residual", "abs_residual_pp", "entry_odds", "bet_side",
    "n_books", "market_prob_std", "market_prob_range", "previous_market_move_pp",
    "close_ts", "close_minutes_to_kickoff", "window_min", "p_market_close", "close_odds",
    "market_move_pp", "signed_market_move_pp", "toward_wowza",
    "entry_fair_probability", "close_fair_probability",
    "clv_pct", "clv_quality", "quality_not_assessed",
    "residual_band", "abs_residual_band", "time_band",
    "result", "pnl_flat",
]


def _load_snapshots() -> pd.DataFrame:
    p = config.OUTPUT_DIR / "v11_shadow_snapshots.csv"
    if not p.exists():
        return pd.DataFrame()
    s = pd.read_csv(p)
    for c in ("p_model_over", "v11_p_market", "minutes_to_kickoff", "odds_over25",
              "odds_under25", "n_books", "book_dispersion"):
        if c in s.columns:
            s[c] = pd.to_numeric(s[c], errors="coerce")
    return s.sort_values("snapshot_ts")


def _attach_momentum(s: pd.DataFrame) -> pd.DataFrame:
    """previous_market_move_pp — how far the price had ALREADY moved before this snapshot.

    Brief section 11: is Wowza predicting continuation, predicting reversal, or contributing
    something independent? Answering that needs the move that preceded the observation, which
    only exists from the second snapshot of a fixture onward — the first is NaN rather than 0,
    because "the market had not moved yet" and "we have no prior observation" are different
    facts and coding the second as 0.0 would invent a flat market.
    """
    s = s.sort_values(["fixture_id", "snapshot_ts"]).copy()
    first_p = s.groupby("fixture_id")["v11_p_market"].transform("first")
    s["previous_market_move_pp"] = (s["v11_p_market"] - first_p) * 100.0
    is_first = ~s.duplicated("fixture_id", keep="first")
    s.loc[is_first, "previous_market_move_pp"] = np.nan
    return s


def _attach_results(d: pd.DataFrame) -> pd.DataFrame:
    """Join settled outcomes where they exist. NULL everywhere else — never inferred.

    COVERAGE IS BIASED AND THE BIAS MATTERS. v11 grades from v9's public bets_ledger.csv, which
    holds only fixtures v9 actually TIPPED, so a fixture v9 passed on can never be graded: 94 of
    336 (28%). The graded subset is therefore conditioned on v9 having disagreed with the market
    enough to fire a tip — which is precisely the variable under study. Any ROI or result-based
    figure here is drawn from that conditioned subset. Movement and CLV are NOT affected: they
    need only prices, which are unconditioned.
    """
    d["result"] = pd.NA
    d["pnl_flat"] = pd.NA
    p = config.OUTPUT_DIR / "v11_graded.csv"
    if not p.exists():
        return d
    try:
        g = pd.read_csv(p)
    except Exception:
        return d
    if "over25_result" not in g.columns or "match" not in g.columns:
        return d
    # v11_graded is keyed by match+date, the snapshots by fixture_id; bridge on the pair the
    # snapshot file also carries.
    snaps = _load_snapshots()
    if snaps.empty or "match" not in snaps.columns:
        return d
    bridge = snaps.drop_duplicates("fixture_id")[["fixture_id", "match", "date"]]
    g2 = g[["match", "date", "over25_result", "v11_pnl"]].dropna(subset=["over25_result"])
    m = bridge.merge(g2, on=["match", "date"], how="inner")
    if m.empty:
        return d
    res = dict(zip(m["fixture_id"], m["over25_result"]))
    pnl = dict(zip(m["fixture_id"], m["v11_pnl"]))
    d["result"] = d["fixture_id"].map(res)
    d["pnl_flat"] = d["fixture_id"].map(pnl)
    return d


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(detail, eligible). detail keeps flagged rows; eligible is what carries a claim."""
    s = _load_snapshots()
    if s.empty:
        return pd.DataFrame(), pd.DataFrame()
    s = s[s["p_model_over"].notna() & s["v11_p_market"].notna()]
    if s.empty:
        return pd.DataFrame(), pd.DataFrame()
    s = _attach_momentum(s)

    close = closing_snapshot(s)
    if close is None or close.empty:
        print("[movement] no pre-kickoff closing snapshots — cannot measure movement")
        return pd.DataFrame(), pd.DataFrame()

    d = mv.compute(s, close)
    if d.empty:
        return d, d

    # Dispersion fields the brief asks for. book_dispersion is the per-snapshot std of book
    # probabilities where captured; range is not stored, so it stays NULL rather than being
    # approximated from the std (which would fabricate a number that looks measured).
    # Minutes from entry to close. Without it the time-to-kickoff table cannot be read: a
    # late entry has almost no window left, so its small absolute movement is mechanical.
    d["window_min"] = (pd.to_numeric(d["minutes_to_kickoff"], errors="coerce")
                       - pd.to_numeric(d["close_minutes_to_kickoff"], errors="coerce"))
    d["market_prob_std"] = d.get("book_dispersion")
    d["market_prob_range"] = pd.NA
    d = _attach_results(d)
    for c in DETAIL_COLS:
        if c not in d.columns:
            d[c] = pd.NA
    return d[DETAIL_COLS + ["moved", "p_market_entry"]], mv.eligible(d)


def _chrono(e: pd.DataFrame, unit: str) -> list[dict]:
    """Chronological stability (brief section 13): by month, by week, and split halves.

    A DEGENERATE GROUPING IS REPORTED AS SUCH, NOT EMITTED. With every eligible entry inside one
    ISO week — which is the case today: 2026-08-17 to 2026-08-19, three days — a "week 2026-W34"
    row is a verbatim copy of the overall row. Emitting it puts a line labelled `week ...` in a
    file called "stability" that carries no independent information, and a reader scanning for
    persistence would take it as evidence of persistence. So a grouping with fewer than two
    non-empty periods produces one explicit NOT_ASSESSABLE row naming the span instead.

    The halves split is still emitted when it exists, but it is a split of whatever span the data
    covers, NOT a test across time: over three days it separates two arbitrary halves of one
    weekend, and any difference between them is as likely to be fixture mix as drift.
    """
    out = []
    t = pd.to_datetime(e["entry_ts"], errors="coerce", utc=True)
    e = e.assign(_month=t.dt.strftime("%Y-%m"), _week=t.dt.strftime("%G-W%V"))
    span_days = int(t.dt.date.nunique())
    span = f"{t.min():%Y-%m-%d} to {t.max():%Y-%m-%d}, {span_days} distinct day(s)"

    for key, label in (("_month", "month"), ("_week", "week")):
        periods = [v for v in sorted(e[key].dropna().unique())]
        if len(periods) < 2:
            out.append({"segment": f"{label}ly stability NOT_ASSESSABLE",
                        "unit": unit, "n_obs": len(e),
                        "n_fixtures": int(e["fixture_id"].nunique()), "n_moved": 0,
                        "sample_status": "INSUFFICIENT_SAMPLE",
                        "note": f"all eligible entries fall in a single {label} "
                                f"({periods[0] if periods else 'none'}); span {span}. "
                                f"A per-{label} row would duplicate the overall row."})
            continue
        for v, g in e.groupby(key, sort=True):
            r = mv.summarise(g, f"{label} {v}", unit).__dict__
            r["note"] = ""
            out.append(r)

    # Halves by fixture ORDER, not by row order — an even split of rows would put most of one
    # busy fixture's 40 snapshots in one half.
    fx = (e.sort_values("entry_ts").drop_duplicates("fixture_id")["fixture_id"].tolist())
    if len(fx) >= 4:
        half = len(fx) // 2
        first, second = set(fx[:half]), set(fx[half:])
        for label, keep in (("first half of sample", first), ("second half of sample", second)):
            r = mv.summarise(e[e["fixture_id"].isin(keep)], label, unit).__dict__
            r["note"] = (f"split of a {span_days}-day span, NOT a test across time"
                         if span_days < 14 else "")
            out.append(r)
    return out


def report(detail: pd.DataFrame, e: pd.DataFrame) -> dict[str, pd.DataFrame]:
    n_fix = detail["fixture_id"].nunique()
    print(f"\n=== eligibility ===")
    print(f"  snapshot observations built        : {len(detail):,} over {n_fix} fixtures")
    vc = detail["clv_quality"].value_counts()
    for k, v in vc.items():
        print(f"    {k:26} {v:>6,}")
    print(f"  eligible for a movement claim     : {len(e):,} over "
          f"{e['fixture_id'].nunique()} fixtures")

    fl = mv.fixture_level(e)
    print(f"\n=== PRIMARY: one observation per fixture ({len(fl)} fixtures, earliest entry) ===")
    prim = mv.summarise(fl, "overall (fixture-level, earliest entry)", "fixture")
    _print_summary(prim)

    moved_fl = fl[fl["moved"].fillna(False)]
    p_rate, anchor = mv.placebo_toward_rate(moved_fl)
    print(f"\n  PLACEBO — price moved toward a FIXED anchor ({anchor:.3f}, no model input)")
    print(f"    toward anchor                : {p_rate:.1%}")
    excess = prim.toward_rate - p_rate
    print(f"    model's excess over placebo  : {excess:+.1%}")
    if p_rate >= prim.toward_rate - 0.02:
        print("    -> THE MODEL ADDS NOTHING here: a constant does as well, so the headline is")
        print("       mean reversion in a noisy opening price, not foresight.")
    elif p_rate > 0.5:
        print("    -> the market does mean-revert; the model still beats the constant anchor,")
        print("       and only that excess can be skill")
    else:
        print("    -> no mean reversion in the control; the headline is not explained by it")

    print(f"\n=== SECONDARY: all eligible snapshots (cluster-bootstrapped by fixture) ===")
    sec = mv.summarise(e, "overall (snapshot-level, cluster CI)", "snapshot")
    _print_summary(sec)

    summary_rows = [prim.__dict__, sec.__dict__,
                    {"segment": "PLACEBO fixed anchor (fixture-level)", "unit": "fixture",
                     "n_obs": len(moved_fl), "n_fixtures": len(moved_fl),
                     "n_moved": len(moved_fl), "toward_rate": round(p_rate, 4),
                     "sample_status": mv.sample_status(len(moved_fl))}]
    summary_rows += _chrono(e, "snapshot")

    by_res = mv.summarise_by(fl, "residual_band", "fixture", prefix="residual ")
    by_abs = mv.summarise_by(fl, "abs_residual_band", "fixture", prefix="abs residual ")
    by_model = mv.summarise_by(fl, "model_type", "fixture", prefix="model ")
    # Time buckets are the one question that lives INSIDE a fixture, so they use every snapshot
    # with a clustered interval — a fixture can legitimately contribute one row per horizon.
    by_time = mv.summarise_by(e, "time_band", "snapshot", prefix="T-")
    by_league = mv.summarise_by(fl, "league", "fixture")

    print(f"\n=== by residual band (fixture-level) ===")
    _print_table(by_res)
    print(f"\n=== by |residual| (fixture-level) ===")
    _print_table(by_abs)
    print(f"\n=== by model type (fixture-level) ===")
    _print_table(by_model)
    print(f"\n=== by time to kickoff (snapshot-level, clustered) ===")
    print("  NOTE: mean |move| SHRINKS toward kickoff because the WINDOW to the close shrinks")
    print("  with it — a T-40m entry has ~34 minutes left. `per hour` is the comparable figure,")
    print("  and it rises ~45x: the market moves far FASTER per unit time near kickoff. Reading")
    print("  the absolute column alone states the opposite of the truth.")
    _print_time_table(by_time)
    print(f"\n=== by league (fixture-level) ===")
    _print_table(by_league)

    return {
        "v11_market_movement_detail.csv": detail,
        "v11_movement_summary.csv": pd.DataFrame(summary_rows),
        "v11_movement_by_residual.csv": pd.concat([by_res, by_abs], ignore_index=True),
        "v11_movement_by_model.csv": by_model,
        "v11_movement_by_time.csv": by_time,
        "v11_movement_by_league.csv": by_league,
    }


def _print_summary(s: mv.Summary) -> None:
    print(f"  n moved / n obs / fixtures   : {s.n_moved} / {s.n_obs} / {s.n_fixtures}   "
          f"[{s.sample_status}]")
    print(f"  DIRECTION  toward Wowza      : {_pct(s.toward_rate)}  "
          f"95% CI [{_pct(s.toward_ci_lo)}, {_pct(s.toward_ci_hi)}]  "
          f"z {s.z_vs_chance:+.2f}  p {s.p_value:.4f}")
    print(f"  MAGNITUDE  mean signed move  : {s.mean_signed_move_pp:+.3f}pp  "
          f"95% CI [{s.signed_ci_lo:+.3f}, {s.signed_ci_hi:+.3f}]")
    print(f"             median signed move: {s.median_signed_move_pp:+.3f}pp")
    print(f"             when correct      : {s.mean_move_when_correct_pp:+.3f}pp   "
          f"when wrong: {s.mean_move_when_wrong_pp:+.3f}pp")
    print(f"             window to close   : {s.mean_window_min:.0f} min   "
          f"signed move per hour: {s.signed_move_per_hour_pp:+.4f}pp")
    print(f"             mean |move|       : {s.mean_abs_move_pp:.3f}pp   "
          f"P(>=0.5pp) {_pct(s.p_move_ge_05pp)}  P(>=1pp) {_pct(s.p_move_ge_1pp)}  "
          f"P(>=2pp) {_pct(s.p_move_ge_2pp)}")
    print(f"  EXECUTABLE CLV               : n {s.n_clv}  mean {s.mean_clv_pct:+.3f}%  "
          f"median {s.median_clv_pct:+.3f}%  positive {_pct(s.pct_positive_clv)}")
    print(f"             mean entry / close odds: {s.mean_entry_odds:.3f} / "
          f"{s.mean_close_odds:.3f}")


def _print_time_table(df: pd.DataFrame) -> None:
    if df.empty:
        print("  (no rows)")
        return
    print(f"  {'bucket':<12} {'fix':>4} {'movd':>5} {'toward':>7} {'window':>9} "
          f"{'mean signed':>12} {'per hour':>9} {'mean clv':>9}  status")
    for _, r in df.iterrows():
        ms, ph, cl, w = (r["mean_signed_move_pp"], r["signed_move_per_hour_pp"],
                         r["mean_clv_pct"], r["mean_window_min"])
        print(f"  {str(r['segment'])[:12]:<12} {int(r['n_fixtures']):>4} "
              f"{int(r['n_moved']):>5} {_pct(r['toward_rate']):>7} "
              f"{'      n/a' if pd.isna(w) else f'{w:>6.0f}min':>9} "
              f"{'    n/a' if pd.isna(ms) else f'{ms:>+11.3f}':>12} "
              f"{'   n/a' if pd.isna(ph) else f'{ph:>+8.3f}':>9} "
              f"{'   n/a' if pd.isna(cl) else f'{cl:>+8.3f}':>9}  {r['sample_status']}")


def _pct(v) -> str:
    return "  n/a" if v is None or pd.isna(v) else f"{100 * float(v):.1f}%"


def _print_table(df: pd.DataFrame) -> None:
    if df.empty:
        print("  (no rows)")
        return
    print(f"  {'segment':<34} {'fix':>4} {'movd':>5} {'toward':>7} {'95% CI':>15} "
          f"{'mean signed':>12} {'clv n':>6} {'mean clv':>9}  status")
    for _, r in df.iterrows():
        ci = f"[{_pct(r['toward_ci_lo'])},{_pct(r['toward_ci_hi'])}]"
        ms = r["mean_signed_move_pp"]
        cl = r["mean_clv_pct"]
        print(f"  {str(r['segment'])[:34]:<34} {int(r['n_fixtures']):>4} "
              f"{int(r['n_moved']):>5} {_pct(r['toward_rate']):>7} {ci:>15} "
              f"{'    n/a' if pd.isna(ms) else f'{ms:>+11.3f}':>12} "
              f"{int(r['n_clv']):>6} {'   n/a' if pd.isna(cl) else f'{cl:>+8.3f}':>9}  "
              f"{r['sample_status']}")


def main() -> int:
    detail, e = build()
    if detail.empty:
        print("[movement] not enough history yet — needs >=2 snapshots per fixture with a "
              "known pre-kickoff horizon")
        return 0
    if e.empty:
        print(f"[movement] {len(detail)} observations built but NONE are eligible; "
              f"flags: {dict(detail['clv_quality'].value_counts())}")
        return 0
    outs = report(detail, e)
    print()
    for name, df in outs.items():
        p = config.OUTPUT_DIR / name
        df.to_csv(p, index=False)
        print(f"[movement] {name:<38} {len(df):>6,} rows")
    print("\n[movement] RESEARCH ONLY — not a betting signal. See MARKET_MOVEMENT_RESEARCH.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
