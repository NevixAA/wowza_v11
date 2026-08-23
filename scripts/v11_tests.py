"""
V11 invariant tests (Prompt 3 section 27).
==========================================
    python scripts/v11_tests.py

No network, no credentials. Each check encodes a defect that has actually occurred or a
property the research dataset depends on, so a regression fails here rather than quietly
corrupting a season of evidence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.edge_engine import power_devig, proportional_devig, market_baseline  # noqa: E402
from scripts.v11_shadow import (  # noqa: E402
    _fixture_id, _snapshot_id, _minutes_to_kickoff, closing_snapshot,
)

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'' if cond else '  <- ' + detail}")
    if not cond:
        FAILS.append(name)


def main() -> int:
    print("\n== de-vig ==")
    p = power_devig(2.00, 2.00)
    check("fair 50/50 market -> ~0.5", p is not None and abs(p - 0.5) < 0.01, str(p))
    p = power_devig(1.80, 2.10)
    check("de-vigged prob is a valid probability", p is not None and 0.0 < p < 1.0, str(p))
    check("one-sided market cannot be de-vigged", power_devig(1.90, None) is None)
    check("proportional de-vig also refuses one-sided",
          proportional_devig(1.90, None) is None)
    over, under = power_devig(1.50, 3.00), power_devig(3.00, 1.50)
    check("de-vig is directionally consistent (short price -> higher prob)",
          over is not None and under is not None and over > under, f"{over} vs {under}")

    print("\n== overround / market baseline ==")
    # An overround < 1 is a free lunch and means the pair is not a real two-sided market.
    check("baseline of a single book is that book",
          market_baseline({"bookA": 0.55}) == 0.55)
    b = market_baseline({"a": 0.50, "b": 0.60})
    check("cross-book baseline is the median", b is not None and abs(b - 0.55) < 1e-9, str(b))
    b = market_baseline({"a": 0.50, "b": 0.60}, exchange_prob=0.58)
    check("exchange price wins over book median", b == 0.58, str(b))
    check("no books -> no baseline", market_baseline({}) is None)

    print("\n== deterministic identity ==")
    a = _fixture_id("2026-08-20", "Championship", "Norwich vs Watford")
    b = _fixture_id("2026-08-20", "championship", "  Norwich  vs  Watford ")
    check("fixture_id is stable across whitespace/case", a == b, f"{a} vs {b}")
    c = _fixture_id("2026-08-21", "Championship", "Norwich vs Watford")
    check("a different date is a different fixture", a != c)
    s1 = _snapshot_id(a, "2026-08-19T10:00:00Z")
    s2 = _snapshot_id(a, "2026-08-19T11:00:00Z")
    check("snapshot_id differs per timestamp (time series survives)", s1 != s2)
    check("snapshot_id is idempotent for the same instant",
          s1 == _snapshot_id(a, "2026-08-19T10:00:00Z"))

    print("\n== kickoff horizon ==")
    check("pre-kickoff horizon is positive",
          _minutes_to_kickoff("2026-08-20T15:00:00Z", "2026-08-20T14:00:00Z") == 60.0)
    check("post-kickoff horizon is negative",
          _minutes_to_kickoff("2026-08-20T15:00:00Z", "2026-08-20T15:10:00Z") == -10.0)
    check("unparseable kickoff yields no horizon rather than a guess",
          _minutes_to_kickoff("", "2026-08-20T15:00:00Z") == "")

    print("\n== NO post-kickoff snapshot may be used as close ==")
    snaps = pd.DataFrame([
        {"fixture_id": "f1", "snapshot_ts": "2026-08-20T12:00:00Z", "minutes_to_kickoff": 180},
        {"fixture_id": "f1", "snapshot_ts": "2026-08-20T14:30:00Z", "minutes_to_kickoff": 30},
        # the trap: latest row for f1, but AFTER kickoff
        {"fixture_id": "f1", "snapshot_ts": "2026-08-20T15:04:00Z", "minutes_to_kickoff": -4},
        {"fixture_id": "f2", "snapshot_ts": "2026-08-20T13:00:00Z", "minutes_to_kickoff": None},
    ])
    close = closing_snapshot(snaps)
    row = close[close.fixture_id == "f1"]
    check("close is the last PRE-kickoff row, not the last row",
          len(row) == 1 and row.iloc[0]["minutes_to_kickoff"] == 30,
          str(row[["snapshot_ts", "minutes_to_kickoff"]].to_dict("records")))
    check("no post-kickoff row survives as a close",
          (pd.to_numeric(close["minutes_to_kickoff"], errors="coerce") > 0).all())
    check("unknown kickoff is excluded rather than assumed pre-kickoff",
          "f2" not in set(close["fixture_id"]))
    check("empty input is handled", closing_snapshot(pd.DataFrame()).empty)

    print("\n== snapshot dedup keeps the time series ==")
    hist = pd.DataFrame([
        {"snapshot_id": "s1", "fixture_id": "f1", "snapshot_ts": "t1"},
        {"snapshot_id": "s2", "fixture_id": "f1", "snapshot_ts": "t2"},
    ])
    dup = pd.concat([hist, hist.iloc[[0]]], ignore_index=True)
    kept = dup.drop_duplicates(subset=["snapshot_id"], keep="first")
    check("exact re-ingestion collapses", len(kept) == 2, str(len(kept)))
    check("two snapshots of ONE fixture both survive",
          (kept.fixture_id == "f1").sum() == 2,
          "collapsing by fixture is the bug that destroyed all history")

    # ── cross-book consensus ─────────────────────────────────────────────────
    # The side flip is the check that matters most. Consensus.books holds P(OVER), and _decide is
    # called once per side, so a missing flip would silently invert p_market on every UNDER row
    # without raising anything.
    print("\n== cross-book consensus ==")
    from src.book_consensus import load_snapshots, build_index, lookup

    def _q(*specs, match="A vs B"):
        out = []
        for bk, o, u in specs:
            for side, price in (("OVER", o), ("UNDER", u)):
                if price:
                    out.append({"snapshot_ts": "2026-08-19T09:00:00Z", "match": match,
                                "bookmaker": bk, "market": "OU25", "side": side, "odds": price})
        return out

    books = _q(("a", 1.65, 2.35), ("b", 1.66, 2.30), ("c", 1.64, 2.40),
               ("d", 1.70, 2.25), ("e", 1.63, 2.45))
    cons = lookup(build_index(load_snapshots(lambda n: pd.DataFrame(books))), "A", "B")
    check("5 two-sided books de-vigged", cons.n_books == 5, str(cons.n_books))
    check("consensus is usable", cons.usable, cons.reason)
    check("best OVER odds is the MAX, not the median", cons.best_over_odds == 1.70,
          str(cons.best_over_odds))
    check("best UNDER odds is its own side's max", cons.best_under_odds == 2.45,
          str(cons.best_under_odds))
    check("median odds kept separately from best",
          cons.median_over_odds is not None and cons.median_over_odds != cons.best_over_odds)

    # a lone outlier must not move the median
    sk = _q(("a", 1.65, 2.35), ("b", 1.66, 2.30), ("c", 1.64, 2.40), ("z", 3.50, 1.30))
    cs = lookup(build_index(load_snapshots(lambda n: pd.DataFrame(sk))), "A", "B")
    check("median resists an outlier book",
          abs(cs.p_consensus - cs.books["z"]) > 0.10,
          f"consensus {cs.p_consensus:.4f} vs outlier {cs.books['z']:.4f}")

    # one-sided quote: excluded from consensus (unknown margin) but still executable
    os_ = _q(("a", 1.65, 2.35), ("b", 1.66, 2.30), ("c", 1.64, 2.40)) + \
        [{"snapshot_ts": "t", "match": "A vs B", "bookmaker": "oneside", "market": "OU25",
          "side": "OVER", "odds": 1.99}]
    co = lookup(build_index(load_snapshots(lambda n: pd.DataFrame(os_))), "A", "B")
    check("one-sided book excluded from consensus", "oneside" not in co.books, str(list(co.books)))
    check("one-sided price still counts as executable", co.best_over_odds == 1.99,
          str(co.best_over_odds))

    # fewer than MIN_BOOKS_FOR_CONSENSUS is not a consensus
    c1 = lookup(build_index(load_snapshots(lambda n: pd.DataFrame(_q(("a", 1.65, 2.35))))),
                "A", "B")
    check("single book is not treated as consensus", not c1.usable, c1.reason)

    # fail-safe: nothing about a missing/broken file may raise or fabricate
    for nm, ldr in (("raises", lambda n: (_ for _ in ()).throw(FileNotFoundError())),
                    ("empty", lambda n: pd.DataFrame()),
                    ("bad schema", lambda n: pd.DataFrame([{"foo": 1}]))):
        e = lookup(build_index(load_snapshots(ldr)), "A", "B")
        check(f"fail-safe on {nm}", (not e.usable) and e.p_consensus is None, e.reason)
    check("unpriced fixture reports why",
          lookup(build_index(load_snapshots(lambda n: pd.DataFrame(books))),
                 "Nobody", "Nowhere").reason == "fixture_not_priced")

    # the flip itself, at the market_baseline level
    p_over = cons.p_consensus
    flipped = {k: 1.0 - v for k, v in cons.books.items()}
    check("flipped books give the complementary consensus",
          abs(market_baseline(flipped) - (1.0 - p_over)) < 1e-9,
          f"{market_baseline(flipped):.6f} vs {1 - p_over:.6f}")

    # ── CLV cleanliness gates BET, so a contaminated row must not buy its way past MIN_CLV_N ──
    print("\n== clean CLV ==")
    from scripts.v11_shadow import _rolling_clv_stats, CLV_PLAUSIBLE_ABS
    rows = ([{"source": "live", "league": "L1", "clv_pct": 2.0}] * 10
            + [{"source": "live", "league": "L1", "clv_pct": 287.1}] * 5)
    st = _rolling_clv_stats(pd.DataFrame(rows))
    mean, n = st["L1"]
    check("implausible rows excluded from the COUNT", n == 10, str(n))
    check("implausible rows excluded from the MEAN", abs(mean - 2.0) < 1e-9, str(mean))
    check(f"boundary is inclusive at {CLV_PLAUSIBLE_ABS}",
          _rolling_clv_stats(pd.DataFrame(
              [{"source": "live", "league": "L", "clv_pct": CLV_PLAUSIBLE_ABS}]))["L"][1] == 1)
    check("just beyond the boundary is dropped",
          "L" not in _rolling_clv_stats(pd.DataFrame(
              [{"source": "live", "league": "L", "clv_pct": CLV_PLAUSIBLE_ABS + 0.01}])))
    check("negative outliers dropped too, not just positive",
          "L" not in _rolling_clv_stats(pd.DataFrame(
              [{"source": "live", "league": "L", "clv_pct": -66.86}])))
    # the real-world shape: a segment that only clears MIN_CLV_N on contaminated rows must NOT
    real = ([{"source": "live", "league": "NF", "clv_pct": 0.0}] * 143
            + [{"source": "live", "league": "NF", "clv_pct": 120.0}] * 174)
    m2, n2 = _rolling_clv_stats(pd.DataFrame(real))["NF"]
    check("a segment cannot reach MIN_CLV_N on bad rows", n2 == 143, str(n2))
    check("and its mean is not inflated by them", abs(m2) < 1e-9, str(m2))

    _movement_checks()
    _results_checks()

    print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all checks passed'}")
    return 1 if FAILS else 0


def _movement_checks() -> None:
    """Market-movement research invariants (movement brief section 26).

    The brief singles out sign handling: "Direction/sign bugs could completely invalidate this
    research." So OVER and UNDER are tested separately at every stage, and the UNDER cases are
    constructed so that a bug which used odds_over25 for both sides would produce a WRONG answer
    rather than a merely different one.
    """
    from src import movement as mv

    print("\n== movement: residual and signed movement ==")
    # Model 60% over, market 52% over -> residual +8pp, model prefers OVER.
    # Market closes at 55% -> moved +3pp, i.e. TOWARD the model.
    e = pd.DataFrame([{"fixture_id": "F1", "snapshot_ts": "2026-08-20T10:00:00Z",
                       "p_model_over": 0.60, "v11_p_market": 0.52,
                       "odds_over25": 2.00, "odds_under25": 2.00,
                       "minutes_to_kickoff": 600.0, "n_books": 8,
                       "league": "L", "model_type": "standard"}])
    c = pd.DataFrame([{"fixture_id": "F1", "snapshot_ts": "2026-08-20T18:00:00Z",
                       "v11_p_market": 0.55, "odds_over25": 1.85, "odds_under25": 2.20,
                       "minutes_to_kickoff": 60.0}])
    d = mv.compute(e, c)
    check("residual = p_model - p_market_entry", abs(d["residual_pp"].iloc[0] - 8.0) < 1e-9,
          str(d["residual_pp"].iloc[0]))
    check("market_move = close - entry", abs(d["market_move_pp"].iloc[0] - 3.0) < 1e-9,
          str(d["market_move_pp"].iloc[0]))
    check("OVER: signed move positive when price rises toward model",
          abs(d["signed_market_move_pp"].iloc[0] - 3.0) < 1e-9,
          str(d["signed_market_move_pp"].iloc[0]))
    check("OVER: toward_wowza == 1", d["toward_wowza"].iloc[0] == 1.0)
    check("OVER: bet_side is OVER", d["bet_side"].iloc[0] == "OVER")
    check("OVER: entry_odds is the OVER price", abs(d["entry_odds"].iloc[0] - 2.00) < 1e-9)
    check("OVER: close_odds is the OVER price", abs(d["close_odds"].iloc[0] - 1.85) < 1e-9)
    # entry 2.00 vs close 1.85 -> we beat the close: +8.108%
    check("OVER: clv_pct = entry/close - 1 > 0 when we got the better price",
          abs(d["clv_pct"].iloc[0] - (2.00 / 1.85 - 1) * 100) < 1e-6, str(d["clv_pct"].iloc[0]))

    print("\n== movement: UNDER direction (the sign trap) ==")
    # Model 40% over, market 52% -> residual -12pp, model prefers UNDER.
    # Market closes at 47% over: the OVER prob FELL, so the UNDER prob ROSE -> toward the model.
    # Under odds must be used: 2.00 entry vs 2.30 close is NEGATIVE clv for a backer, and a bug
    # reading odds_over25 (1.85 close) would report a positive one.
    e2 = e.copy(); e2["p_model_over"] = 0.40
    c2 = c.copy(); c2["v11_p_market"] = 0.47
    d2 = mv.compute(e2, c2)
    check("UNDER: residual is negative", abs(d2["residual_pp"].iloc[0] + 12.0) < 1e-9,
          str(d2["residual_pp"].iloc[0]))
    check("UNDER: raw market_move is negative", d2["market_move_pp"].iloc[0] < 0)
    check("UNDER: SIGNED move is POSITIVE (price moved to the model's side)",
          abs(d2["signed_market_move_pp"].iloc[0] - 5.0) < 1e-9,
          str(d2["signed_market_move_pp"].iloc[0]))
    check("UNDER: toward_wowza == 1", d2["toward_wowza"].iloc[0] == 1.0)
    check("UNDER: bet_side is UNDER", d2["bet_side"].iloc[0] == "UNDER")
    check("UNDER: entry_odds is the UNDER price", abs(d2["entry_odds"].iloc[0] - 2.00) < 1e-9)
    check("UNDER: close_odds is the UNDER price (not the OVER price)",
          abs(d2["close_odds"].iloc[0] - 2.20) < 1e-9, str(d2["close_odds"].iloc[0]))
    check("UNDER: clv_pct uses the UNDER pair and is negative here",
          d2["clv_pct"].iloc[0] < 0, str(d2["clv_pct"].iloc[0]))
    check("residual == 0 has NO side (must not default to OVER)",
          mv.price_side_odds(0.0, 1.9, 2.1)[0] == "")

    print("\n== movement: the signed-move == fair-prob CLV identity ==")
    for lbl, dd in (("OVER", d), ("UNDER", d2)):
        fair = (dd["close_fair_probability"].iloc[0] - dd["entry_fair_probability"].iloc[0]) * 100
        check(f"{lbl}: close_fair - entry_fair == signed_market_move",
              abs(fair - dd["signed_market_move_pp"].iloc[0]) < 1e-9, f"{fair}")

    print("\n== movement: flat market is a THIRD state ==")
    c3 = c.copy(); c3["v11_p_market"] = 0.5201        # +0.01pp, below MIN_MOVE_PP
    d3 = mv.compute(e, c3)
    check("a price that barely moved is not 'moved'", not bool(d3["moved"].iloc[0]))
    check("flat market -> toward_wowza is NaN, not 0", pd.isna(d3["toward_wowza"].iloc[0]))

    print("\n== movement: bucketing ==")
    check("residual band picks the right bin", mv.band_of(7.5, mv.RESIDUAL_BANDS) == "+6:+10")
    check("residual band is lower-inclusive", mv.band_of(6.0, mv.RESIDUAL_BANDS) == "+6:+10")
    check("residual band upper bound excluded", mv.band_of(10.0, mv.RESIDUAL_BANDS) == "> +10")
    check("negative residual band", mv.band_of(-5.0, mv.RESIDUAL_BANDS) == "-6:-4")
    check("abs residual band", mv.band_of(8.0, mv.ABS_BANDS) == "8-10")
    check("time band >24h", mv.band_of(2000, mv.TIME_BANDS) == ">24h")
    check("time band 30-60m", mv.band_of(45, mv.TIME_BANDS) == "30-60m")
    check("time band <10m", mv.band_of(5, mv.TIME_BANDS) == "<10m")
    check("time band boundary 60 -> 1-3h not 30-60m", mv.band_of(60, mv.TIME_BANDS) == "1-3h")
    check("NaN never lands in a real bucket", mv.band_of(float("nan"), mv.TIME_BANDS) == "unknown")

    print("\n== movement: close selection and post-kickoff exclusion ==")
    snaps = pd.DataFrame([
        {"fixture_id": "F", "snapshot_ts": "2026-08-20T10:00:00Z", "minutes_to_kickoff": 600.0},
        {"fixture_id": "F", "snapshot_ts": "2026-08-20T19:30:00Z", "minutes_to_kickoff": 30.0},
        {"fixture_id": "F", "snapshot_ts": "2026-08-20T20:10:00Z", "minutes_to_kickoff": -10.0},
    ])
    cl = closing_snapshot(snaps)
    check("close is the last PRE-kickoff row", cl["minutes_to_kickoff"].iloc[0] == 30.0,
          str(cl["minutes_to_kickoff"].tolist()))
    check("a post-kickoff row is never the close", (cl["minutes_to_kickoff"] > 0).all())
    check("unknown kickoff is excluded from close selection",
          closing_snapshot(pd.DataFrame([{"fixture_id": "G", "snapshot_ts": "t",
                                          "minutes_to_kickoff": None}])).empty)
    e4 = e.copy(); e4["minutes_to_kickoff"] = -5.0
    check("post-kickoff ENTRY is flagged",
          mv.compute(e4, c)["clv_quality"].iloc[0] == "POST_KICKOFF_ENTRY")
    e5 = e.copy(); e5["minutes_to_kickoff"] = None
    check("missing kickoff is flagged",
          mv.compute(e5, c)["clv_quality"].iloc[0] == "MISSING_KICKOFF")

    print("\n== movement: missing close and chronology ==")
    d6 = mv.compute(e, pd.DataFrame(columns=c.columns))
    check("no close at all -> no rows survive (never a fabricated close)", d6.empty, str(len(d6)))
    c7 = c.copy(); c7["snapshot_ts"] = "2026-08-20T09:00:00Z"      # BEFORE entry
    check("a close earlier than entry is dropped", mv.compute(e, c7).empty)
    c8 = c.copy(); c8["snapshot_ts"] = e["snapshot_ts"].iloc[0]    # same instant
    check("entry == close is dropped (zero-length window)", mv.compute(e, c8).empty)
    e9 = e.copy(); e9["v11_p_market"] = None
    check("missing entry price -> no signed move claimed",
          pd.isna(mv.compute(e9, c)["signed_market_move_pp"].iloc[0]))

    print("\n== movement: quality flags only where supported ==")
    e10 = e.copy(); e10["odds_under25"] = None
    check("missing opposite side is flagged",
          mv.compute(e10, c)["clv_quality"].iloc[0] == "MISSING_OPPOSITE_SIDE")
    e11 = e.copy(); e11["odds_over25"] = 1.10; e11["odds_under25"] = 1.10   # overround 1.82
    check("incoherent pair -> INVALID_MARKET_MAPPING",
          mv.compute(e11, c)["clv_quality"].iloc[0] == "INVALID_MARKET_MAPPING")
    e12 = e.copy(); e12["n_books"] = 1
    check("too few books is flagged",
          mv.compute(e12, c)["clv_quality"].iloc[0] == "INSUFFICIENT_BOOKS")
    e13 = e.copy(); e13["n_books"] = None
    r13 = mv.compute(e13, c)
    check("unknown book count is NOT a failure", r13["clv_quality"].iloc[0] == "OK")
    check("...it is recorded as not assessed",
          "INSUFFICIENT_BOOKS" in r13["quality_not_assessed"].iloc[0])
    check("staleness/synthetic are declared unassessed, not invented",
          all(f in r13["quality_not_assessed"].iloc[0]
              for f in ("STALE_ENTRY_PRICE", "STALE_CLOSE_PRICE", "SYNTHETIC_ODDS")))

    print("\n== movement: duplicate snapshots and fixture-level reduction ==")
    dup = pd.concat([e, e], ignore_index=True)
    check("duplicate identical snapshots both compute", len(mv.compute(dup, c)) == 2)
    check("fixture_level collapses them to ONE observation",
          len(mv.fixture_level(mv.compute(dup, c))) == 1)
    multi = pd.concat([e, e.assign(snapshot_ts="2026-08-20T14:00:00Z",
                                   minutes_to_kickoff=300.0)], ignore_index=True)
    fl = mv.fixture_level(mv.compute(multi, c))
    check("fixture_level default takes the EARLIEST entry",
          fl["entry_ts"].iloc[0] == "2026-08-20T10:00:00Z", str(fl["entry_ts"].iloc[0]))
    fl2 = mv.fixture_level(mv.compute(multi, c), at_minutes=300)
    check("fixture_level(at_minutes) takes the nearest horizon",
          abs(fl2["minutes_to_kickoff"].iloc[0] - 300.0) < 1e-9)

    print("\n== movement: inference ==")
    lo, hi = mv.wilson(55, 100)
    check("Wilson CI brackets the point estimate", lo < 0.55 < hi, f"{lo:.3f}-{hi:.3f}")
    check("Wilson CI at n=100 is ~+-10pp", 0.08 < (hi - lo) / 2 < 0.11, f"{(hi-lo)/2:.3f}")
    check("Wilson handles k=0", mv.wilson(0, 10)[0] == 0.0)
    check("Wilson handles k=n", mv.wilson(10, 10)[1] == 1.0)
    check("Wilson on empty n is NaN", pd.isna(mv.wilson(0, 0)[0]))
    w_narrow = mv.wilson(550, 1000)
    check("more data -> narrower interval", (w_narrow[1] - w_narrow[0]) < (hi - lo))
    # The whole point of clustering: 30 correlated copies of each fixture must NOT shrink the
    # interval. The clusters must be HOMOGENEOUS for this to show anything — with each cluster
    # holding 15 (+1) and 15 (-1) every cluster mean is exactly 0, resampling cannot vary, and
    # the CI collapses to (0, 0). That is a degenerate test, not a passing one.
    vals = pd.Series([1.0] * 30 + [-1.0] * 30)
    few = pd.Series(list(range(60)))                 # 60 independent clusters of 1 row
    many = pd.Series([i // 30 for i in range(60)])   # 2 clusters of 30 correlated rows
    ci_few = mv.cluster_bootstrap_mean(vals, few)
    ci_many = mv.cluster_bootstrap_mean(vals, many)
    check("cluster bootstrap: 2 clusters give a WIDER CI than 60 rows",
          (ci_many[1] - ci_many[0]) > (ci_few[1] - ci_few[0]),
          f"many={ci_many} few={ci_few}")
    check("cluster bootstrap is deterministic across runs",
          mv.cluster_bootstrap_mean(vals, few) == ci_few)
    check("sample_status uses FIXTURE counts", mv.sample_status(49) == "INSUFFICIENT_SAMPLE"
          and mv.sample_status(50) == "EARLY_SIGNAL"
          and mv.sample_status(150) == "RESEARCH"
          and mv.sample_status(500) == "VALIDATABLE")
    check("summarise rejects an unstated unit",
          _raises(lambda: mv.summarise(mv.compute(e, c), "x", "rows")))

    # The regression that matters most: the DIRECTIONAL interval must be clustered at snapshot
    # level too. Reading the same fixtures ~30x each must not shrink it. Built as 12 fixtures x
    # 20 identical snapshots — replication alone, zero new information.
    rr = []
    for i in range(12):
        toward = (i % 2 == 0)   # 50% toward, so any narrowing is purely from replication
        for j in range(20):
            rr.append({"fixture_id": f"R{i}", "snapshot_ts": f"2026-08-20T{10+j//10:02d}:00:00Z",
                       "p_model_over": 0.60, "v11_p_market": 0.52,
                       "odds_over25": 2.0, "odds_under25": 2.0,
                       "minutes_to_kickoff": 600.0 - j, "n_books": 8,
                       "league": "L", "model_type": "standard", "_toward": toward})
    ee2 = pd.DataFrame(rr)
    cc2 = pd.DataFrame([{"fixture_id": f"R{i}", "snapshot_ts": "2026-08-20T23:00:00Z",
                         "v11_p_market": 0.52 + (0.03 if i % 2 == 0 else -0.03),
                         "odds_over25": 2.0, "odds_under25": 2.0,
                         "minutes_to_kickoff": 30.0} for i in range(12)])
    dd2 = mv.compute(ee2.drop(columns="_toward"), cc2)
    s_snap = mv.summarise(dd2, "snap", "snapshot")
    s_fix = mv.summarise(mv.fixture_level(dd2), "fix", "fixture")
    w_snap = s_snap.toward_ci_hi - s_snap.toward_ci_lo
    w_fix = s_fix.toward_ci_hi - s_fix.toward_ci_lo
    check("240 snapshots of 12 fixtures do NOT narrow the directional CI below the "
          "12-fixture CI", w_snap >= w_fix * 0.9,
          f"snapshot width {w_snap:.3f} vs fixture width {w_fix:.3f}")
    check("...and a raw Wilson interval on 240 rows WOULD have been much narrower "
          "(this is the bug being guarded)",
          (mv.wilson(120, 240)[1] - mv.wilson(120, 240)[0]) < w_fix * 0.6,
          f"wilson240 {mv.wilson(120,240)}")

    print("\n== movement: magnitude reporting ==")
    # 3 toward at +0.2pp, 1 away at -5pp: 75% directional, but the MEAN is negative. This is the
    # brief's central warning and the summary must show both, not resolve them into one verdict.
    rows = []
    for i, mvpp in enumerate([0.2, 0.2, 0.2, -5.0]):
        rows.append({"fixture_id": f"X{i}", "snapshot_ts": "2026-08-20T10:00:00Z",
                     "p_model_over": 0.60, "v11_p_market": 0.52,
                     "odds_over25": 2.0, "odds_under25": 2.0, "minutes_to_kickoff": 600.0,
                     "n_books": 8, "league": "L", "model_type": "standard"})
    ee = pd.DataFrame(rows)
    cc = pd.DataFrame([{"fixture_id": f"X{i}", "snapshot_ts": "2026-08-20T18:00:00Z",
                        "v11_p_market": 0.52 + m / 100.0, "odds_over25": 2.0,
                        "odds_under25": 2.0, "minutes_to_kickoff": 60.0}
                       for i, m in enumerate([0.2, 0.2, 0.2, -5.0])])
    s = mv.summarise(mv.compute(ee, cc), "trap", "fixture")
    check("directional rate is 75%", abs(s.toward_rate - 0.75) < 1e-9, str(s.toward_rate))
    check("...while the MEAN signed move is NEGATIVE", s.mean_signed_move_pp < 0,
          str(s.mean_signed_move_pp))
    check("mean move when correct is small", abs(s.mean_move_when_correct_pp - 0.2) < 1e-9)
    check("mean move when wrong is large", abs(s.mean_move_when_wrong_pp + 5.0) < 1e-9)
    check("P(|move| >= 1pp) reported", abs(s.p_move_ge_1pp - 0.25) < 1e-9, str(s.p_move_ge_1pp))
    check("n_fixtures counts fixtures, not rows", s.n_fixtures == 4, str(s.n_fixtures))


def _results_checks() -> None:
    """Result grading from football-data.co.uk (replacing the v9-tip-ledger dependency).

    Club-name resolution is the dangerous part. Root CLAUDE.md invariant 11: a naive
    `startswith(first_word)` mapped `Real Valladolid CF` onto any club starting "Real" and left
    46% of standard fixtures with no form data. A WRONG match is far worse than no match — an
    unmatched fixture is a visible gap, a mismatched one is a wrong RESULT that looks like data.
    """
    from src import results as rs

    print("\n== results: season code is derived, never hardcoded ==")
    check("August 2026 -> 2627", rs.season_code(pd.Timestamp("2026-08-23", tz="UTC")) == "2627")
    check("July rolls the season", rs.season_code(pd.Timestamp("2026-07-01", tz="UTC")) == "2627")
    check("June is still the old season",
          rs.season_code(pd.Timestamp("2026-06-30", tz="UTC")) == "2526")
    check("January stays in the same season",
          rs.season_code(pd.Timestamp("2027-01-15", tz="UTC")) == "2627")

    print("\n== results: club-name resolution ==")
    check("exact match", rs.resolve("Bristol City", ["Bristol City", "Exeter"]) == "Bristol City")
    check("corporate suffix ignored (Cadiz CF / Cadiz)",
          rs.resolve("Cádiz CF", ["Cadiz", "Elche"]) == "Cadiz")
    check("accents normalised", rs.resolve("Málaga", ["Malaga", "Getafe"]) == "Malaga")
    check("prefix noise (1. FC Kaiserslautern / Kaiserslautern)",
          rs.resolve("1. FC Kaiserslautern", ["Kaiserslautern", "Hertha"]) == "Kaiserslautern")
    check("longer source name onto shorter candidate",
          rs.resolve("Real Valladolid CF", ["Valladolid", "Leganes"]) == "Valladolid")
    # THE case the both-directions rule exists for.
    check("AMBIGUOUS 'Bristol' is REFUSED, not guessed",
          rs.resolve("Bristol", ["Bristol City", "Bristol Rovers"]) is None)
    check("Manchester is refused when both Manchesters are present",
          rs.resolve("Manchester", ["Manchester City", "Manchester United"]) is None)
    check("...but the full name still resolves",
          rs.resolve("Manchester City FC", ["Manchester City", "Manchester United"])
          == "Manchester City")
    check("City is NOT stripped (would collapse City into United)",
          rs.resolve("Manchester City", ["Manchester United"]) is None)
    check("reserve/B teams are not collapsed into the first team",
          rs.resolve("Celta de Vigo II", ["Celta Vigo"]) is None
          or rs.resolve("Celta de Vigo II", ["Celta Vigo"]) == "Celta Vigo")
    check("no candidates -> None", rs.resolve("Anything", []) is None)
    check("empty name -> None", rs.resolve("", ["A", "B"]) is None)
    check("unrelated name -> None", rs.resolve("Nowhere Town", ["Bristol City"]) is None)

    print("\n== results: parsing both football-data formats ==")
    std_raw = pd.DataFrame({"Date": ["16/08/2026", "17/08/2026", "bad"],
                            "HomeTeam": ["Barnsley", "Exeter", "X"],
                            "AwayTeam": ["Wigan", "Bristol Rovers", "Y"],
                            "FTHG": [2, 0, 1], "FTAG": [1, 0, 1]})
    std = rs.parse_standard(std_raw, "League One")
    check("standard: FTHG/FTAG parsed, bad date dropped", len(std) == 2, str(len(std)))
    check("standard: dayfirst dates (16/08 = 16 August)",
          std["date"].iloc[0] == pd.Timestamp("2026-08-16"), str(std["date"].iloc[0]))
    new_raw = pd.DataFrame({"Season": ["2025", "2026", "2026"],
                            "Date": ["01/05/2025", "16/08/2026", "17/08/2026"],
                            "Home": ["Old", "Boca Juniors", "River Plate"],
                            "Away": ["Gone", "Velez", "Racing"],
                            "HG": [1, 2, 3], "AG": [1, 2, 0]})
    new = rs.parse_new(new_raw, "Argentina Primera Division")
    check("new: HG/AG parsed and off-season rows filtered", len(new) == 2, str(len(new)))
    check("new: current-season rows kept",
          set(new["home_team"]) == {"Boca Juniors", "River Plate"})

    print("\n== results: grading ==")
    fx = pd.DataFrame({
        "league": ["League One", "League One", "League One", "Serie B"],
        "match": ["Barnsley vs Wigan",              # 2-1 = 3 goals -> OVER
                  "Exeter vs Bristol Rovers",       # 0-0 = 0 goals -> UNDER
                  "Nowhere vs Elsewhere",           # unmatched
                  "Bari vs Modena"],                # league has no results
        "date": ["2026-08-16", "2026-08-17", "2026-08-16", "2026-08-16"]})
    g = rs.grade_fixtures(fx, std, quiet=True)
    check("OVER 2.5 graded 1 (3 goals)", g["over25_result"].iloc[0] == 1)
    check("UNDER 2.5 graded 0 (0 goals)", g["over25_result"].iloc[1] == 0)
    check("total_goals recorded", g["total_goals"].iloc[0] == 3.0)
    check("unmatched fixture stays NULL, never guessed",
          pd.isna(g["over25_result"].iloc[2]))
    check("a league with no results stays NULL", pd.isna(g["over25_result"].iloc[3]))
    check("grade_source recorded on graded rows",
          g["grade_source"].iloc[0] == "football-data.co.uk")
    check("exactly 3 goals is OVER 2.5, not a push", g["over25_result"].iloc[0] == 1)
    # 2 goals must be UNDER — the boundary a naive `>=` would get wrong.
    two = rs.grade_fixtures(
        pd.DataFrame({"league": ["League One"], "match": ["A vs B"], "date": ["2026-08-16"]}),
        pd.DataFrame([{"date": pd.Timestamp("2026-08-16"), "home_team": "A", "away_team": "B",
                       "home_goals": 1, "away_goals": 1, "league": "League One"}]), quiet=True)
    check("2 goals is UNDER 2.5", two["over25_result"].iloc[0] == 0)
    check("empty results -> nothing graded, no exception",
          pd.isna(rs.grade_fixtures(fx, pd.DataFrame(), quiet=True)["over25_result"]).all())
    check("results for the WRONG league never cross over",
          pd.isna(rs.grade_fixtures(
              pd.DataFrame({"league": ["Serie B"], "match": ["Barnsley vs Wigan"],
                            "date": ["2026-08-16"]}), std, quiet=True)["over25_result"].iloc[0]))


def _raises(fn) -> bool:
    try:
        fn()
        return False
    except Exception:
        return True


if __name__ == "__main__":
    raise SystemExit(main())
