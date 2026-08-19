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

    print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all checks passed'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
