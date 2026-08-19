"""
Cross-book consensus from v9's per-book odds snapshots.
=======================================================
Closes what Prompt 3 section 7 calls v11's highest-priority gap. `edge_engine.market_baseline`
already knows how to build a consensus from per-book fair probabilities, but the only caller,
`v11_shadow._decide`, invoked it as `market_baseline({}, None)` — an EMPTY dict. With no books it
returns None and the code silently falls back to a single book's two-way de-vig. So v11's
"consensus" has always been one bookmaker's opinion, and the market-first design rests on the
market price being well estimated.

The data was already being paid for and thrown away. OddsAPI returns EVERY bookmaker's price in
the same response v9 already requests, and v9's `not ov25` selection guards kept the first book and
discarded the rest. Since 2026-08-19 v9 additionally appends every quote to
`output/book_odds_snapshots.csv` — the first live capture recorded 3,480 quotes from 17
bookmakers for one board, at zero extra API cost.

TWO DIFFERENT NUMBERS, and conflating them is the classic error:

  * CONSENSUS is for the PROBABILITY. The median across books estimates the market's true belief;
    any single book carries its own bias and margin.
  * BEST EXECUTABLE is for the EV. You bet at the best price available, not the median one. Using
    the median odds to compute EV understates every edge; using the best price as the probability
    estimate manufactures edge out of one book being an outlier.

So this module returns both, plus dispersion — which is itself information: books disagreeing
widely means the price is not yet settled, and the uncertainty penalty should reflect that.

FAIL-SAFE BY CONSTRUCTION. A missing or unparseable file yields an EMPTY index, and every lookup
then returns a Consensus with `p_consensus=None`. Callers fall back to exactly their previous
single-book behaviour, so wiring this in cannot regress a run — it can only add information when
the data is present.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median
from typing import Callable, Iterable

import pandas as pd

from .edge_engine import power_devig, proportional_devig

# Columns v9's src/predict.py writes. Kept as a constant so a schema drift is a loud KeyError
# during load rather than a silently empty index.
REQUIRED_COLS = ("snapshot_ts", "match", "bookmaker", "market", "side", "odds")

# An exchange is a sharper anchor than any bookmaker median, because its price is what other
# people will actually trade against rather than a margin-loaded offer. Treated specially by
# market_baseline, so name them explicitly rather than letting them vote as ordinary books.
EXCHANGE_KEYS = {"betfair_ex_uk", "betfair_ex_eu", "betfair_ex_au", "matchbook", "smarkets"}

# Below this many books the median is not a consensus, it is a small sample with a fancy name.
MIN_BOOKS_FOR_CONSENSUS = 3


@dataclass
class Consensus:
    """What the books collectively say about one side of one fixture's market."""
    p_consensus: float | None = None
    n_books: int = 0
    dispersion: float | None = None          # population stdev of per-book fair probs
    spread: float | None = None              # max-min, a cruder but more legible spread
    best_over_odds: float | None = None      # best EXECUTABLE price, not the median
    best_under_odds: float | None = None
    median_over_odds: float | None = None
    median_under_odds: float | None = None
    exchange_prob: float | None = None
    books: dict[str, float] = field(default_factory=dict)
    reason: str = "no_data"

    @property
    def usable(self) -> bool:
        return self.p_consensus is not None and self.n_books >= MIN_BOOKS_FOR_CONSENSUS


def normalise_match(name: object) -> str:
    """Fixture identity for joining. Deliberately crude but STABLE on both sides of the join.

    v9 writes `match` as "Home vs Away" into this file and builds the same string from the same
    predictions.csv columns elsewhere, so the two sides agree without needing club-name
    resolution. Anything fancier (identity tokens, alias tables) belongs in src/team_names on the
    v9 side; doing it here would let the two drift apart.
    """
    return " ".join(str(name).strip().lower().split())


def load_snapshots(loader: Callable[[str], pd.DataFrame],
                   filename: str = "book_odds_snapshots.csv") -> pd.DataFrame:
    """Load the per-book snapshots via v11's existing v9 loader. Never raises."""
    try:
        df = loader(filename)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        # Loud in the log, empty in the return: a schema change must be visible but must not
        # take a collector run down.
        print(f"[book_consensus] unexpected schema, missing {missing} — ignoring the file")
        return pd.DataFrame()
    out = df.copy()
    out["odds"] = pd.to_numeric(out["odds"], errors="coerce")
    out = out[out["odds"] > 1.0]
    out["_match"] = out["match"].map(normalise_match)
    out["market"] = out["market"].astype(str).str.upper().str.strip()
    out["side"] = out["side"].astype(str).str.upper().str.strip()
    out["bookmaker"] = out["bookmaker"].astype(str).str.strip()
    out = out[out["side"].isin(("OVER", "UNDER"))]
    # LATEST quote per (fixture, market, book, side). The file is append-only and stores price
    # CHANGES, so the last row for a key is that book's current price.
    out = (out.sort_values("snapshot_ts")
              .drop_duplicates(["_match", "market", "bookmaker", "side"], keep="last"))
    return out.reset_index(drop=True)


def _fair_prob(over: float | None, under: float | None) -> float | None:
    """One book's two-way de-vig, power method with a proportional fallback."""
    if not over or not under or over <= 1.0 or under <= 1.0:
        return None
    return power_devig(over, under) or proportional_devig(over, under)


def build_index(df: pd.DataFrame) -> dict[tuple[str, str], Consensus]:
    """Per (normalised match, market) consensus. Empty input gives an empty index."""
    if df is None or df.empty:
        return {}
    index: dict[tuple[str, str], Consensus] = {}
    for (m, mkt), g in df.groupby(["_match", "market"], sort=False):
        over = g[g["side"] == "OVER"].set_index("bookmaker")["odds"].to_dict()
        under = g[g["side"] == "UNDER"].set_index("bookmaker")["odds"].to_dict()
        books: dict[str, float] = {}
        for bk in set(over) | set(under):
            # A book needs BOTH sides to be de-vigged. A one-sided quote carries an unknown
            # margin, so including it would bias the consensus by however much that book charges.
            p = _fair_prob(over.get(bk), under.get(bk))
            if p is not None and 0.0 < p < 1.0:
                books[bk] = float(p)
        if not books:
            index[(m, mkt)] = Consensus(reason="no_two_sided_book")
            continue
        exch = [p for bk, p in books.items() if bk.lower() in EXCHANGE_KEYS]
        vals = list(books.values())
        disp = None
        if len(vals) > 1:
            mu = sum(vals) / len(vals)
            disp = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))
        index[(m, mkt)] = Consensus(
            p_consensus=float(median(vals)),
            n_books=len(books),
            dispersion=disp,
            spread=(max(vals) - min(vals)) if len(vals) > 1 else None,
            best_over_odds=max(over.values()) if over else None,
            best_under_odds=max(under.values()) if under else None,
            median_over_odds=float(median(list(over.values()))) if over else None,
            median_under_odds=float(median(list(under.values()))) if under else None,
            exchange_prob=float(median(exch)) if exch else None,
            books=books,
            reason="ok" if len(books) >= MIN_BOOKS_FOR_CONSENSUS else "too_few_books",
        )
    return index


def lookup(index: dict[tuple[str, str], Consensus], home: object, away: object,
           market: str = "OU25") -> Consensus:
    """Consensus for one fixture, or an unusable Consensus when absent."""
    if not index:
        return Consensus(reason="no_index")
    key = (normalise_match(f"{home} vs {away}"), str(market).upper().strip())
    return index.get(key, Consensus(reason="fixture_not_priced"))
