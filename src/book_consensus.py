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

FAIL-CLOSED. A missing or unparseable file yields an EMPTY index, and every lookup then returns a
Consensus with `p_consensus=None` and no dispersion. The shadow then records the fixture as
OBSERVE (no measured dispersion) rather than pretending a single book is a tight market.

FIXTURE IDENTITY is (canonical league, "Home vs Away", market). The league is resolved through
src/league_registry (v9's config/leagues.json) on both sides, so two leagues that happen to share
a club pairing can never pool their quotes. v9 publishes no fixture id in this file yet; when it
does, the key should become that id (Data Truth 2026-11-01).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median
from typing import Callable, Iterable

import pandas as pd

from . import league_registry as LR
from .edge_engine import power_devig

# Columns v9's src/predict.py writes. Kept as a constant so a schema drift is a loud KeyError
# during load rather than a silently empty index.
REQUIRED_COLS = ("snapshot_ts", "league", "match", "bookmaker", "market", "side", "odds")

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
    n_rejected: int = 0                      # two-sided pairs refused by the overround gate

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
    out["_league"] = out["league"].map(LR.canonical_league)
    unk = int(out["_league"].isna().sum())
    if unk:
        print(f"[book_consensus] dropped {unk} quote(s) whose league is not in the registry")
    out = out[out["_league"].notna()].copy()
    out["_match"] = out["match"].map(normalise_match)
    out["market"] = out["market"].astype(str).str.upper().str.strip()
    out["side"] = out["side"].astype(str).str.upper().str.strip()
    out["bookmaker"] = out["bookmaker"].astype(str).str.strip()
    out = out[out["side"].isin(("OVER", "UNDER"))]
    # LATEST quote per (fixture, market, book, side). The file is append-only and stores price
    # CHANGES, so the last row for a key is that book's current price.
    out = (out.sort_values("snapshot_ts")
              .drop_duplicates(["_league", "_match", "market", "bookmaker", "side"], keep="last"))
    return out.reset_index(drop=True)


#: PAIR QUALITY GATE, the same bounds as Pro (wowzaV9-Pro src/market/devig.py MIN_OVERROUND /
#: MAX_OVERROUND) and as v11's own board check (config.OVERROUND_MIN/MAX). A book's pair enters
#: the consensus only when 1/over + 1/under is inside [1.00, 1.25].
#: Data Truth round 2 (2026-11-02): exchange pairs entered with NO check - 1.01/1.01 placeholders
#: (overround 1.98, de-vigs to exactly 0.5) and Matchbook O 1.56 / U 1.22 (1.46) - and the exchange
#: median then OVERRODE the whole cross-book median (`market_baseline` is exchange-first).
OVERROUND_MIN = 1.00
OVERROUND_MAX = 1.25

#: THE ONE p_market ESTIMATOR, recorded on every row as `p_market_estimator`: power de-vig per
#: book, gated pairs only, median across books. Exchanges vote as ordinary books; they no longer
#: replace the median, and there is no proportional fallback. Before 2026-11-02 p_market switched
#: between snapshots of ONE fixture (1,448 of 2,221 fixtures) among an exchange median, a cross-book
#: median and one book's de-vig, so a "price move" could be an estimator change.
ESTIMATOR = "power_median_gated"


def pair_ok(over, under) -> bool:
    try:
        o, u = float(over), float(under)
    except (TypeError, ValueError):
        return False
    if not (o > 1.0 and u > 1.0):
        return False
    return OVERROUND_MIN <= 1.0 / o + 1.0 / u <= OVERROUND_MAX


def _fair_prob(over: float | None, under: float | None) -> float | None:
    """One book's two-way POWER de-vig, or None when the pair fails the gate."""
    if not pair_ok(over, under):
        return None
    p = power_devig(float(over), float(under))
    return p if p is not None and 0.0 < p < 1.0 else None


def consensus_from_books(over: dict, under: dict) -> Consensus:
    """THE consensus for one fixture-market from each book's latest OVER / UNDER price. Used by the
    live index (build_index) and by the historical restatement, so the two cannot drift."""
    books: dict[str, float] = {}
    bad: set = set()
    for bk in set(over) | set(under):
        # A book needs BOTH sides to be de-vigged. A one-sided quote carries an unknown
        # margin, so including it would bias the consensus by however much that book charges.
        if bk not in over or bk not in under:
            continue
        p = _fair_prob(over.get(bk), under.get(bk))
        if p is None:
            bad.add(bk)
            continue
        books[bk] = float(p)
    rejected = len(bad)
    # Best executable price: every quoted price EXCEPT a book whose two-sided pair failed the gate
    # (a 1.56/1.22 exchange pair is not an executable 1.56). A one-sided quote stays executable.
    ok_over = [o for b, o in over.items() if b not in bad]
    ok_under = [u for b, u in under.items() if b not in bad]
    if not books:
        return Consensus(reason="no_two_sided_book", n_rejected=rejected)
    exch = [p for bk, p in books.items() if bk.lower() in EXCHANGE_KEYS]
    vals = list(books.values())
    disp = None
    if len(vals) > 1:
        mu = sum(vals) / len(vals)
        disp = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))
    return Consensus(
        p_consensus=float(median(vals)),
        n_books=len(books),
        dispersion=disp,
        spread=(max(vals) - min(vals)) if len(vals) > 1 else None,
        best_over_odds=max(ok_over) if ok_over else None,
        best_under_odds=max(ok_under) if ok_under else None,
        median_over_odds=float(median(ok_over)) if ok_over else None,
        median_under_odds=float(median(ok_under)) if ok_under else None,
        exchange_prob=float(median(exch)) if exch else None,
        books=books,
        reason="ok" if len(books) >= MIN_BOOKS_FOR_CONSENSUS else "too_few_books",
        n_rejected=rejected,
    )


RESTATED_REL = "corrections/v11_p_market_restated.csv"
#: scripts/v11_shadow.P_MARKET_BASIS: the shadow's v11_p_market is P(OVER) from 2026-11-02 (round 5).
P_MARKET_BASIS_OVER = "over"


def research_snapshots(path, restated_path=None) -> pd.DataFrame:
    """THE loader for research on v11_shadow_snapshots.csv: rows carry the one-estimator P(OVER)
    as v11_p_market (one_estimator_p_market), the published value as v11_p_market_original, and
    rows with neither are dropped. Used by v11_market_movement, v11_microstructure and
    v11_momentum_control so the three tables rest on one series."""
    from pathlib import Path
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_csv(path, low_memory=False)
    rp = Path(restated_path) if restated_path is not None else path.parent / RESTATED_REL
    return one_estimator_p_market(d, pd.read_csv(rp, low_memory=False) if rp.exists() else None)


def one_estimator_p_market(d: pd.DataFrame, restated: pd.DataFrame | None) -> pd.DataFrame:
    """Snapshot rows with `v11_p_market` = P(OVER) from THE one estimator, for research readers.

    A row the shadow wrote with p_market_estimator == ESTIMATOR AND v11_p_market_basis == 'over'
    keeps its value. Any other row takes `p_market_restated` (always P(OVER)) from the sidecar
    (scripts/v11_restate_p_market.py) by snapshot_id: that includes rows written with the estimator
    but before the basis column, whose value was the CHOSEN side's probability (P(UNDER) whenever
    the UNDER decision ranked higher). A row with neither is DROPPED - its value came from an
    unknown estimator or side. The value the row was published with is kept as
    `v11_p_market_original`. Nothing on disk is changed."""
    d = d.copy()
    d["v11_p_market_original"] = d.get("v11_p_market")

    def _col(name):
        return d[name] if name in d.columns else pd.Series(None, index=d.index, dtype=object)

    own = ((_col("p_market_estimator").astype(str) == ESTIMATOR)
           & (_col("v11_p_market_basis").astype(str) == P_MARKET_BASIS_OVER))
    val = pd.Series(float("nan"), index=d.index)
    val[own] = pd.to_numeric(d.loc[own, "v11_p_market"], errors="coerce")
    if restated is not None and not restated.empty:
        m = restated.drop_duplicates("snapshot_id").set_index("snapshot_id")["p_market_restated"]
        fill = ~own
        val[fill] = pd.to_numeric(d.loc[fill, "snapshot_id"].map(m), errors="coerce")
    d["v11_p_market"] = val
    return d[d["v11_p_market"].notna()]


def build_index(df: pd.DataFrame) -> dict[tuple[str, str, str], Consensus]:
    """Per (canonical league, normalised match, market) consensus. Empty input -> empty index."""
    if df is None or df.empty:
        return {}
    index: dict[tuple[str, str, str], Consensus] = {}
    for (lg, m, mkt), g in df.groupby(["_league", "_match", "market"], sort=False):
        over = g[g["side"] == "OVER"].set_index("bookmaker")["odds"].to_dict()
        under = g[g["side"] == "UNDER"].set_index("bookmaker")["odds"].to_dict()
        index[(lg, m, mkt)] = consensus_from_books(over, under)
    return index


def lookup(index: dict[tuple[str, str, str], Consensus], league: object, home: object,
           away: object, market: str = "OU25") -> Consensus:
    """Consensus for one fixture, or an unusable Consensus when absent. A league that is not in
    the registry never matches (reason 'league_not_in_registry')."""
    if not index:
        return Consensus(reason="no_index")
    lg = LR.canonical_league(league)
    if lg is None:
        return Consensus(reason="league_not_in_registry")
    key = (lg, normalise_match(f"{home} vs {away}"), str(market).upper().strip())
    return index.get(key, Consensus(reason="fixture_not_priced"))
