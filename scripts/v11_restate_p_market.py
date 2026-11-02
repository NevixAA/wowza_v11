"""
Restate historical v11_p_market with the ONE estimator, beside the original (Data Truth round 2).
==================================================================================================
Before 2026-11-02 the shadow's p_market switched estimator between snapshots of the same fixture:
an exchange median whenever an exchange pair was present (with NO quality gate - 1.01/1.01
placeholders and 46%-margin Matchbook pairs included), else a cross-book median, else one book's
de-vig, with a proportional fallback. v11_shadow_snapshots.csv is immutable research history, so it
is NOT rewritten. This script writes a sidecar keyed by snapshot_id:

    output/corrections/v11_p_market_restated.csv
        snapshot_id, fixture_id, snapshot_ts, price_source_original, v11_p_market_original,
        p_market_restated, n_books_restated, n_pairs_rejected, n_exchange_rejected,
        exchange_anchor_failed_gate, p_market_estimator

p_market_restated is src/book_consensus.consensus_from_books (the function the live shadow uses)
applied AS OF each snapshot to v9's per-book quote history (output/book_odds_snapshots.csv plus
output/book_odds_snapshots_archive/*.csv, latest quote per book and side at or before the
snapshot), for the side OVER. With fewer than MIN_BOOKS_FOR_CONSENSUS gated books it is the same
estimator on v9's own pair (odds_over25 / odds_under25), exactly as the live shadow now does.

    python scripts/v11_restate_p_market.py --v9 <wowza-betting checkout>            # full
    python scripts/v11_restate_p_market.py --v9 <wowza-betting checkout> --append   # CI

--append keeps every existing sidecar row byte for byte and adds rows only for snapshots that are
not in it yet and do not carry the one estimator's P(OVER) themselves (from 2026-11-02 round 5
the shadow writes v11_p_market = P(OVER) with v11_p_market_basis = 'over').

Read-only on v9. Never touches v11_shadow_snapshots.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
import config  # noqa: E402
from src import league_registry as LR  # noqa: E402
from src.book_consensus import (ESTIMATOR, EXCHANGE_KEYS, MIN_BOOKS_FOR_CONSENSUS,  # noqa: E402
                                P_MARKET_BASIS_OVER, consensus_from_books, normalise_match,
                                pair_ok)
from src.edge_engine import power_devig  # noqa: E402

OUT_REL = Path("corrections") / "v11_p_market_restated.csv"


def load_book_history(v9: Path) -> pd.DataFrame:
    files = sorted((v9 / "output" / "book_odds_snapshots_archive").glob("*.csv"))
    cur = v9 / "output" / "book_odds_snapshots.csv"
    if cur.exists():
        files.append(cur)
    cols = ["snapshot_ts", "league", "match", "bookmaker", "market", "side", "odds"]
    frames = [pd.read_csv(f, usecols=lambda c: c in cols) for f in files]
    q = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    q["market"] = q["market"].astype(str).str.upper().str.strip()
    q = q[q["market"] == "OU25"].copy()
    q["side"] = q["side"].astype(str).str.upper().str.strip()
    q = q[q["side"].isin(("OVER", "UNDER"))]
    q["odds"] = pd.to_numeric(q["odds"], errors="coerce")
    q = q[q["odds"] > 1.0]
    q["_league"] = q["league"].map(LR.canonical_league)
    q = q[q["_league"].notna()]
    q["_match"] = q["match"].map(normalise_match)
    q["_ts"] = pd.to_datetime(q["snapshot_ts"], utc=True, errors="coerce")
    q["bookmaker"] = q["bookmaker"].astype(str).str.strip()
    return q.dropna(subset=["_ts"]).drop_duplicates(
        ["_ts", "_league", "_match", "bookmaker", "side", "odds"])


def single_book(o_over, o_under):
    return power_devig(float(o_over), float(o_under)) if pair_ok(o_over, o_under) else None


def restate(snaps: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    s = snaps.copy()
    s["_ts"] = pd.to_datetime(s["snapshot_ts"], utc=True, errors="coerce")
    s["_league"] = s["league"].map(LR.canonical_league)
    s["_match"] = s["match"].map(normalise_match)
    qg = {k: g.sort_values("_ts") for k, g in quotes.groupby(["_league", "_match"], sort=False)}
    out = []
    for key, g in s.groupby(["_league", "_match"], sort=False, dropna=False):
        q = qg.get(key)
        qi, latest = 0, {"OVER": {}, "UNDER": {}}
        qrows = list(zip(q["_ts"], q["bookmaker"], q["side"], q["odds"])) if q is not None else []
        for _, r in g.sort_values("_ts").iterrows():
            while qi < len(qrows) and pd.notna(r["_ts"]) and qrows[qi][0] <= r["_ts"]:
                ts, bk, sd, od = qrows[qi]
                latest[sd][bk] = od
                qi += 1
            c = consensus_from_books(latest["OVER"], latest["UNDER"])
            exch_bad = sum(1 for bk in set(latest["OVER"]) & set(latest["UNDER"])
                           if bk.lower() in EXCHANGE_KEYS
                           and not pair_ok(latest["OVER"][bk], latest["UNDER"][bk]))
            if c.usable:
                p, n = c.p_consensus, c.n_books
            else:
                p, n = single_book(r.get("odds_over25"), r.get("odds_under25")), 1
            out.append({
                "snapshot_id": r["snapshot_id"], "fixture_id": r["fixture_id"],
                "snapshot_ts": r["snapshot_ts"],
                "price_source_original": r.get("price_source"),
                "v11_p_market_original": r.get("v11_p_market"),
                "v11_side": r.get("v11_side"),
                "p_market_restated": None if p is None else round(float(p), 6),
                "n_books_restated": n if p is not None else 0,
                "n_pairs_rejected": c.n_rejected, "n_exchange_rejected": exch_bad,
                "exchange_anchor_failed_gate": bool(
                    str(r.get("price_source")) == "exchange" and exch_bad > 0),
                "p_market_estimator": ESTIMATOR if p is not None else None,
            })
    return pd.DataFrame(out)


def pending_snapshots(snaps: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    """Snapshots the sidecar still has to cover: not in it yet, and not already carrying the one
    estimator's P(OVER) themselves (p_market_estimator == ESTIMATOR and v11_p_market_basis ==
    'over', written by the shadow from 2026-11-02 round 5)."""
    done = set(old["snapshot_id"].astype(str))
    est = snaps["p_market_estimator"] if "p_market_estimator" in snaps.columns else None
    basis = snaps["v11_p_market_basis"] if "v11_p_market_basis" in snaps.columns else None
    own = pd.Series(False, index=snaps.index)
    if est is not None and basis is not None:
        own = (est.astype(str) == ESTIMATOR) & (basis.astype(str) == P_MARKET_BASIS_OVER)
    return snaps[~snaps["snapshot_id"].astype(str).isin(done) & ~own]


def append_rows(path: Path, columns: list, new: pd.DataFrame) -> None:
    """Append rows to the sidecar WITHOUT rewriting it: existing bytes are untouched."""
    for c in columns:
        if c not in new.columns:
            new[c] = None
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        if size:
            fh.seek(size - 1)
            ends_nl = fh.read(1) == b"\n"
        else:
            ends_nl = True
    with open(path, "a", encoding="utf-8", newline="") as fh:
        if not ends_nl:
            fh.write("\n")
        new[columns].to_csv(fh, header=False, index=False, lineterminator="\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v9", required=True, help="wowza-betting checkout (read-only)")
    ap.add_argument("--append", action="store_true",
                    help="keep every existing sidecar row verbatim and restate only snapshots that "
                         "need it and are not in the sidecar yet (the CI mode)")
    a = ap.parse_args()
    cols = ["snapshot_id", "fixture_id", "snapshot_ts", "league", "match", "odds_over25",
            "odds_under25", "price_source", "v11_p_market", "v11_side", "p_market_estimator",
            "v11_p_market_basis"]
    sf = config.OUTPUT_DIR / "v11_shadow_snapshots.csv"
    head = pd.read_csv(sf, nrows=0).columns
    snaps = pd.read_csv(sf, low_memory=False, usecols=[c for c in cols if c in head])
    p = config.OUTPUT_DIR / OUT_REL
    old = pd.read_csv(p, low_memory=False) if (a.append and p.exists()) else None
    if old is not None:
        snaps = pending_snapshots(snaps, old)
        if snaps.empty:
            print(f"[restate] --append: nothing to restate ({len(old):,} rows kept)")
            return 0
    q = load_book_history(Path(a.v9))
    print(f"[restate] {len(snaps):,} snapshots, {len(q):,} OU25 book quotes")
    r = restate(snaps, q)
    r["generated_at"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    p.parent.mkdir(parents=True, exist_ok=True)
    if old is not None:
        append_rows(p, list(old.columns), r)
        print(f"[restate] --append: {len(r):,} row(s) added after {len(old):,} kept verbatim")
        return 0
    r.to_csv(p, index=False)
    print(f"[restate] wrote {p}: {len(r):,} rows; exchange anchor failed the gate on "
          f"{int(r['exchange_anchor_failed_gate'].sum()):,}; pairs rejected on "
          f"{int((r['n_pairs_rejected'] > 0).sum()):,} snapshot(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
