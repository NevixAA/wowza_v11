"""
V11 shadow collector — reads v9's PUBLIC data, logs the market-first engine's decisions.
=========================================================================================
Runs in the wowza-v11 repo, parallel to frozen v9. For each upcoming O/U 2.5 fixture in v9's
committed predictions.csv it records what the market-first engine WOULD decide, so v11 can be
compared to v9's live results at the monthly review. Collection only — no Telegram, no dash,
no stakes, no effect on v9.

Pipeline per fixture: de-vig -> consensus p_market -> model-as-residual blend (<= per-segment
cap: NF 0.45 / std 0.40 vs 0.30 elsewhere) -> uncertainty lower bound -> EV lower bound ->
CLV gate -> longshot cap. v1 = single-book de-vig (multi-book line-shopping comes next, once
V11 fetches its own per-book odds).

Output: output/v11_shadow_log.csv  (one row per fixture, latest snapshot kept).
"""
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
import config
from src.edge_engine import (power_devig, proportional_devig, market_baseline, blend,
                             lower_bound_prob, ev_lb, _band, MAX_BET_ODDS, RARE_EVENT_MARKETS)

TIER_RANK = {"SNIPER": 4, "MARKSMAN": 3, "VALUABLE": 2, "OBSERVE": 1, "NO_BET": 0}
MOAT_W_CAP = {"new_format": 0.45, "standard": 0.40}
DEFAULT_W_CAP = 0.30
# MEASURED, not assumed (Prompt 3 sections 11 and 12). These were hardcoded N_EFF = 1000 and
# ECE = 0.02 since v11 was written, and they push in opposite directions so being wrong about
# them is not neutral: a flat N_EFF of 1000 claims more evidence than a thin segment has, which
# inflates the model's blend weight, while a flat ECE of 0.02 understates uncertainty, which
# inflates EV_lb and lets a bet clear a floor it should not. src/evidence.py derives both from
# settled outcomes with the section-11 hierarchy (model+market+league -> model+market -> global)
# and falls back CONSERVATIVELY when a segment is too thin to speak: less model weight, wider
# uncertainty. Being unsure should cost the model confidence, not grant it.
from src.evidence import EvidenceStore, FALLBACK_N_EFF, FALLBACK_ECE
from src.book_consensus import load_snapshots, build_index, lookup as consensus_lookup

_EVIDENCE = EvidenceStore.from_json(config.OUTPUT_DIR / "v11_evidence.json")
N_EFF = FALLBACK_N_EFF     # per-row values come from _EVIDENCE.lookup() below
ECE = FALLBACK_ECE


def _load_v9(name: str) -> pd.DataFrame:
    """Read a v9 output CSV.

    Order: explicit test override -> local v9 clone -> public raw HTTP with backoff.

    raw.githubusercontent.com RATE-LIMITS unauthenticated requests, and this collector
    fetches several files twice an hour. On 2026-08-17 a plain fetch returned
    `429 Too Many Requests`, which the previous implementation turned into an unhandled
    HTTPError — the run died and wrote nothing. v11's snapshot history stops dead at
    2026-08-11T09:26, six days before, which this explains.

    So: prefer a checkout on disk (free, unthrottled, and what CI should use), and treat a
    429 as retryable rather than fatal. Prompt 3 section 26 — record degraded states, never
    silently corrupt or lose data.
    """
    import os
    import time

    override = config.OUTPUT_DIR / f"_v9_{name}"
    if override.exists():
        return pd.read_csv(override)

    # A sibling clone of wowza-betting, if one is present. V9_LOCAL lets CI point at an
    # actions/checkout of the baseline repo instead of hammering raw HTTP.
    for cand in (os.getenv("V9_LOCAL", ""),
                 config.BASE_DIR.parent / "v9",
                 config.BASE_DIR.parent / "wowza-betting"):
        if not cand:
            continue
        p = Path(cand) / "output" / name
        if p.exists():
            return pd.read_csv(p)

    last = None
    for attempt in range(4):
        try:
            r = requests.get(f"{config.V9_RAW_BASE}/{name}", timeout=30)
            if r.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"[v11] raw.githubusercontent 429 for {name}; "
                      f"retry {attempt + 1}/3 in {wait}s")
                last = requests.HTTPError(f"429 for {name}")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return pd.read_csv(StringIO(r.text))
        except Exception as e:
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(
        f"could not read v9 {name} after retries ({last}). Provide a local v9 checkout "
        f"via V9_LOCAL to avoid raw-HTTP rate limits."
    ) from last


def _blend_weight(odds, n_eff, market, w_cap):
    b = _band(odds)
    if b is None:
        return 0.0
    w = min(w_cap, b["max_w"])
    if n_eff < b["min_n"]:
        w *= n_eff / max(1, b["min_n"])
    if market in RARE_EVENT_MARKETS:
        w *= 0.5
    return max(0.0, min(w_cap, w))


def _validate_odds(o_over, o_under, o_over15=None, o_over35=None):
    """DATA-VALIDATION FIRST (external review): reject contaminated prices before any decision.
    Checks overround sanity + O/U line ordering. Returns (ok, reason)."""
    try:
        oo, ou = float(o_over), float(o_under)
    except (TypeError, ValueError):
        return False, "non-numeric odds"
    if oo <= 1.0 or ou <= 1.0:
        return False, "odds <= 1.0"
    orr = 1.0 / oo + 1.0 / ou
    if not (config.OVERROUND_MIN < orr <= config.OVERROUND_MAX):
        return False, f"overround {orr:.3f} outside ({config.OVERROUND_MIN},{config.OVERROUND_MAX}]"
    ladder = []
    for v in (o_over15, o_over, o_over35):          # over1.5 < over2.5 < over3.5 (odds)
        try:
            fv = float(v)
            if fv > 1.0:
                ladder.append(fv)
        except (TypeError, ValueError):
            pass
    if len(ladder) >= 2 and any(ladder[i] >= ladder[i + 1] for i in range(len(ladder) - 1)):
        return False, "O/U ordering violated"
    return True, ""


def _decide(p_model, o_bet, o_other, n_eff, ece, rclv, rclv_n, w_cap,
            cons=None, side="OVER"):
    out = {"tier": "NO_BET", "p_market": None, "p_blend": None, "ev_lb": None, "abs_edge": None,
           "n_books": 0, "book_dispersion": None, "price_source": "single_book"}
    best = o_bet
    if not o_bet or o_bet <= 1.0 or best > MAX_BET_ODDS:
        return out
    if not o_other or o_other <= 1.0:
        out["tier"] = "OBSERVE"; return out
    p_market = power_devig(o_bet, o_other) or proportional_devig(o_bet, o_other)
    if p_market is None:
        return out

    # ── CROSS-BOOK CONSENSUS (2026-08-19) ────────────────────────────────────
    # This line used to read `market_baseline({}, None)`. With an empty dict market_baseline
    # returns None, so the `or p_market` fallback always fired and v11's "consensus" was one
    # book's two-way de-vig — Prompt 3 section 7's highest-priority gap. v9 now publishes every
    # bookmaker's quote to output/book_odds_snapshots.csv at zero extra API cost (first capture:
    # 3,480 quotes from 17 books), so a real consensus is available.
    #
    # SIDE HANDLING IS THE TRAP. Consensus.books holds P(OVER) per book, because it is built as
    # devig(over_odds, under_odds). `_decide` is called once per side, so on the UNDER call every
    # book probability AND the exchange anchor must be flipped. Getting this wrong would not
    # crash — it would silently invert the market estimate on half of all rows.
    stale_pen = 0.0
    if cons is not None and getattr(cons, "usable", False):
        flip = (str(side).upper() == "UNDER")
        books = {k: (1.0 - v if flip else v) for k, v in cons.books.items()}
        exch = cons.exchange_prob
        if exch is not None and flip:
            exch = 1.0 - exch
        cb = market_baseline(books, exch)
        if cb is not None and 0.0 < cb < 1.0:
            p_market = cb
            out["price_source"] = "exchange" if exch is not None else "cross_book_median"
        out["n_books"] = cons.n_books
        out["book_dispersion"] = None if cons.dispersion is None else round(cons.dispersion, 5)
        # BEST EXECUTABLE price for THIS side, which is a different quantity from the consensus
        # probability: you bet at the best available price, not the median one. On the first real
        # board the best OVER price beat the median by +3.12% (max +9.84%) — comparable to the
        # entire edge this system looks for, and previously discarded.
        bx = cons.best_under_odds if flip else cons.best_over_odds
        if bx and bx > best and bx <= MAX_BET_ODDS:
            best = float(bx)
        # Books disagreeing means the price is not settled yet. Feed that straight into the
        # uncertainty rather than treating a wide market as if it were a tight one.
        if cons.dispersion:
            stale_pen = min(0.02, float(cons.dispersion))

    w = _blend_weight(best, n_eff, "over25", w_cap)
    p_blend = blend(p_model, p_market, w)
    p_lb = lower_bound_prob(p_blend, ece, n_eff, "over25", stale_penalty=stale_pen)
    _ev = ev_lb(p_lb, best)
    ae = p_blend - p_market
    out.update(p_market=round(p_market, 4), p_blend=round(p_blend, 4),
               ev_lb=round(_ev, 4), abs_edge=round(ae, 4))
    b = _band(best)
    if b is None or _ev < b["ev_floor"] or ae < b["abs_floor"]:
        return out
    # CLV gate now needs a big enough CLEAN sample, not just mean>0 (external review): below
    # MIN_CLV_N the CLV evidence is too immature to certify a BET -> cap at VALUABLE (=PAPER).
    clv_ok = (rclv is not None and rclv > 0 and rclv_n is not None and rclv_n >= config.MIN_CLV_N)
    if _ev >= b["ev_floor"] * 2 and ae >= b["abs_floor"] * 1.5 and clv_ok:
        out["tier"] = "SNIPER"
    elif _ev >= b["ev_floor"] * 1.3 and clv_ok:
        out["tier"] = "MARKSMAN"
    else:
        out["tier"] = "VALUABLE"
    return out


def _tier_to_state(tier: str) -> str:
    """Three explicit states (external review). BET only when CLV-certified (SNIPER/MARKSMAN);
    PAPER when it clears economics but CLV isn't validated yet; else NO_BET."""
    if tier in ("SNIPER", "MARKSMAN"):
        return "BET"
    if tier == "VALUABLE":
        return "PAPER"
    return "NO_BET"


# Reject CLV observations whose magnitude is impossible for a goal O/U line.
#
# ROOT CAUSE — already fixed upstream, this guard is for the HISTORICAL residue.
#
# v9 commit c19ca31 (2026-08-10) "stop corrupt O/U lines entering the closing-odds archives":
# the O/U parser matched ANY bet whose name contained "Over/Under", so NON-GOAL markets — corners,
# cards, team totals — leaked their "Over 2.5" price into the goal `over25` key, and BTTS prices
# were copied into `over25`/`under25`. A corners or BTTS-no price sits around 1.20-1.30, so an
# entry of 1.96 against a "closing" of 1.20 produces clv_pct +63 out of nothing.
#
# Confirmed here: 22 contaminated ledger rows have a closing price exactly equal to an archived
# BTTS price, and ZERO clean rows do. The timeline matches exactly — contaminated rows run through
# July (81%) and 1-9 August (45%), and stop dead at match_date 2026-08-09, the day before the fix.
#
# TWO WRONG DIAGNOSES ARE RECORDED HERE ON PURPOSE, because both were plausible and both would
# have produced a wrong fix:
#   1. "mis-joined market or fixture" — refuted: the archived over25/under25 pair is a coherent
#      market, overround median 1.016 over 3,446 fixtures. It WAS a real market. Just not the
#      goals one.
#   2. "in-play prices used as the close" — the leaked prices look exactly like a 0-0 second half
#      (under shortens, over lengthens), which is why this fit so well. But predict already skips
#      kicked-off fixtures (`if dt <= now: continue`, src/predict.py), and the BTTS equality above
#      is not something an in-play goals price would produce.
#
# So the mechanism was never temporal. c19ca31 states it cleaned data GOING FORWARD and left
# "existing archive rows unchanged... a one-time historical re-filter can be done separately if
# wanted". This filter IS that re-filter, applied at READ time rather than by rewriting history —
# which also honours "no deleting, we can't lose data".
#
# 25% is generous for a genuine closing move and far below the contamination band. Filtering here
# only ever makes the gate STRICTER, the safe direction for a system whose default is NO_BET.
#
# WHY THIS FUNCTION IS THE PLACE TO FIX IT. Its result feeds `clv_ok` in `_decide`, which is the
# gate that promotes a signal from PAPER to BET. It previously returned a count it CALLED `clean_n`
# while only dropping NaN, so nothing was ever cleaned. The consequence, measured:
#
#     new_format mean CLV   unfiltered +26.26%   ->   clean -0.015%
#     segments cleared to BET        1            ->   0
#
# So the unfiltered mean would have certified a +26% CLV edge that does not exist, and opened BET on
# it. Clean CLV for new_format is indistinguishable from zero.
#
# 25% is deliberately generous: a genuine closing move that large is already remarkable, and it sits
# far below the contamination band. Filtering here only ever makes the gate STRICTER, which is the
# safe direction for a system whose default is NO_BET.
CLV_PLAUSIBLE_ABS = 25.0


def _rolling_clv_stats(bl: pd.DataFrame) -> dict:
    """league -> (mean_clv_pct, clean_n) from settled live rows.

    `clean_n` now means what it says: implausible observations are excluded from BOTH the mean and
    the count, so a contaminated segment cannot buy its way past MIN_CLV_N on bad rows.
    """
    if bl is None or bl.empty:
        return {}
    if "source" in bl.columns:
        bl = bl[bl["source"].astype(str) == "live"]
    bl = bl.copy()
    bl["clv_pct"] = pd.to_numeric(bl.get("clv_pct"), errors="coerce")
    bl = bl.dropna(subset=["clv_pct"])
    n_present = len(bl)
    bl = bl[bl["clv_pct"].abs() <= CLV_PLAUSIBLE_ABS]
    if n_present and len(bl) < n_present:
        print(f"[v11] CLV: rejected {n_present - len(bl)} of {n_present} observation(s) as "
              f"implausible (|clv_pct| > {CLV_PLAUSIBLE_ABS}%)")
    if bl.empty or "league" not in bl.columns:
        return {}
    g = bl.groupby("league")["clv_pct"]
    means, counts = g.mean(), g.count()
    return {lg: (float(means[lg]), int(counts[lg])) for lg in means.index}


def _fixture_id(date, league, match) -> str:
    """Deterministic fixture identity (Prompt 3 section 5). v9 publishes no fixture_id, so
    one is derived from (date, league, match). Stable across runs and machines, which is what
    lets snapshots taken hours apart join to the same fixture."""
    import hashlib
    import re
    import unicodedata

    def _n(s):
        nf = unicodedata.normalize("NFKD", str(s or ""))
        a = "".join(c for c in nf if not unicodedata.combining(c)).lower()
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", a)).strip()

    raw = f"{str(date or '')[:10]}|{_n(league)}|{_n(match)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _snapshot_id(fixture_id: str, snapshot_ts) -> str:
    """One observation of one fixture at one instant. The dedup key — exact re-ingestion is
    idempotent, while genuine time-series rows are always distinct."""
    import hashlib
    return hashlib.sha1(f"{fixture_id}|{snapshot_ts}".encode("utf-8")).hexdigest()[:16]


def _minutes_to_kickoff(kickoff_utc, snapshot_ts) -> float | str:
    """Horizon of this observation. Required for the T-12h/T-6h/T-3h/T-1h analysis in
    Prompt 3 section 16 — without it a snapshot cannot be placed on the run-up to kickoff."""
    try:
        ko = pd.to_datetime(kickoff_utc, utc=True, errors="coerce")
        sn = pd.to_datetime(snapshot_ts, utc=True, errors="coerce")
        if pd.isna(ko) or pd.isna(sn):
            return ""
        return round((ko - sn).total_seconds() / 60.0, 1)
    except Exception:
        return ""


def closing_snapshot(snapshots: "pd.DataFrame") -> "pd.DataFrame":
    """The latest PRE-kickoff snapshot per fixture — the only legitimate 'close'.

    Prompt 3 sections 9 and 27: never use post-kickoff information, and prove it. Any CLV or
    closing-price derivation must go through here rather than taking the last row per fixture,
    because the last row is not always pre-kickoff: predictions.csv is filtered by DATE, so a
    fixture that kicked off earlier the same day can still be captured (Cardiff City v Wrexham
    AFC, 3.6 minutes after kickoff, 2026-08-17). Those rows are kept as research data and
    excluded here.
    """
    if snapshots is None or snapshots.empty:
        return snapshots
    d = snapshots.copy()
    mtk = pd.to_numeric(d.get("minutes_to_kickoff"), errors="coerce")
    # Unknown kickoff (NaN) is EXCLUDED: we cannot prove such a row is pre-kickoff, and
    # guessing in the permissive direction is exactly how post-kickoff data leaks into CLV.
    d = d[mtk.notna() & (mtk > 0)]
    if d.empty:
        return d
    return d.sort_values("snapshot_ts").drop_duplicates(subset=["fixture_id"], keep="last")


def run():
    df = _load_v9("predictions.csv")
    try:
        rstats = _rolling_clv_stats(_load_v9("bets_ledger.csv"))
    except Exception:
        rstats = {}
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Cross-book prices, loaded ONCE for the whole board rather than per fixture. Absent or
    # malformed file -> empty index -> every lookup is unusable -> _decide behaves exactly as it
    # did before, so this cannot regress a run.
    try:
        _books_idx = build_index(load_snapshots(_load_v9))
        _n_usable = sum(1 for c in _books_idx.values() if c.usable)
        print(f"[v11] cross-book index: {len(_books_idx)} fixture-markets, "
              f"{_n_usable} with >=3 books")
    except Exception as e:
        _books_idx = {}
        print(f"[v11] cross-book index unavailable ({e}) — falling back to single-book de-vig")

    rows = []
    for _, r in df.iterrows():
        try:
            o_over = float(r.get("odds_over25")); o_under = float(r.get("odds_under25"))
            p_over = float(r.get("p_over25"))
        except (TypeError, ValueError):
            continue
        if not (0.0 <= p_over <= 1.0) or str(r.get("date", ""))[:10] < today:
            continue
        lg = str(r.get("league", "")); rc, rn = rstats.get(lg, (None, None))
        w_cap = MOAT_W_CAP.get(str(r.get("model_type", "")), DEFAULT_W_CAP)

        # DATA-VALIDATION FIRST — reject contaminated prices before deciding.
        valid, reason = _validate_odds(o_over, o_under, r.get("odds_over15"), r.get("odds_over35"))
        if not valid:
            side, best, state = "-", {"tier": "NO_BET", "p_market": None, "p_blend": None,
                                      "ev_lb": None, "abs_edge": None}, "NO_BET"
        else:
            # Segment-specific evidence rather than one global constant for every fixture.
            _ev = _EVIDENCE.lookup(str(r.get("model_type", "")), lg)
            _cons = consensus_lookup(_books_idx, r.get("home_team", ""), r.get("away_team", ""),
                                     "OU25")
            over = _decide(p_over, o_over, o_under, _ev.n_eff, _ev.ece, rc, rn, w_cap,
                           cons=_cons, side="OVER")
            under = _decide(1.0 - p_over, o_under, o_over, _ev.n_eff, _ev.ece, rc, rn, w_cap,
                            cons=_cons, side="UNDER")
            side, best = (("OVER", over) if TIER_RANK[over["tier"]] >= TIER_RANK[under["tier"]]
                          else ("UNDER", under))
            state = _tier_to_state(best["tier"])
            if best["tier"] in ("NO_BET", "OBSERVE"):
                side = "-"

        _date = str(r.get("date", ""))[:10]
        _match = f"{r.get('home_team', '')} vs {r.get('away_team', '')}"
        _fid = _fixture_id(_date, lg, _match)
        _kickoff = r.get("kickoff_utc", "")
        _mtk = _minutes_to_kickoff(_kickoff, ts)
        rows.append({
            "fixture_id": _fid,
            "snapshot_id": _snapshot_id(_fid, ts),
            "snapshot_ts": ts, "date": _date, "league": lg,
            "match": _match,
            "kickoff_ts": _kickoff,
            "minutes_to_kickoff": _mtk,
            # v9 stamps these into predictions.csv as of wowza-betting 5f23677. NULL rather
            # than invented where absent (Prompt 3 section 6).
            "v9_generated_at": r.get("generated_at", ""),
            "v9_git_sha": r.get("git_sha", ""),
            "v9_model_sha": r.get("model_sha", ""),
            "model_type": r.get("model_type", ""),
            "odds_over25": o_over, "odds_under25": o_under,
            "valid_odds": valid, "reject_reason": reason,
            "live_side": r.get("best_side", r.get("bet", "")), "live_tier": r.get("signal_tier", ""),
            "p_model_over": round(p_over, 3),
            # WHY the engine decided this. `reject_reason` above carries only the odds-
            # validation verdict, so classify()'s own explanation was being thrown away:
            # all 4,193 stored snapshots had reject_reason NaN and v11_tier NO_BET with no
            # record of the cause. It was in fact "below floors" — every row had a negative
            # EV lower bound (max -0.0384) — which is the engine working correctly, but
            # nothing in the dataset said so. Prompt 3 section 6 wants the reject reason kept.
            "v11_reason": best.get("reason", ""),
            # Which evidence produced this decision. Without it a tier cannot be
            # reproduced later, because the inputs would be invisible.
            "n_eff_used": _ev.n_eff if valid else "",
            "ece_used": _ev.ece if valid else "",
            "evidence_scope": _ev.scope if valid else "",
            "evidence_source": _ev.source if valid else "",
            # How p_market was estimated. Recorded because "cross-book median of 15 books" and
            # "one book's de-vig" are different measurements, and a later analysis that mixes
            # them without knowing which is which cannot interpret its own result. This is also
            # how we will tell whether consensus actually changes decisions or merely restates
            # the single book.
            "price_source": best.get("price_source", ""),
            "n_books": best.get("n_books", 0),
            "book_dispersion": best.get("book_dispersion", ""),
            # Post-kickoff guard (Prompt 3 sections 9 and 27). predictions.csv is pre-match
            # only, but its date filter is DAY-granular, so a fixture that kicked off earlier
            # today can still be snapshotted: Cardiff City v Wrexham AFC was captured 3.6
            # minutes AFTER kickoff on 2026-08-17. Such a row is legitimate research data and
            # is kept, but it must NEVER be usable as a closing price.
            "is_post_kickoff": bool(
                isinstance(_mtk, (int, float)) and _mtk is not True and _mtk < 0
            ),
            "v11_state": state, "v11_side": side, "v11_tier": best["tier"],
            "v11_p_market": best["p_market"], "v11_p_blend": best["p_blend"],
            "v11_ev_lb": best["ev_lb"], "v11_abs_edge": best["abs_edge"],
            "rolling_clv_pct": round(rc, 2) if rc is not None else "",
            "rolling_clv_n": rn if rn is not None else "",
        })
    if not rows:
        # A collector that writes nothing must FAIL, not report success. Previously this
        # returned 0 quietly, so a run that read no fixtures looked identical to a healthy
        # one and the workflow committed nothing. Same disease that hid two multi-day
        # outages in v9.
        print("[v11] ERROR: no upcoming fixtures produced a row. Either v9's "
              "predictions.csv was unreadable or every row was filtered out.")
        return -1

    new = pd.DataFrame(rows)

    # ── append-only snapshot history ─────────────────────────────────────────
    # This file is IMMUTABLE RESEARCH HISTORY. Previously the only output collapsed to one
    # row per (date, match) with keep="last" on every run, so a twice-hourly collector
    # retained a mean of 1.00 snapshots per fixture — every open->moving->close curve it
    # could have built was deleted 30 minutes later. Measured 2026-08-17: 161 rows,
    # max snapshots per fixture = 1.
    #
    # Dedup here is ONLY against exact re-ingestion (same snapshot_id) and keeps the FIRST
    # occurrence, so replaying a run can never rewrite recorded history. It must never
    # collapse by fixture: two rows for one fixture at different timestamps are the entire
    # point of the dataset.
    snap_f = config.OUTPUT_DIR / "v11_shadow_snapshots.csv"
    latest_f = config.OUTPUT_DIR / "v11_shadow_log.csv"   # the LATEST view; consumers read this

    hist = pd.DataFrame()
    if snap_f.exists():
        try:
            hist = pd.read_csv(snap_f)
        except Exception as e:
            print(f"[v11] WARNING: could not read snapshot history ({e}); "
                  f"refusing to overwrite it")
            return -1
    elif latest_f.exists():
        # One-time migration: the pre-fix log holds real observations. Prompt 2 section 3 /
        # Prompt 3 section 19 — never discard research data, even collapsed data.
        try:
            hist = pd.read_csv(latest_f)
            if "snapshot_id" not in hist.columns:
                hist["fixture_id"] = [
                    _fixture_id(r.get("date"), r.get("league"), r.get("match"))
                    for _, r in hist.iterrows()
                ]
                hist["snapshot_id"] = [
                    _snapshot_id(r["fixture_id"], r.get("snapshot_ts"))
                    for _, r in hist.iterrows()
                ]
            print(f"[v11] migrated {len(hist)} pre-fix row(s) into snapshot history")
        except Exception as e:
            print(f"[v11] WARNING: could not migrate legacy log ({e})")
            hist = pd.DataFrame()

    combined = pd.concat([hist, new], ignore_index=True) if not hist.empty else new
    before = len(combined)
    combined = combined.drop_duplicates(subset=["snapshot_id"], keep="first")
    dropped = before - len(combined)
    combined.to_csv(snap_f, index=False)

    # ── latest view (operational convenience, fully derivable) ───────────────
    latest = (combined.sort_values("snapshot_ts")
              .drop_duplicates(subset=["fixture_id"], keep="last"))
    latest.to_csv(latest_f, index=False)

    per_fix = combined.groupby("fixture_id").size()
    sc = latest["v11_state"].value_counts().to_dict() if "v11_state" in latest.columns else {}
    rej = int((latest["valid_odds"] == False).sum()) if "valid_odds" in latest.columns else 0
    print(f"[v11] +{len(rows)} snapshot(s) | history {len(combined)} rows across "
          f"{per_fix.size} fixtures (mean {per_fix.mean():.2f}, max {per_fix.max()} "
          f"snapshots/fixture; {dropped} exact duplicate(s) skipped)")
    print(f"[v11] latest view {len(latest)} fixtures; states {sc}; odds-rejected {rej}")
    return len(rows)


if __name__ == "__main__":
    # Propagate failure. `run()`'s return value used to be discarded, so a run that read no
    # fixtures — or hit raw.githubusercontent's 429 — still exited 0 and the workflow looked
    # healthy. Prompt 3 section 26: failures must be visible.
    _n = run()
    raise SystemExit(0 if isinstance(_n, int) and _n > 0 else 1)
