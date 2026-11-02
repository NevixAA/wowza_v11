"""
V11 grader + scoreboard — turns the shadow log into an actual v9-vs-v11 verdict.
=================================================================================
Grades every logged pick (v9's live tip AND v11's market-first decision) against the real
match result and writes a head-to-head scoreboard: picks, W-L, P/L in units (flat 1u), hit
rate — overall and per month. READ-ONLY collection; no Telegram, no stakes.

v1 result source: v9's PUBLIC bets_ledger.csv (has total_goals for settled fixtures). Coverage
is therefore limited to fixtures v9 tipped + settled; full coverage (grading v11 picks on
fixtures v9 didn't tip) needs a football-data results fetch — documented as the next step.
CLV comparison is a later add (needs closing-line capture for v11's own picks).

Outputs: output/v11_graded.csv  (per-fixture, both picks graded)
         output/v11_scoreboard.csv  (v9 vs v11 aggregates, overall + monthly)
"""
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "scripts"))
import config
from v11_shadow import _load_v9   # reuse the v9-public-data reader

# PROVENANCE + ATOMIC WRITE for derived research artifacts.
#
# `generated_at` is stamped as a COLUMN, not left to the file mtime: `git checkout` resets mtime,
# so on a fresh CI runner every artifact would look seconds old and a staleness check could never
# fire. The column survives the checkout.
#
# Written temp -> read back -> os.replace, so an interrupted run leaves the previous good file
# rather than a half-written CSV that pandas will happily parse as short.
def _read_or_none(p):
    import pandas as _pd
    try:
        return _pd.read_csv(p)
    except Exception:
        return None


def _sample_size(df, name: str):
    """The artifact's MEANINGFUL sample size, not its row count.

    The first version of the shrink guard compared len(df) and never fired: v11_graded.csv has one
    row per logged fixture (521) whether or not a result is known, so a collapse from 344 SETTLED
    fixtures to 186 left the row count untouched. Row count is the wrong measure for every file
    whose rows are placeholders until an outcome arrives.
    """
    if df is None or len(df) == 0:
        return None
    if "over25_result" in df.columns:
        return int(df["over25_result"].notna().sum())
    if {"scope", "n"} <= set(df.columns):
        r = df[df["scope"] == "overall"]
        if not r.empty:
            try:
                return int(r["n"].iloc[0])
            except Exception:
                pass
    return len(df)


def _write_derived(df, name: str, *, allow_shrink: bool = False):
    """Stamp provenance, refuse a local downgrade, then write atomically.

    THE LOCAL-DOWNGRADE GUARD, added after I caused exactly this. Running v11_grade.py on a
    laptop that cannot reach football-data.co.uk falls back to v9's tip ledger and REWRITES
    v11_graded.csv with a strictly worse sample — measured: 344 settled fixtures -> 186. CI can
    reach football-data; a laptop generally cannot. So a local run silently replaced good CI
    output with degraded output, and every downstream analysis then inherited the smaller sample.

    Outside CI, a write that would SHRINK an existing artifact is refused. In CI it is allowed
    (a real methodology change may legitimately reduce n) but printed, so section 7's
    monotonicity contract is visible rather than assumed.
    """
    import os
    import pandas as _pd
    out = df.copy()
    out["generated_at"] = _pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    out["calculation_version"] = "1.0.0"
    p = config.OUTPUT_DIR / name
    p.parent.mkdir(parents=True, exist_ok=True)

    in_ci = os.getenv("GITHUB_ACTIONS", "").lower() == "true"
    if p.exists() and not allow_shrink:
        prev_n, new_n = _sample_size(_read_or_none(p), name), _sample_size(out, name)
        if prev_n is not None and new_n is not None and new_n < prev_n:
            msg = f"{name}: would shrink sample {prev_n} -> {new_n}"
            if not in_ci and os.getenv("V11_ALLOW_SHRINK") != "1":
                raise RuntimeError(
                    f"refusing local downgrade: {msg}. A local run often has less data than CI "
                    f"(football-data.co.uk is unreachable from many networks), so this would "
                    f"replace good CI output with a worse sample. Set V11_ALLOW_SHRINK=1 only if "
                    f"the reduction is a deliberate methodology change.")
            print(f"[write] MONOTONICITY WARNING {msg} (allowed in CI; record why)")

    tmp = p.with_suffix(p.suffix + ".tmp")
    out.to_csv(tmp, index=False)
    back = _pd.read_csv(tmp)
    if len(back) != len(out):
        tmp.unlink(missing_ok=True)
        raise ValueError(f"{name}: wrote {len(out)} rows, read back {len(back)}")
    os.replace(tmp, p)
    return p



BET_TIERS = {"SNIPER", "MARKSMAN", "VALUABLE"}


def _ledger_results(bl: pd.DataFrame) -> pd.DataFrame:
    """v9 bets_ledger -> one row per settled tip: canonical league, date, home, away, over.

    bets_ledger has no total_goals column, so over 2.5 happened iff a tipped OVER won or a tipped
    UNDER lost. The league is the REGISTRY's canonical name; a row whose league is unknown, or
    whose side/result/clubs/date is missing, is dropped (never guessed) and counted."""
    cols = ["league", "date", "home_team", "away_team", "over"]
    need = {"league", "result", "side", "home_team", "away_team", "match_date"}
    if bl is None or bl.empty or not need <= set(bl.columns):
        return pd.DataFrame(columns=cols)
    from src import league_registry as LR
    d = bl.copy()
    res = d["result"].astype(str).str.upper()
    side = d["side"].astype(str).str.upper()
    d = d[res.isin(["WIN", "LOSS"]) & side.isin(["OVER", "UNDER"])].copy()
    if d.empty:
        return pd.DataFrame(columns=cols)
    res, side = d["result"].astype(str).str.upper(), d["side"].astype(str).str.upper()
    out = pd.DataFrame({
        "league": d["league"].map(LR.canonical_league),
        "date": pd.to_datetime(d["match_date"], errors="coerce").dt.strftime("%Y-%m-%d"),
        "home_team": d["home_team"], "away_team": d["away_team"],
        "over": ((side == "OVER") & (res == "WIN")) | ((side == "UNDER") & (res == "LOSS"))})
    bad = out[["league", "date", "home_team", "away_team"]].isna().any(axis=1)
    if bad.any():
        print(f"[grade] ledger: dropped {int(bad.sum())} settled row(s) with an unknown league, "
              f"date or club")
    return out[~bad].reset_index(drop=True)


def _ledger_over(led: pd.DataFrame, league, match, date) -> bool | None:
    """over_happened for one shadow fixture from the ledger, or None. LEAGUE-SCOPED and same-date,
    clubs resolved with v9's canonical result resolver (src/team_names.resolve_result). This used
    to be a raw lowercase/strip-punctuation key with no league, which could join two clubs of the
    same spelling in different competitions."""
    from src import league_registry as LR
    from src.team_names import resolve_result
    lg = LR.canonical_league(league)
    m = str(match) if pd.notna(match) else ""
    if lg is None or " vs " not in m or led is None or led.empty or pd.isna(date):
        return None
    c = led[(led["league"] == lg) & (led["date"] == str(date)[:10])]
    if c.empty:
        return None
    h_raw, a_raw = (x.strip() for x in m.split(" vs ", 1))
    h = resolve_result(h_raw, sorted(c["home_team"].astype(str).unique()))
    a = resolve_result(a_raw, sorted(c["away_team"].astype(str).unique()))
    if not h or not a:
        return None
    hit = c[(c["home_team"].astype(str) == h) & (c["away_team"].astype(str) == a)]
    vals = set(hit["over"].astype(bool))
    return vals.pop() if len(vals) == 1 else None          # two tips that disagree -> unknown


def _grade(side, over_happened, o_over, o_under):
    """WIN/LOSS + flat-1u pnl for a pick, or None if not a bet / no result."""
    side = str(side).upper()
    if side not in ("OVER", "UNDER") or over_happened is None:
        return None
    win = (side == "OVER" and over_happened) or (side == "UNDER" and not over_happened)
    try:
        odds = float(o_over if side == "OVER" else o_under)
    except (TypeError, ValueError):
        return None
    if odds <= 1.0:
        return None
    return {"win": bool(win), "pnl": round((odds - 1.0) if win else -1.0, 3)}


def _agg(df, syscol_side, syscol_tier):
    """Aggregate a system's graded picks: n, wins, losses, pnl, hit%."""
    picks = df[df[syscol_tier].isin(BET_TIERS) & df[f"{syscol_side}_pnl"].notna()]
    n = len(picks)
    if n == 0:
        return {"picks": 0, "W": 0, "L": 0, "pnl": 0.0, "hit": None}
    w = int((picks[f"{syscol_side}_pnl"] > 0).sum())
    return {"picks": n, "W": w, "L": n - w,
            "pnl": round(picks[f"{syscol_side}_pnl"].sum(), 2),
            "hit": round(w / n * 100, 1)}


# ── v9's record, honestly (Data Truth round 2, 2026-11-02) ──────────────────────────────────────
# The v9 rows of the scoreboard used to be graded from the LATEST snapshot per fixture
# (v11_shadow_log.csv). Measured on 2,221 fixtures (ba1bbae): 688 picks, -68.56u. That view
#   * priced every tip at the LAST odds seen, not the price v9 tipped at;
#   * dropped every fixture whose latest tier was no longer a tip (427 fixtures ended AVOID);
#   * counted USA MLS, a PAPER league in v9's registry (105 of the 688 picks);
#   * counted a rescheduled fixture twice, once under each date (fixture_id includes the date).
# Now: the FIRST snapshot in which v9 tipped the fixture (side and price as first seen), every
# fixture v9 ever tipped, paper leagues excluded, and one row per real match under the owner's
# reschedule rule (2026-11-02): a match played within 48 h of the ORIGINAL kickoff settles its tips
# on the actual match; a match moved further than that voids the tips placed before the move.
# The latest-view number is kept beside it as system "v9_latest_view (legacy)", never overwritten.
RESCHEDULE_SETTLE_HOURS = 48
SNAP_COLS = ["fixture_id", "snapshot_ts", "date", "league", "match", "kickoff_ts",
             "minutes_to_kickoff", "live_side", "live_tier", "odds_over25", "odds_under25"]


def _over_value(v):
    """over25_result as 1/0/None. The graded file mixes ints (football-data) and booleans
    (ledger fallback, written as 'True'/'False'), which a plain to_numeric turns into NaN."""
    if v is None or (isinstance(v, float) and v != v):
        return None
    s = str(v).strip().lower()
    if s in ("1", "1.0", "true"):
        return 1
    if s in ("0", "0.0", "false"):
        return 0
    return None


def _kickoff_of(row) -> pd.Timestamp | None:
    k = pd.to_datetime(row.get("kickoff_ts"), utc=True, errors="coerce")
    if pd.isna(k):
        k = pd.to_datetime(row.get("date"), utc=True, errors="coerce")
    return None if pd.isna(k) else k


def v9_first_sight(snaps: pd.DataFrame) -> pd.DataFrame:
    """One row per fixture_id that v9 ever tipped: side, tier and price of the FIRST pre-kickoff
    snapshot carrying the tip, plus the fixture's first-seen time and latest kickoff."""
    cols = ["fixture_id", "league", "match", "date", "kickoff", "first_seen", "first_tip_ts",
            "side", "tier", "odds"]
    if snaps is None or snaps.empty:
        return pd.DataFrame(columns=cols)
    s = snaps.copy()
    s["_side"] = s["live_side"].astype(str).str.upper()
    span = s.sort_values("snapshot_ts").groupby("fixture_id").agg(
        first_seen=("snapshot_ts", "first"), _last=("snapshot_ts", "last"))
    lastrow = s.sort_values("snapshot_ts").drop_duplicates("fixture_id", keep="last").set_index(
        "fixture_id")
    t = s[s["live_tier"].astype(str).isin(BET_TIERS) & s["_side"].isin(["OVER", "UNDER"])]
    mtk = pd.to_numeric(t.get("minutes_to_kickoff"), errors="coerce")
    t = t[~(mtk < 0)]                     # a tip first seen after kickoff has no pre-match price
    if t.empty:
        return pd.DataFrame(columns=cols)
    f = t.sort_values("snapshot_ts").drop_duplicates("fixture_id", keep="first")
    out = []
    for _, r in f.iterrows():
        fid = r["fixture_id"]
        lr = lastrow.loc[fid]
        try:
            odds = float(r["odds_over25"] if r["_side"] == "OVER" else r["odds_under25"])
        except (TypeError, ValueError):
            odds = None
        out.append({"fixture_id": fid, "league": r["league"], "match": r["match"],
                    "date": lr.get("date"), "kickoff": _kickoff_of(lr),
                    "first_seen": span.loc[fid, "first_seen"], "first_tip_ts": r["snapshot_ts"],
                    "side": r["_side"], "tier": r["live_tier"],
                    "odds": odds if odds and odds > 1.0 else None})
    return pd.DataFrame(out, columns=cols)


#: Re-date HANDOFF: the old fixture_id's last snapshot and the new one's first snapshot are this
#: close (the collector runs hourly in match hours; overnight gaps are up to ~9 h).
REDATE_HANDOFF_HOURS = 12


def _fixture_spans(snaps: pd.DataFrame) -> pd.DataFrame:
    s = snaps.sort_values("snapshot_ts")
    last = s.drop_duplicates("fixture_id", keep="last").set_index("fixture_id")
    out = last[["league", "match", "date"]].copy()
    g = s.groupby("fixture_id")["snapshot_ts"]
    out["first_seen"] = pd.to_datetime(g.first().reindex(out.index), utc=True, errors="coerce")
    out["last_seen"] = pd.to_datetime(g.last().reindex(out.index), utc=True, errors="coerce")
    out["kickoff"] = pd.Series([_kickoff_of(r) for _, r in last.iterrows()], index=out.index,
                               dtype=object)
    return out


def reschedule_chains(snaps: pd.DataFrame) -> dict:
    """fixture_id -> (final fixture_id, hours moved from the ORIGINAL kickoff) for every fixture
    that v9 re-dated. fixture_id contains the date, so a re-dated match gets a new id.

    Evidence of a re-date, not a second meeting of the same clubs (Ireland and MLS play the same
    home pairing twice a season): the same league + match appears under a new fixture_id
      (1) first seen BEFORE the earlier id's kickoff, i.e. while that match was still upcoming, and
      (2) as a HANDOFF: the earlier id is never snapshotted again after the new id appears
          (its last snapshot is no later than REDATE_HANDOFF_HOURS after the new first one).
    Two genuine meetings are snapshotted side by side until the first is played. Fixtures not in a
    chain are absent from the result."""
    if snaps is None or snaps.empty:
        return {}
    sp = _fixture_spans(snaps)
    sp["_key"] = sp["league"].astype(str) + "|" + sp["match"].astype(str).str.strip().str.lower()
    out = {}
    for _, g in sp.groupby("_key"):
        if len(g) < 2:
            continue
        g = g.sort_values("first_seen")
        chain = [g.index[0]]
        for fid in g.index[1:]:
            prev = chain[-1]
            ko_prev = sp.loc[prev, "kickoff"]
            seen = sp.loc[fid, "first_seen"]
            gone = sp.loc[prev, "last_seen"]
            handoff = (pd.notna(seen) and pd.notna(gone)
                       and (gone - seen).total_seconds() <= REDATE_HANDOFF_HOURS * 3600)
            if ko_prev is not None and pd.notna(seen) and seen < ko_prev and handoff:
                chain.append(fid)
            else:
                if len(chain) > 1:
                    _close_chain(chain, sp, out)
                chain = [fid]
        if len(chain) > 1:
            _close_chain(chain, sp, out)
    return out


def _close_chain(chain, sp, out):
    final = chain[-1]
    ko0, kof = sp.loc[chain[0], "kickoff"], sp.loc[final, "kickoff"]
    for fid in chain:
        hrs = None if ko0 is None or kof is None else abs((kof - ko0).total_seconds()) / 3600
        out[fid] = (final, hrs)


def v9_scoreboard_rows(snaps: pd.DataFrame, over_by_fid: dict) -> pd.DataFrame:
    """Every v9 tip, one row per tipped fixture_id, with `counted` and `exclude_reason`.
    Counted rows are one per real match; their P/L is flat 1u at the first-sight price."""
    from src import league_registry as LR
    tips = v9_first_sight(snaps)
    cols = list(tips.columns) + ["settles_on", "over25_result", "pnl", "counted", "exclude_reason"]
    if tips.empty:
        return pd.DataFrame(columns=cols)
    chains = reschedule_chains(snaps)
    tips["settles_on"] = tips["fixture_id"]
    tips["exclude_reason"] = ""
    for i, r in tips.iterrows():
        if LR.is_paper(r["league"]):
            tips.at[i, "exclude_reason"] = "paper_league"
            continue
        if r["fixture_id"] in chains:
            final, hrs = chains[r["fixture_id"]]
            if hrs is None:
                tips.at[i, "exclude_reason"] = "rescheduled_kickoff_unknown"
                continue
            if r["fixture_id"] != final and hrs > RESCHEDULE_SETTLE_HOURS:
                tips.at[i, "exclude_reason"] = f"rescheduled_void(+{hrs:.0f}h)"
                continue
            tips.at[i, "settles_on"] = final
    live = tips[tips["exclude_reason"] == ""]
    # One row per real match: the EARLIEST tip among the fixture_ids that settle on it.
    keep = live.sort_values("first_tip_ts").drop_duplicates("settles_on", keep="first").index
    for i in live.index.difference(keep):
        tips.at[i, "exclude_reason"] = f"same_match_as({tips.at[i, 'settles_on']})"
    tips["over25_result"] = [_over_value(over_by_fid.get(str(f))) for f in tips["settles_on"]]
    pnl = []
    for _, r in tips.iterrows():
        ov = _over_value(r["over25_result"])        # NaN must stay missing, never bool(nan)=True
        g = _grade(r["side"], None if ov is None else bool(ov), r["odds"], r["odds"])
        pnl.append(g["pnl"] if g else None)
    tips["pnl"] = pnl
    tips["counted"] = (tips["exclude_reason"] == "") & tips["pnl"].notna()
    tips.loc[(tips["exclude_reason"] == "") & tips["pnl"].isna(), "exclude_reason"] = "no_result"
    return tips[cols]


def _agg_v9(t: pd.DataFrame) -> dict:
    c = t[t["counted"]] if len(t) else t
    n = len(c)
    if n == 0:
        return {"picks": 0, "W": 0, "L": 0, "pnl": 0.0, "hit": None}
    w = int((c["pnl"] > 0).sum())
    return {"picks": n, "W": w, "L": n - w, "pnl": round(float(c["pnl"].sum()), 2),
            "hit": round(w / n * 100, 1)}


V9_COLS = {"side": "live_first_side", "tier": "live_first_tier", "odds": "live_first_odds",
           "first_tip_ts": "live_first_tip_ts", "settles_on": "live_settles_on",
           "pnl": "live_first_pnl", "counted": "live_counted",
           "exclude_reason": "live_exclude_reason"}


def attach_v9_first_sight(graded: pd.DataFrame, snaps: pd.DataFrame, over_by_fid: dict):
    """The corrected v9 grading as NEW columns beside the legacy live_* ones (kept unchanged)."""
    tips = v9_scoreboard_rows(snaps, over_by_fid)
    if graded.empty:
        return graded, tips
    t = tips.rename(columns=V9_COLS)[["fixture_id"] + list(V9_COLS.values())]
    t["fixture_id"] = t["fixture_id"].astype(str)
    g = graded.copy()
    g["fixture_id"] = g["fixture_id"].astype(str)
    g = g.merge(t, on="fixture_id", how="left")
    # Month of the match a tip settles on (a re-dated fixture settles in its actual month).
    month_of = dict(zip(g["fixture_id"], g["month"]))
    tips = tips.assign(month=[month_of.get(str(f)) for f in tips["settles_on"]])
    return g, tips


def build_scoreboard(graded: pd.DataFrame, tips: pd.DataFrame) -> pd.DataFrame:
    sb = []
    months = sorted({m for m in graded.get("month", pd.Series(dtype=object)).dropna()}
                    | {m for m in tips.get("month", pd.Series(dtype=object)).dropna()})
    for scope in ["overall"] + months:
        g = graded if scope == "overall" else graded[graded["month"] == scope]
        t = tips if scope == "overall" or tips.empty else tips[tips["month"] == scope]
        sb.append({"scope": scope, "system": "v9", **_agg_v9(t)})
        sb.append({"scope": scope, "system": "v11", **_agg(g, "v11", "v11_tier")})
        # Kept beside, never silently replaced: the number this file used to publish as "v9".
        sb.append({"scope": scope, "system": "v9_latest_view (legacy)",
                   **_agg(g, "live", "live_tier")})
    return pd.DataFrame(sb)


def _load_snapshots() -> pd.DataFrame:
    f = config.OUTPUT_DIR / "v11_shadow_snapshots.csv"
    if not f.exists():
        return pd.DataFrame(columns=SNAP_COLS)
    head = pd.read_csv(f, nrows=0).columns
    return pd.read_csv(f, usecols=[c for c in SNAP_COLS if c in head], low_memory=False)


def run():
    log_f = config.OUTPUT_DIR / "v11_shadow_log.csv"
    if not log_f.exists():
        print("[grade] no v11_shadow_log.csv yet — run v11_shadow first"); return
    log = pd.read_csv(log_f)

    # ── RESULTS: actual goals FIRST, v9's tip ledger only as a fallback ──────────────────
    #
    # The ledger holds only fixtures v9 TIPPED, so grading from it settled 94/336 (28%) and left
    # whole leagues at zero: League Two 0/24, Serie B 0/11, Ireland 0/5. That is not merely thin
    # — the graded subset is CONDITIONED ON v9 having disagreed with the market enough to fire a
    # tip, which is the very variable the movement research studies. Every ROI figure drawn from
    # it was selected on the independent variable.
    #
    # football-data.co.uk publishes full results for every league v11 shadows, free and
    # unauthenticated, so the conditioning is removable. The ledger is kept as a fallback rather
    # than deleted: it covers a handful of API-Football-only competitions football-data does not
    # carry, and losing those would trade one gap for another.
    # Keyed by the shadow's own fixture_id (built from the canonical league), not by raw strings.
    fd_lookup: dict = {}
    try:
        from src import results as rs
        leagues = sorted(log["league"].dropna().astype(str).unique())
        res = rs.fetch_results(leagues)
        if not res.empty:
            graded_fx = rs.grade_fixtures(
                log[["fixture_id", "league", "match", "date"]].drop_duplicates("fixture_id"), res)
            for _, g in graded_fx.iterrows():
                if pd.notna(g.get("over25_result")):
                    fd_lookup[str(g["fixture_id"])] = int(g["over25_result"])
    except Exception as e:                                             # noqa: BLE001
        print(f"[grade] football-data results unavailable ({type(e).__name__}: {e}); "
              f"falling back to v9's tip ledger, which is SELECTION-BIASED")

    try:
        led = _ledger_results(_load_v9("bets_ledger.csv"))
    except Exception as e:
        print(f"[grade] could not load v9 results: {e}"); led = None

    n_fd = n_ledger = 0
    rows = []
    over_by_fid: dict = {}
    for _, r in log.iterrows():
        # Actual goals win. A ledger value is only consulted where football-data has nothing.
        over = fd_lookup.get(str(r.get("fixture_id")))
        if over is not None:
            n_fd += 1
        else:
            over = _ledger_over(led, r.get("league"), r.get("match"), r.get("date"))
            if over is not None:
                n_ledger += 1
        over_by_fid[str(r.get("fixture_id"))] = _over_value(over)
        o_over, o_under = r.get("odds_over25"), r.get("odds_under25")
        v9 = _grade(r.get("live_side"), over, o_over, o_under) if str(r.get("live_tier")) in BET_TIERS else None
        v11 = _grade(r.get("v11_side"), over, o_over, o_under) if str(r.get("v11_tier")) in BET_TIERS else None
        rows.append({
            "fixture_id": r.get("fixture_id"),
            "date": r.get("date"), "month": str(r.get("date"))[:7] if pd.notna(r.get("date")) else None,
            "league": r.get("league"),
            "match": r.get("match"), "over25_result": over,
            "live_side": r.get("live_side"), "live_tier": r.get("live_tier"),
            "live_pnl": v9["pnl"] if v9 else None,
            "v11_side": r.get("v11_side"), "v11_tier": r.get("v11_tier"),
            "v11_pnl": v11["pnl"] if v11 else None,
        })
    graded = pd.DataFrame(rows)
    graded, tips = attach_v9_first_sight(graded, _load_snapshots(), over_by_fid)
    _write_derived(graded, "v11_graded.csv")
    sb_df = build_scoreboard(graded, tips)
    _write_derived(sb_df, "v11_scoreboard.csv")

    n_res = int(graded["over25_result"].notna().sum())
    print(f"[grade] graded {n_res}/{len(graded)} fixtures with known results")
    # Provenance split, printed every run. The ledger share is the part that carries selection
    # bias, so a reader must be able to see how much of the sample still depends on it rather
    # than having to trust that the fix worked.
    print(f"[grade]   from actual results (football-data): {n_fd:,}")
    print(f"[grade]   from v9 tip ledger (SELECTION-BIASED, fallback only): {n_ledger:,}")
    if n_ledger and not n_fd:
        print("[grade]   WARNING: every graded row came from the tip ledger. Result-based "
              "figures are conditioned on v9 having tipped the fixture.")
    ov = sb_df[sb_df["scope"] == "overall"]
    for _, r in ov.iterrows():
        hit = f"{r['hit']}%" if r["hit"] is not None else "-"
        print(f"  {r['system']:4}: {r['picks']:3} picks | {r['W']}W-{r['L']}L | "
              f"P/L {r['pnl']:+.2f}u | hit {hit}")


if __name__ == "__main__":
    run()
