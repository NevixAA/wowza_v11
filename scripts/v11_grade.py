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
import re
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


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _result_lookup(bl: pd.DataFrame) -> dict:
    """(home, away, date) -> over_happened (bool). Derived from v9's tipped side + WIN/LOSS
    (bets_ledger has no total_goals column): over 2.5 happened iff a tipped OVER won or a
    tipped UNDER lost."""
    if bl is None or bl.empty or "result" not in bl.columns or "side" not in bl.columns:
        return {}
    look = {}
    for _, r in bl.iterrows():
        res = str(r.get("result", "")).upper()
        side = str(r.get("side", "")).upper()
        if res not in ("WIN", "LOSS") or side not in ("OVER", "UNDER"):
            continue
        over = (side == "OVER" and res == "WIN") or (side == "UNDER" and res == "LOSS")
        key = (_norm(r.get("home_team", "")), _norm(r.get("away_team", "")),
               str(r.get("match_date", ""))[:10])
        look[key] = bool(over)
    return look


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
    fd_lookup: dict = {}
    try:
        from src import results as rs
        leagues = sorted(log["league"].dropna().astype(str).unique())
        res = rs.fetch_results(leagues)
        if not res.empty:
            graded_fx = rs.grade_fixtures(
                log[["league", "match", "date"]].drop_duplicates(), res)
            for _, g in graded_fx.iterrows():
                if pd.notna(g.get("over25_result")):
                    fd_lookup[(str(g["league"]), str(g["match"]),
                               str(g["date"])[:10])] = int(g["over25_result"])
    except Exception as e:                                             # noqa: BLE001
        print(f"[grade] football-data results unavailable ({type(e).__name__}: {e}); "
              f"falling back to v9's tip ledger, which is SELECTION-BIASED")

    try:
        lookup = _result_lookup(_load_v9("bets_ledger.csv"))
    except Exception as e:
        print(f"[grade] could not load v9 results: {e}"); lookup = {}

    n_fd = n_ledger = 0
    rows = []
    for _, r in log.iterrows():
        h, a = str(r.get("match", "")).split(" vs ", 1) if " vs " in str(r.get("match", "")) else ("", "")
        # Actual goals win. A ledger value is only consulted where football-data has nothing.
        over = fd_lookup.get((str(r.get("league")), str(r.get("match")),
                              str(r.get("date", ""))[:10]))
        if over is not None:
            n_fd += 1
        else:
            over = lookup.get((_norm(h), _norm(a), str(r.get("date", ""))[:10]))
            if over is not None:
                n_ledger += 1
        o_over, o_under = r.get("odds_over25"), r.get("odds_under25")
        v9 = _grade(r.get("live_side"), over, o_over, o_under) if str(r.get("live_tier")) in BET_TIERS else None
        v11 = _grade(r.get("v11_side"), over, o_over, o_under) if str(r.get("v11_tier")) in BET_TIERS else None
        rows.append({
            "date": r.get("date"), "month": str(r.get("date", ""))[:7], "league": r.get("league"),
            "match": r.get("match"), "over25_result": over,
            "live_side": r.get("live_side"), "live_tier": r.get("live_tier"),
            "live_pnl": v9["pnl"] if v9 else None,
            "v11_side": r.get("v11_side"), "v11_tier": r.get("v11_tier"),
            "v11_pnl": v11["pnl"] if v11 else None,
        })
    graded = pd.DataFrame(rows)
    _write_derived(graded, "v11_graded.csv")

    # ── scoreboard: overall + per month ──
    sb = []
    scopes = [("overall", graded)] + [(m, g) for m, g in graded.groupby("month") if m]
    for scope, g in scopes:
        v9a = _agg(g, "live", "live_tier")
        v11a = _agg(g, "v11", "v11_tier")
        sb.append({"scope": scope, "system": "v9", **v9a})
        sb.append({"scope": scope, "system": "v11", **v11a})
    sb_df = pd.DataFrame(sb)
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
