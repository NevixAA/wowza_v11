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
    try:
        lookup = _result_lookup(_load_v9("bets_ledger.csv"))
    except Exception as e:
        print(f"[grade] could not load v9 results: {e}"); lookup = {}

    rows = []
    for _, r in log.iterrows():
        h, a = str(r.get("match", "")).split(" vs ", 1) if " vs " in str(r.get("match", "")) else ("", "")
        over = lookup.get((_norm(h), _norm(a), str(r.get("date", ""))[:10]))
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
    graded.to_csv(config.OUTPUT_DIR / "v11_graded.csv", index=False)

    # ── scoreboard: overall + per month ──
    sb = []
    scopes = [("overall", graded)] + [(m, g) for m, g in graded.groupby("month") if m]
    for scope, g in scopes:
        v9a = _agg(g, "live", "live_tier")
        v11a = _agg(g, "v11", "v11_tier")
        sb.append({"scope": scope, "system": "v9", **v9a})
        sb.append({"scope": scope, "system": "v11", **v11a})
    sb_df = pd.DataFrame(sb)
    sb_df.to_csv(config.OUTPUT_DIR / "v11_scoreboard.csv", index=False)

    n_res = int(graded["over25_result"].notna().sum())
    print(f"[grade] graded {n_res}/{len(graded)} fixtures with known results")
    ov = sb_df[sb_df["scope"] == "overall"]
    for _, r in ov.iterrows():
        hit = f"{r['hit']}%" if r["hit"] is not None else "-"
        print(f"  {r['system']:4}: {r['picks']:3} picks | {r['W']}W-{r['L']}L | "
              f"P/L {r['pnl']:+.2f}u | hit {hit}")


if __name__ == "__main__":
    run()
