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
N_EFF = 1000
ECE = 0.02


def _load_v9(name: str) -> pd.DataFrame:
    """Read a v9 output CSV: local override output/_v9_<name> if present (for testing), else
    fetch from the v9 public repo."""
    local = config.OUTPUT_DIR / f"_v9_{name}"
    if local.exists():
        return pd.read_csv(local)
    r = requests.get(f"{config.V9_RAW_BASE}/{name}", timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


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


def _decide(p_model, o_bet, o_other, n_eff, ece, rclv, w_cap):
    out = {"tier": "NO_BET", "p_market": None, "p_blend": None, "ev_lb": None, "abs_edge": None}
    best = o_bet
    if not o_bet or o_bet <= 1.0 or best > MAX_BET_ODDS:
        return out
    if not o_other or o_other <= 1.0:
        out["tier"] = "OBSERVE"; return out
    p_market = power_devig(o_bet, o_other) or proportional_devig(o_bet, o_other)
    if p_market is None:
        return out
    p_market = market_baseline({}, None) or p_market
    w = _blend_weight(best, n_eff, "over25", w_cap)
    p_blend = blend(p_model, p_market, w)
    p_lb = lower_bound_prob(p_blend, ece, n_eff, "over25")
    _ev = ev_lb(p_lb, best)
    ae = p_blend - p_market
    out.update(p_market=round(p_market, 4), p_blend=round(p_blend, 4),
               ev_lb=round(_ev, 4), abs_edge=round(ae, 4))
    b = _band(best)
    if b is None or _ev < b["ev_floor"] or ae < b["abs_floor"]:
        return out
    clv_ok = (rclv is not None and rclv > 0)
    if _ev >= b["ev_floor"] * 2 and ae >= b["abs_floor"] * 1.5 and clv_ok:
        out["tier"] = "SNIPER"
    elif _ev >= b["ev_floor"] * 1.3 and clv_ok:
        out["tier"] = "MARKSMAN"
    else:
        out["tier"] = "VALUABLE"
    return out


def _rolling_clv_by_league(bl: pd.DataFrame) -> dict:
    if bl is None or bl.empty:
        return {}
    if "source" in bl.columns:
        bl = bl[bl["source"].astype(str) == "live"]
    bl = bl.copy()
    bl["clv_pct"] = pd.to_numeric(bl.get("clv_pct"), errors="coerce")
    bl = bl.dropna(subset=["clv_pct"])
    if bl.empty or "league" not in bl.columns:
        return {}
    return bl.groupby("league")["clv_pct"].mean().to_dict()


def run():
    df = _load_v9("predictions.csv")
    try:
        rclv = _rolling_clv_by_league(_load_v9("bets_ledger.csv"))
    except Exception:
        rclv = {}
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    rows = []
    for _, r in df.iterrows():
        try:
            o_over = float(r.get("odds_over25")); o_under = float(r.get("odds_under25"))
            p_over = float(r.get("p_over25"))
        except (TypeError, ValueError):
            continue
        if not (o_over > 1.0 and o_under > 1.0 and 0.0 <= p_over <= 1.0):
            continue
        if str(r.get("date", ""))[:10] < today:
            continue
        lg = str(r.get("league", "")); rc = rclv.get(lg)
        w_cap = MOAT_W_CAP.get(str(r.get("model_type", "")), DEFAULT_W_CAP)
        over = _decide(p_over, o_over, o_under, N_EFF, ECE, rc, w_cap)
        under = _decide(1.0 - p_over, o_under, o_over, N_EFF, ECE, rc, w_cap)
        side, best = (("OVER", over) if TIER_RANK[over["tier"]] >= TIER_RANK[under["tier"]]
                      else ("UNDER", under))
        if best["tier"] in ("NO_BET", "OBSERVE"):
            side = "-"
        rows.append({
            "snapshot_ts": ts, "date": str(r.get("date", ""))[:10], "league": lg,
            "match": f"{r.get('home_team', '')} vs {r.get('away_team', '')}",
            "model_type": r.get("model_type", ""),
            "odds_over25": o_over, "odds_under25": o_under,
            "live_side": r.get("best_side", r.get("bet", "")), "live_tier": r.get("signal_tier", ""),
            "p_model_over": round(p_over, 3),
            "v11_side": side, "v11_tier": best["tier"], "v11_p_market": best["p_market"],
            "v11_p_blend": best["p_blend"], "v11_ev_lb": best["ev_lb"],
            "v11_abs_edge": best["abs_edge"],
            "rolling_clv_pct": round(rc, 2) if rc is not None else "",
        })
    if not rows:
        print("[v11] no upcoming fixtures"); return 0

    new = pd.DataFrame(rows)
    out_f = config.OUTPUT_DIR / "v11_shadow_log.csv"
    if out_f.exists():
        try:
            new = pd.concat([pd.read_csv(out_f), new], ignore_index=True)
        except Exception:
            pass
    new = new.sort_values("snapshot_ts").drop_duplicates(subset=["date", "match"], keep="last")
    new.to_csv(out_f, index=False)
    n_bet = int(new["v11_tier"].isin(["SNIPER", "MARKSMAN", "VALUABLE"]).sum())
    print(f"[v11] logged {len(rows)} fixtures ({n_bet} V11 bets) -> {out_f.name} (tracked {len(new)})")
    return len(rows)


if __name__ == "__main__":
    run()
