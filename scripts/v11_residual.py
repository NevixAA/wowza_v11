"""
V11 residual-vs-market experiment — THE core test (from the external review).
=============================================================================
The real question isn't "what's Wowza's standalone AUC?" It's: **does the model add
information AFTER the market price?** For every fixture with a known result, compare

    MARKET-only        : de-vigged consensus probability p_market
    MARKET + WOWZA     : model-blended probability p_blend = w·p_model + (1-w)·p_market

on Brier score and log-loss (lower = better). Interpretation:
- if MARKET+WOWZA ≈ MARKET → the model carries ~no incremental info → weight should →0.
- if MARKET+WOWZA beats MARKET out-of-sample in a segment (e.g. Ligue 2) → that's a genuine
  information edge that deserves money.

v1 caveats: single-book consensus (power de-vig of v9's over/under at log time, not the close);
result derived from v9's ledger (tipped side + WIN/LOSS → over/under-2.5 outcome). Sparse until
results accumulate — the script reports n so you don't over-read a tiny sample.
Output: output/v11_residual.csv
"""
import math
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "scripts"))
import config
from src.edge_engine import power_devig, proportional_devig, blend
from v11_shadow import _blend_weight, MOAT_W_CAP, DEFAULT_W_CAP, N_EFF, _load_v9
from v11_grade import _result_lookup, _norm


def _clip(p):
    return min(1 - 1e-6, max(1e-6, p))


def _brier(p, y):
    return (p - y) ** 2


def _logloss(p, y):
    p = _clip(p)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def run():
    log_f = config.OUTPUT_DIR / "v11_shadow_log.csv"
    if not log_f.exists():
        print("[residual] no shadow log yet"); return
    log = pd.read_csv(log_f)
    try:
        lookup = _result_lookup(_load_v9("bets_ledger.csv"))
    except Exception:
        lookup = {}

    rows = []
    for _, r in log.iterrows():
        m = str(r.get("match", ""))
        if " vs " not in m:
            continue
        h, a = m.split(" vs ", 1)
        over = lookup.get((_norm(h), _norm(a), str(r.get("date", ""))[:10]))
        if over is None:                      # no result yet
            continue
        try:
            oo = float(r.get("odds_over25")); ou = float(r.get("odds_under25"))
            p_model = float(r.get("p_model_over"))
        except (TypeError, ValueError):
            continue
        p_mkt = power_devig(oo, ou) or proportional_devig(oo, ou)
        if p_mkt is None:
            continue
        w = _blend_weight(oo, N_EFF, "over25",
                          MOAT_W_CAP.get(str(r.get("model_type", "")), DEFAULT_W_CAP))
        p_bl = blend(p_model, p_mkt, w)
        rows.append({"league": r.get("league"), "model_type": r.get("model_type"),
                     "y": 1.0 if over else 0.0, "p_market": p_mkt, "p_blend": p_bl,
                     "residual": round(p_model - p_mkt, 4)})

    d = pd.DataFrame(rows)
    out = []

    def _agg(scope, g):
        n = len(g)
        if n == 0:
            return
        bm = g.apply(lambda x: _brier(x.p_market, x.y), axis=1).mean()
        bb = g.apply(lambda x: _brier(x.p_blend, x.y), axis=1).mean()
        lm = g.apply(lambda x: _logloss(x.p_market, x.y), axis=1).mean()
        lb = g.apply(lambda x: _logloss(x.p_blend, x.y), axis=1).mean()
        out.append({"scope": scope, "n": n,
                    "brier_market": round(bm, 4), "brier_market_wowza": round(bb, 4),
                    "logloss_market": round(lm, 4), "logloss_market_wowza": round(lb, 4),
                    "wowza_helps": bool(bb < bm and lb < lm)})

    if len(d):
        _agg("overall", d)
        for lg, g in d.groupby("league"):
            _agg(str(lg), g)
    res = pd.DataFrame(out)
    res.to_csv(config.OUTPUT_DIR / "v11_residual.csv", index=False)
    print(f"[residual] {len(d)} graded fixtures -> v11_residual.csv")
    if len(res):
        print(res[res["scope"] == "overall"].to_string(index=False))
    else:
        print("[residual] 0 results yet — accumulating (needs settled fixtures to score)")


if __name__ == "__main__":
    run()
