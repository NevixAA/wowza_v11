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




def _clip(p):
    return min(1 - 1e-6, max(1e-6, p))


def _brier(p, y):
    return (p - y) ** 2


def _logloss(p, y):
    p = _clip(p)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def _as_bool(v):
    """over25_result -> True/False/None, tolerating BOTH encodings present in the file.

    v11_graded.csv carries a mix: the football-data path writes 1/0, and the older tip-ledger
    path writes True/False. `int("True")` raises, which is how this surfaced. Returning None for
    anything unrecognised means an unparseable result is EXCLUDED rather than silently coerced —
    a wrong outcome label is far worse than a missing one.
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().lower()
    if s in ("1", "1.0", "true", "yes", "over", "win"):
        return True
    if s in ("0", "0.0", "false", "no", "under", "loss"):
        return False
    return None


def run():
    log_f = config.OUTPUT_DIR / "v11_shadow_log.csv"
    if not log_f.exists():
        print("[residual] no shadow log yet"); return
    log = pd.read_csv(log_f)

    # ── RESULTS COME FROM v11_graded.csv, THE SINGLE GRADED SOURCE ──────────────────────
    #
    # This used to call _result_lookup(_load_v9("bets_ledger.csv")) — a SECOND, independent
    # grading path that re-derived outcomes from v9's tip ledger. The ledger holds only fixtures
    # v9 TIPPED, so this experiment was permanently capped at whatever v9 had bet on:
    #
    #     v11_graded.csv settled fixtures    344
    #     residual n                         194
    #
    # The 150-fixture gap was not a refresh failure — the file commits daily — it was residual
    # reading a worse source. `v11_grade.py` was moved to actual football-data results on
    # 2026-08-23; this path was missed, so half the fix landed. Worse, the sample it did use was
    # CONDITIONED ON v9 having disagreed with the market enough to fire a tip, which is the exact
    # variable a market-relative experiment is measuring.
    #
    # Reading the graded file instead gives one grading path for the whole repo. `v11_grade.py`
    # runs immediately before this step in the workflow, so the file is current by construction.
    lookup: dict = {}
    graded_f = config.OUTPUT_DIR / "v11_graded.csv"
    n_from_graded = 0
    if graded_f.exists():
        g = pd.read_csv(graded_f)
        if {"match", "date", "over25_result"} <= set(g.columns):
            for _, gr in g.iterrows():
                if pd.isna(gr.get("over25_result")):
                    continue
                mm = str(gr.get("match", ""))
                if " vs " not in mm:
                    continue
                hh, aa = mm.split(" vs ", 1)
                val = _as_bool(gr["over25_result"])
                if val is None:
                    continue
                lookup[(_norm(hh), _norm(aa), str(gr.get("date", ""))[:10])] = val
                n_from_graded += 1
    # Ledger kept ONLY as a fallback for fixtures the graded file has no row for — never as the
    # primary source. Its entries do not overwrite a real graded result.
    n_from_ledger = 0
    try:
        for k, v in _result_lookup(_load_v9("bets_ledger.csv")).items():
            if k not in lookup:
                lookup[k] = v
                n_from_ledger += 1
    except Exception:
        pass
    print(f"[residual] results: {n_from_graded} from v11_graded.csv, "
          f"{n_from_ledger} extra from the v9 tip ledger (fallback only)")

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
    _write_derived(res, "v11_residual.csv")
    print(f"[residual] {len(d)} graded fixtures -> v11_residual.csv")
    if len(res):
        print(res[res["scope"] == "overall"].to_string(index=False))
    else:
        print("[residual] 0 results yet — accumulating (needs settled fixtures to score)")


if __name__ == "__main__":
    run()
