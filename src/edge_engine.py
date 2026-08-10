"""
Market-first edge engine — v10 redesign
========================================
Replaces the model-first EV / relative-edge tiering that produced the −57% OOS result.

WHY (see PlayerProps_Research/): the model has ~zero information over the market price
(AUC_marginal ≈ 0.5; its disagreements with the price are ANTI-predictive). So a real,
repeatable edge cannot come from "model disagrees with book." It must come from MARKET
STRUCTURE — proper de-vigging, model↔market blending with heavy market weight, uncertainty
shrinkage, and (above all) Closing-Line Value. This engine therefore DEFAULTS TO NO BET and
only emits a tier when several independent conditions all hold.

Per-candidate pipeline:
    power_devig(over,under)      -> market fair prob (two-sided; Power method)
    market_baseline(books)       -> consensus (exchange > cross-book median > single book)
    blend(p_model, p_market)     -> w·model + (1−w)·market ; w shrinks on long odds / low n
    lower_bound_prob(...)        -> p_lb = p_blend − uncertainty penalty
    ev_lb / abs_edge             -> conservative EV + odds-neutral edge
    classify(...)                -> NO_BET default; hard caps; CLV gate

Cross-check additions wired in: settlement-alignment gate (#1), longshot hard-cap + penalties,
exchange-first baseline (#4), CLV gate (primary). White/SPA/FDR selection tests live in the
validation module; Power devig (#3) is the default here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median

# ── tunables (odds-band aware) ────────────────────────────────────────────────
MAX_BET_ODDS      = 6.0     # hard cap: never bet longer than this (longshot machine off)
RARE_EVENT_MARKETS = {"sot2", "sot3", "goals2"}   # thin base rate -> extra evidence required
TAU               = {"default": 400.0}            # shrinkage strength (pseudo-count)
Z90               = 1.2816  # one-sided 90% normal quantile for the lower bound
DEFAULT_TX_COST   = 0.02    # slippage + execution uncertainty

# EV_lb / abs_edge floors by odds band (stricter as odds lengthen)
_BANDS = [  # (max_odds, max_blend_w, min_n_eff, ev_lb_floor, abs_edge_floor)
    (2.00, 0.50, 500,  0.010, 0.030),
    (3.50, 0.35, 800,  0.020, 0.020),
    (6.00, 0.20, 1500, 0.040, 0.025),
]


def _band(odds: float):
    for hi, w, n, ev, ae in _BANDS:
        if odds <= hi:
            return {"max_w": w, "min_n": n, "ev_floor": ev, "abs_floor": ae}
    return None  # odds > 6.0 -> no band -> No Bet


# ── de-vig ────────────────────────────────────────────────────────────────────
def proportional_devig(over_odds: float, under_odds: float | None) -> float | None:
    if not over_odds or over_odds <= 1.0:
        return None
    r_o = 1.0 / over_odds
    if not under_odds or under_odds <= 1.0:
        return None  # one-sided -> cannot de-vig honestly (flag OBSERVE upstream)
    r_u = 1.0 / under_odds
    return r_o / (r_o + r_u)


def power_devig(over_odds: float, under_odds: float | None, tol: float = 1e-9) -> float | None:
    """Power-method fair prob for the OVER side. Solve alpha s.t.
    (1/o_over)^alpha + (1/o_under)^alpha = 1, return (1/o_over)^alpha.
    Preferred over multiplicative normalization for favourite–longshot structure.
    Falls back to None when one-sided (no honest de-vig possible)."""
    if not over_odds or not under_odds or over_odds <= 1.0 or under_odds <= 1.0:
        return None
    ro, ru = 1.0 / over_odds, 1.0 / under_odds
    lo, hi = 0.5, 5.0                      # bracket alpha
    for _ in range(100):
        a = (lo + hi) / 2
        s = ro ** a + ru ** a
        if abs(s - 1.0) < tol:
            break
        if s > 1.0:                        # sum too big -> raise alpha (shrinks probs)
            lo = a
        else:
            hi = a
    return ro ** ((lo + hi) / 2)


# ── market baseline (exchange > cross-book median > single book) ──────────────
def market_baseline(book_fair_probs: dict[str, float], exchange_prob: float | None = None) -> float | None:
    """Consensus fair prob. Exchange price is the sharpest anchor; else weighted/plain
    median of per-book de-vigged probs; else the single book."""
    if exchange_prob is not None:
        return float(exchange_prob)
    vals = [v for v in book_fair_probs.values() if v is not None]
    if not vals:
        return None
    return float(median(vals))


# ── model↔market blend ────────────────────────────────────────────────────────
def blend_weight(odds: float, n_eff: int, market: str, w_cap: float = 0.30) -> float:
    """Weight on the MODEL. Market is the baseline; the model is a residual. `w_cap` is the
    per-segment ceiling: 0.30 for zero-signal segments (props, AUC≈0.5), higher for segments
    with genuine measured AUC (our obscure-league team O/U moat). Shrinks on long odds (band
    max), low sample, and rare markets. w = min(w_cap, band_max) with those shrinks applied."""
    b = _band(odds)
    if b is None:
        return 0.0
    w = min(w_cap, b["max_w"])                       # ceiling = min(segment cap, band cap)
    if n_eff < b["min_n"]:
        w *= n_eff / max(1, b["min_n"])
    if market in RARE_EVENT_MARKETS:
        w *= 0.5
    return max(0.0, min(w_cap, w))


def blend(p_model: float, p_market: float, w: float) -> float:
    return w * p_model + (1.0 - w) * p_market


# ── uncertainty-aware lower bound ─────────────────────────────────────────────
def lower_bound_prob(p_blend: float, ece: float, n_eff: int, market: str,
                     lineup_penalty: float = 0.0, stale_penalty: float = 0.0) -> float:
    """p_lb = p_blend − u. u = calibration error + sampling SE (one-sided 90%) +
    lineup/stale/rare penalties. Conservative on purpose."""
    tau = TAU.get(market, TAU["default"])
    se = math.sqrt(max(p_blend * (1 - p_blend), 0.0) / (n_eff + tau))
    rare_pen = 0.02 if market in RARE_EVENT_MARKETS else 0.0
    u = ece + Z90 * se + lineup_penalty + stale_penalty + rare_pen
    return max(0.0, p_blend - u)


def ev_lb(p_lb: float, best_odds: float, tx_cost: float = DEFAULT_TX_COST) -> float:
    return best_odds * p_lb - 1.0 - tx_cost


# ── decision ──────────────────────────────────────────────────────────────────
@dataclass
class Candidate:
    market: str
    over_odds: float
    under_odds: float | None = None
    best_odds: float | None = None          # best price available across books
    p_model: float = 0.5
    book_fair_probs: dict = field(default_factory=dict)
    exchange_prob: float | None = None
    n_eff: int = 0
    ece: float = 0.02
    lineup_confirmed: bool = False
    two_sided: bool = True
    rolling_clv: float | None = None        # positive => selection historically beat close
    settlement_aligned: bool = True         # #1 gate: our label == book's settlement feed
    model_w_cap: float = 0.30               # per-segment model-weight ceiling (moat > 0.30)


def classify(c: Candidate) -> dict:
    """Return {tier, reason, p_market, p_blend, p_lb, ev_lb, abs_edge}. NO_BET by default."""
    best = c.best_odds or c.over_odds
    out = {"tier": "NO_BET", "reason": "", "p_market": None, "p_blend": None,
           "p_lb": None, "ev_lb": None, "abs_edge": None}

    # ── hard gates ──
    if not c.settlement_aligned:
        out["reason"] = "settlement feed not aligned (odds vs outcome definition)"; return out
    if best > MAX_BET_ODDS:
        out["reason"] = f"longshot hard-cap (odds {best} > {MAX_BET_ODDS})"; return out
    if not c.two_sided or not c.under_odds:
        out["reason"] = "one-sided market — cannot de-vig honestly (OBSERVE)"; out["tier"] = "OBSERVE"; return out
    if not c.lineup_confirmed:
        out["reason"] = "lineup not confirmed"; out["tier"] = "OBSERVE"; return out

    p_market = power_devig(c.over_odds, c.under_odds) or proportional_devig(c.over_odds, c.under_odds)
    if p_market is None:
        out["reason"] = "no honest market baseline"; return out
    p_market = market_baseline(c.book_fair_probs, c.exchange_prob) or p_market

    w = blend_weight(best, c.n_eff, c.market, c.model_w_cap)
    p_blend = blend(c.p_model, p_market, w)
    p_lb = lower_bound_prob(p_blend, c.ece, c.n_eff, c.market,
                            lineup_penalty=0.0 if c.lineup_confirmed else 0.03)
    _ev = ev_lb(p_lb, best)
    ae = p_blend - p_market
    out.update(p_market=round(p_market, 4), p_blend=round(p_blend, 4),
               p_lb=round(p_lb, 4), ev_lb=round(_ev, 4), abs_edge=round(ae, 4))

    b = _band(best)
    if b is None:
        out["reason"] = "no odds band"; return out
    if c.market in RARE_EVENT_MARKETS and c.n_eff < b["min_n"] * 2:
        out["reason"] = "rare market — insufficient evidence"; out["tier"] = "OBSERVE"; return out
    if _ev < b["ev_floor"] or ae < b["abs_floor"]:
        out["reason"] = f"below floors (ev_lb {_ev:.3f}<{b['ev_floor']} or abs_edge {ae:.3f}<{b['abs_floor']})"
        return out

    # ── CLV gate: SNIPER/MARKSMAN require a positive rolling CLV track; else VALUABLE only ──
    clv_ok = (c.rolling_clv is not None and c.rolling_clv > 0)
    if _ev >= b["ev_floor"] * 2 and ae >= b["abs_floor"] * 1.5 and clv_ok:
        out["tier"] = "SNIPER"
    elif _ev >= b["ev_floor"] * 1.3 and clv_ok:
        out["tier"] = "MARKSMAN"
    else:
        out["tier"] = "VALUABLE"
        out["reason"] = "passes floors; CLV not yet validated -> VALUABLE (flat/paper)" if not clv_ok else "moderate edge"
    return out
