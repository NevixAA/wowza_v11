# Wowza V11 — Market-Intelligence shadow (data collection)

**A separate, data-collection-only sibling of the frozen v9 live system.** V11 does **not**
send Telegram, has **no dashboard**, places **no stakes**, and **never touches v9**. It reads
v9's public committed data from GitHub and logs what a **market-first** engine *would* decide,
so V11 can be compared to v9's live results at a **monthly review**.

## Why market-first (vs v9's model-first)
v9's live edge is `model_prob − 1/bookmaker_odds`. Our own OOS research showed that
"model disagrees with the book" is **not** a repeatable edge (props −41% to −57%; AUC≈0.5) —
it's a longshot machine. Serious value-betting (OddsJam / RebelBetting / Trademate / Betfair)
is **market-first**: de-vig → consensus fair price → model as a small residual → CLV. V11 tests
that architecture in the shadow.

## Pipeline (`src/edge_engine.py`)
`de-vig (power method) → consensus p_market → blend (model ≤ per-segment cap) → uncertainty
lower bound → EV lower bound → CLV gate → longshot hard-cap` — **defaults to NO_BET**, only
tiers when several independent conditions hold. Per-segment model-weight cap: new-format 0.45,
standard 0.40, else 0.30 (props were the 0.30 zero-signal case; team O/U has real AUC).

## What it collects
`output/v11_shadow_log.csv` — one row per upcoming fixture: the V11 decision (side / tier /
p_market / p_blend / EV-lb / edge) next to v9's live tip, updated to the latest pre-KO snapshot.

## Roadmap
- [x] Shadow log (this) — reads v9 public data, logs market-first decisions
- [ ] **Grader + CLV** — join picks to results, capture closing line, compute CLV per pick
- [ ] **v9-vs-V11 scoreboard** — hit-rate / P&L (units) / CLV side by side, cumulative
- [ ] **Multi-book feed** — V11's own OddsAPI per-book fetch → consensus + best-price
  (line-shopping) — the main missing edge source
- [ ] Monthly review → decide what (if anything) graduates toward v9

## Run
```bash
pip install -r requirements.txt
python scripts/v11_shadow.py     # reads v9 public data, appends output/v11_shadow_log.csv
```
CI (`.github/workflows/v11_collect.yml`) runs it twice/hour and commits the log. Reads only
v9's **public** data — no secrets required for the shadow.
