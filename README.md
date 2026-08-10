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
- [x] Shadow log — reads v9 public data, logs market-first decisions (`v11_shadow.py`)
- [x] **Grader + scoreboard** — grades v9's & v11's picks vs actual results, writes a
  head-to-head (picks / W-L / P&L units / hit%), overall + monthly (`v11_grade.py` →
  `output/v11_graded.csv`, `output/v11_scoreboard.csv`)
- [ ] **CLV per pick** — capture the closing line for v11's own picks + CLV column
- [ ] **Full result coverage** — fetch football-data / API-Football results so v11 picks on
  fixtures v9 didn't tip also get graded (v1 uses v9's ledger → only the overlap)
- [ ] **Multi-book feed** — V11's own OddsAPI per-book fetch → consensus + best-price
  (line-shopping); may be limited in our thin-book leagues
- [ ] Monthly review → decide what (if anything) graduates toward v9

## The monthly review
Open `output/v11_scoreboard.csv` — the `overall` rows show v9 vs v11 (picks, W-L, P/L in
units, hit%). Right now v11 is far more selective than v9 (market-first NO_BET default), so
compare *quality* (P/L per pick, hit%) not volume.

## Run
```bash
pip install -r requirements.txt
python scripts/v11_shadow.py     # reads v9 public data, appends output/v11_shadow_log.csv
```
CI (`.github/workflows/v11_collect.yml`) runs it twice/hour and commits the log. Reads only
v9's **public** data — no secrets required for the shadow.
