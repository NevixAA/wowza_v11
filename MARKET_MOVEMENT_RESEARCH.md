# Market Movement / Entry Timing Research

> **THIS IS NOT A LIVE BETTING STRATEGY.** Nothing in this document or the code it describes
> selects a bet, sizes a stake, sends a notification, or feeds any deployment path. It is
> measurement. The graduation criteria in section 9 are not met and are not close to being met.

---

## 1. The hypothesis

v11's outcome test says Wowza does not improve on the market at predicting match results:

| | Brier | LogLoss |
|---|---|---|
| MARKET only | 0.2272 | 0.6459 |
| MARKET + WOWZA | 0.2273 | 0.6461 |

A dead heat, and if anything marginally worse. That is a real result and it is not being argued
with. But it answers one question — *does Wowza know the outcome better than the price does?* —
and there is a different question it does not answer:

> Does Wowza's **disagreement with the market** predict **where the market moves next**?

These can both be true at once. If Wowza reaches certain information earlier than the broad
market, its disagreement would forecast the pre-kickoff price path even though the closing price
eventually absorbs the same information and ends up just as accurate. The value proposition
would then be:

- **not** "Wowza beats bookmakers"
- **but** "Wowza sometimes gets to the future market price first"

That is a price-discovery / entry-timing claim, and it is what this research tests.

## 2. Definitions

For every pre-kickoff snapshot:

```
p_model             model probability of OVER 2.5
p_market_entry      de-vigged consensus probability of OVER at that snapshot
residual         =  p_model - p_market_entry
abs_residual     =  |residual|

p_market_close      de-vigged consensus at the LAST PRE-KICKOFF snapshot of that fixture
market_move      =  p_market_close - p_market_entry
signed_market_move = market_move * sign(residual)
toward_wowza     =  signed_market_move > 0
```

`sign(residual)` is what makes the algebra work in both directions. If the model prefers UNDER
(residual < 0) and the over-probability falls (market_move < 0), then (−)×(−) = + — the price
moved to the model's side. No special-casing.

### The identity you must know before reading any output

Because `p_market` is an OVER probability, and our side is OVER when residual > 0 and UNDER when
residual < 0:

```
side OVER :  clv_prob = p_close     - p_entry     =  market_move
side UNDER:  clv_prob = (1-p_close) - (1-p_entry) = -market_move
          =>  clv_prob == market_move * sign(residual) == signed_market_move
```

**Signed market movement IS closing-line value measured in fair-probability space.** They are one
measurement, not two independent confirmations. A report that presents "the market moved toward
us" and "we got positive CLV" as two agreeing findings is double-counting a single number.

They separate only for `clv_pct`, computed from actual **executable odds**, which carry vig and
book-specific pricing. That is the only CLV figure here that is independent of the movement
figure, and it is the one that would decide whether any of this is monetisable.

### Sign handling for UNDER

Signed movement is direction-agnostic; **executable CLV is not**. It must use the odds of the
side we would actually back — `odds_under25` when the model prefers UNDER. Using `odds_over25`
for both sides would produce a confident and entirely fictitious CLV series. `price_side_odds()`
in `src/movement.py` is the only place that selects a side, and it is unit-tested in both
directions with cases constructed so that the wrong-odds bug yields a *wrong sign*, not merely a
different number.

## 3. The two statistical traps

### 3.1 Pseudo-replication (this one nearly produced a false discovery)

There are 9,971 snapshots across 336 fixtures — a mean of **29.7 per fixture**. Every snapshot of
one fixture shares the same model opinion, the same closing price, the same market. They are not
29.7 independent observations.

The first implementation of this work used a raw Wilson interval at both levels. The same data
then read:

| unit | toward Wowza | 95% CI | p |
|---|---|---|---|
| fixture (n=181) | 54.7% | [46.4%, 62.8%] | 0.267 |
| snapshot (n=5,919) — **wrong** | 52.6% | [51.3%, 53.9%] | **0.0001** |

The second is not a stronger finding on more data. It is the same 181 fixtures counted ~30 times
each, with the interval shrunk by the square root of the replication. Corrected with a
**cluster bootstrap that resamples fixtures rather than rows**, the snapshot-level result is
52.6% [45.0%, 60.8%], p = 0.51 — in agreement with the fixture-level result, as it must be.

Both units are reported and never blurred. `unit=` is a required argument on the summary
functions so a caller must state which one they mean. A regression test builds 12 fixtures × 20
identical snapshots and asserts the directional CI does **not** narrow.

### 3.2 Mean reversion (the placebo)

The model's probabilities sit systematically more centrally than the market's. So "the price
moved toward the model" and "the price moved toward the middle" can be the same sentence, and
ordinary mean reversion in a noisy opening price would reproduce the headline with no skill at
all.

The control replaces `p_model` with a **fixed, information-free anchor** — the median opening
market probability.

**THE PLACEBO NOW BEATS THE MODEL, AND THIS IS THE HEADLINE RESULT.**

| sample | model toward | placebo toward | model's excess |
|---|---|---|---|
| 181 fixtures (2026-08-23, first run) | 54.7% | 46.7% | **+8.0pp** |
| **281 fixtures (2026-08-23, current)** | **55.7%** | **58.4%** | **−2.7pp** |

On 100 additional fixtures the excess inverted. A constant carrying no information whatsoever
predicts the market's direction slightly *better* than Wowza's residual does. The +8.0pp that
looked like the one encouraging result was noise, and it did not survive a single week of extra
data.

This is exactly the failure mode the control exists to catch, and it is the reason the control
matters more than the headline. Wowza's probabilities sit systematically more centrally than the
market's, so "the price moved toward Wowza" and "the price moved toward the middle" are close to
the same sentence — and the second needs no model at all. The mean-reversion explanation is now
the *better-supported* one.

Note what this does to the magnitude result below. Mean signed movement is +0.521pp with a CI
that **excludes zero** — statistically real. But a real movement toward the centre is not
evidence of foresight, and attributing it to Wowza when a constant does better on direction would
be the whole research programme's central error.

## 4. Data quality rules

Flags are applied **only where the data supports the judgement** (brief section 19). A flag we
cannot evaluate is recorded in `quality_not_assessed`, never silently treated as passing —
"we checked and it was fine" and "we could not check" are different statements.

| flag | condition |
|---|---|
| `MISSING_KICKOFF` | `minutes_to_kickoff` is null |
| `POST_KICKOFF_ENTRY` | `minutes_to_kickoff <= 0` |
| `POST_KICKOFF_CLOSE` | close row is at or after kickoff |
| `MISSING_CLOSE` | no pre-kickoff close for the fixture |
| `MISSING_OPPOSITE_SIDE` | either `odds_over25` or `odds_under25` absent |
| `INVALID_MARKET_MAPPING` | two-way overround outside 0.98–1.25 |
| `INSUFFICIENT_BOOKS` | `n_books < 3` **where known** |

Declared **not assessable** with the current snapshot schema, because inventing a proxy would
misclassify healthy data: `STALE_ENTRY_PRICE`, `STALE_CLOSE_PRICE`, `SYNTHETIC_ODDS`. These need
per-book quote timestamps and a provenance marker the snapshots do not carry.

Additional hard rules:

- The close is the last **pre-kickoff** snapshot, via `closing_snapshot()`. Never the last row —
  `predictions.csv` is filtered by date, so a fixture that kicked off earlier the same day can
  still be captured.
- A close at or before the entry is dropped. A zero-length window cannot measure movement, and
  comparing the close row with itself would contribute a guaranteed movement of exactly 0.
- **No close is ever fabricated.** Missing close → the row does not survive.
- A flat market (|move| < 0.2pp) is a **third state**: `toward_wowza` is NaN, not 0. Counting a
  flat market as a miss would be scoring absence of evidence as evidence against.

## 5. Current sample

| | |
|---|---|
| snapshot observations built | 25,052 over 282 fixtures |
| eligible (`clv_quality == OK`) | 24,182 (96.5%) over 281 fixtures |
| eligible **and** price moved | 19,601 snapshots / **219 fixtures** |
| excluded `INSUFFICIENT_BOOKS` / `MISSING_KICKOFF` | 687 / 183 |
| **eligible entry span** | **2026-08-17 → 08-23 — 7 days, still one ISO week** |

## 6. Results

### Headline

| | fixture-level (primary) | snapshot-level (clustered) |
|---|---|---|
| n moved | 219 | 19,601 (281 fixtures) |
| toward Wowza | 55.7% | 54.9% |
| 95% CI | [49.1%, 62.1%] | [48.7%, 61.2%] |
| p vs 50% | 0.091 | 0.122 |
| mean signed move | **+0.521pp** | +0.455pp |
| 95% CI | **[+0.134, +0.908]** | [+0.106, +0.822] |
| median signed move | +0.460pp | +0.320pp |
| mean move when correct | +2.451pp | +2.191pp |
| mean move when wrong | −1.906pp | −1.660pp |
| mean \|move\| | 2.209pp | 1.952pp |
| P(≥0.5pp) / P(≥1pp) / P(≥2pp) | 86.3% / 63.9% / 40.2% | 81.2% / 62.8% / 36.3% |
| executable CLV mean | +0.101% | **−0.024%** |
| executable CLV median | +0.000% | +0.000% |
| positive CLV | 42.5% | 34.3% |
| **placebo (fixed anchor)** | **58.4% → excess −2.7pp** | — |

### By model type

| | fixtures | toward | mean signed | mean CLV | status |
|---|---|---|---|---|---|
| new_format | 167 | 58.3% | +0.534pp | −0.023% | RESEARCH |
| standard | 114 | 52.2% | +0.503pp | +0.272% | EARLY_SIGNAL |

### By residual band (fixture-level)

| band | fix | moved | toward | mean signed | mean CLV |
|---|---|---|---|---|---|
| < −10 | 32 | 23 | 60.9% | +1.032 | −0.374 |
| −10:−6 | 32 | 22 | 50.0% | +0.260 | −0.403 |
| −6:−4 | 29 | 26 | 61.5% | +0.642 | −0.069 |
| −4:−2 | 34 | 26 | 53.8% | +0.785 | +0.530 |
| −2:+2 | 65 | 52 | 53.8% | +0.141 | +0.021 |
| +2:+4 | 34 | 23 | 52.2% | +0.673 | +0.794 |
| +4:+6 | 23 | 19 | 52.6% | +0.535 | +0.690 |
| +6:+10 | 24 | 22 | 63.6% | +0.428 | −0.391 |
| > +10 | 8 | 6 | 50.0% | +0.860 | +0.615 |

The UNDER-side asymmetry noted on the smaller sample has largely washed out, and `+6:+10` — the
band the brief singled out — is 63.6% toward but with **negative** CLV (−0.391%). Directional
rate and price value are not tracking each other, which is itself a reason not to read either in
isolation.

### By time to kickoff (snapshot-level, clustered)

**Read the `per hour` column, not `mean signed`.** The window from entry to the close shrinks as
kickoff approaches, so a late entry has little distance left to travel and small absolute movement
is *mechanical*.

| bucket | fix | moved | toward | mean window | mean signed | **per hour** | mean CLV |
|---|---|---|---|---|---|---|---|
| >24h | 260 | 16,571 | 55.2% | 4,408 min | +0.472 | +0.006 | −0.065 |
| 12–24h | 159 | 1,778 | 58.6% | 1,023 min | +0.631 | +0.037 | +0.571 |
| 6–12h | 125 | 530 | 46.6% | 493 min | +0.015 | +0.002 | −0.143 |
| 3–6h | 120 | 376 | 43.4% | 238 min | −0.201 | −0.051 | −0.784 |
| 1–3h | 142 | 290 | 46.9% | 105 min | +0.064 | +0.036 | −0.239 |
| 30–60m | 103 | 56 | 60.7% | 35 min | +0.484 | **+0.841** | +0.700 |
| 10–30m | 3 | 0 | n/a | n/a | n/a | n/a | n/a |

My earlier claim that near-kickoff **coverage** was the binding constraint was wrong. Coverage is
good: of 142 kicked-off fixtures, 96.5% have a 1–3h snapshot, 81.7% have 30–60m, and the median
last pre-kickoff snapshot lands **19 minutes** before kickoff. What is thin is the number of
*moved* observations in late buckets, and that is the shrinking window, not missing data.

The 30–60m bucket has the highest per-hour rate by a wide margin (+0.841pp/h against +0.006 at
>24h) — the market's information arrival accelerates sharply toward kickoff, which is what one
would expect. It rests on 56 moved observations from 103 fixtures.

### By league

All 19 leagues are `INSUFFICIENT_SAMPLE`; the largest is USA MLS at 43 fixtures (46.7% toward,
CLV −1.424%).

## 7. Answers to the five questions (brief section 29)

**A. Is the directional effect statistically interesting?**
**No.** 55.7% [49.1%, 62.1%], p = 0.091 — the interval includes 50%. More decisively, the
**placebo now beats it**: an information-free constant scores 58.4%. There is no evidence here
that the model's residual carries directional information the market's own centrality does not.

**B. Is the movement magnitude economically interesting?**
**Statistically real, economically marginal, and not attributable to Wowza.** +0.521pp with a CI
of [+0.134, +0.908] that excludes zero — a genuine change from the earlier 181-fixture read. But
given A, the most parsimonious explanation is mean reversion toward the centre, which requires no
model. At ~2.15 average odds, +0.52pp of fair probability is worth roughly 0.25% in price terms
even if it were real and capturable.

**C. Does it produce positive clean CLV?**
**No.** Median **+0.000%** at both units; mean +0.101% per fixture but **−0.024%** across
snapshots; only 42.5% / 34.3% positive. The strongest residual band on direction (+6:+10, 63.6%)
has *negative* CLV. This is the measurement that matters most and it is at or below zero.

**D. Is it chronologically stable?**
**Still cannot be assessed.** The span grew from 3 to 7 days but remains a single ISO week.
Monthly and weekly rows are emitted as `NOT_ASSESSABLE`. The halves split (53.2% / 60.7%) is a
split within one week. Note what *did* change across those extra four days: the placebo verdict
inverted completely. That is the clearest possible demonstration that nothing here is stable yet.

**E. Is the sample large enough for deployment?**
**No.** 219 moved fixture-level observations against a 500–1,000 bar. All 19 leagues
insufficient.

## 8. Limitations

1. **Seven days, one ISO week.** The binding limitation. The placebo verdict inverting across
   four extra days is the proof that nothing here has settled.
2. **Time-to-kickoff is readable but thin at the near end.** Not a coverage problem — corrected
   from an earlier wrong claim. 96.5% of kicked-off fixtures have a 1–3h snapshot and the median
   last snapshot is 19 min before kickoff. The late buckets are thin in *moved* observations
   because the window to the close shrinks, which the `per hour` column now normalises.
3. **Result/ROI grading — FIXED, pending its first live run.** `v11_grade.py` previously took
   results from v9's public `bets_ledger.csv`, which holds only fixtures **v9 actually tipped**:
   94/336 settled, with League Two 0/24, Serie B 0/11, Ireland 0/5. That subset is conditioned on
   v9 having disagreed with the market enough to fire a tip — the variable under study — so every
   ROI figure drawn from it was selected on the independent variable. Movement and CLV were never
   affected (prices are captured for the whole board).

   `src/results.py` now fetches actual goals from football-data.co.uk for all 31 mapped leagues
   and grading prefers them, keeping the ledger only as a fallback for the API-Football-only
   competitions football-data does not carry. Every run prints the provenance split so the
   remaining ledger share stays visible.

   **Not yet validated live.** This machine sits behind TLS inspection and cannot reach
   `football-data.co.uk` at all (connection refused, not a certificate problem), so the fetch is
   exercised on the first CI run. The parsing, club-name resolution and grading logic is
   unit-tested against synthetic frames built to the real schema, and a failed fetch degrades to
   "nothing newly graded" with every unavailable league named — never to wrong results.
4. **Dispersion analysis is not possible yet.** `n_books` is present on 8.3% of snapshots and
   `book_dispersion` on 7.7%; `market_prob_range` is not stored at all and is left NULL rather
   than approximated from the std, which would fabricate a measured-looking number. No
   `v11_movement_by_dispersion.csv` is produced — an optional output on 7.7% coverage would be
   noise wearing a filename. Brief section 10 stays open.
5. **`market_move` uses consensus, not best executable price.** Correct for measuring where the
   market went; it is not what a bettor would receive.
6. **One market only.** OVER/UNDER 2.5. Nothing here extends to BTTS or side markets.

## 9. Graduation criteria (brief section 23)

Not met. Nothing should be deployed, tipped, or staked on this until **all** hold:

- [ ] ≥ 500 clean movement observations at **fixture level** (preferably 1,000+) — currently 219
- [x] positive mean signed movement with a CI excluding zero — **+0.521pp [+0.134, +0.908]**, the
      one criterion currently met, though see A above on what it is attributable to
- [ ] positive clean executable CLV — currently +0.101% mean / **+0.000% median**, −0.024% across
      snapshots
- [ ] stable across chronological periods — unassessable (7 days, one ISO week)
- [ ] not dependent on one small league — all 19 leagues insufficient
- [ ] survives reasonable segmentation
- [ ] enough executable-price observations
- [ ] a market-relative effect that remains economically meaningful
- [ ] **beats the placebo anchor** by a margin that is itself significant — currently **FAILING
      OUTRIGHT**: the placebo scores 58.4% against the model's 55.7%, an excess of −2.7pp

`toward_wowza > 50%` is explicitly **not** sufficient. Actual betting deployment would require
stronger evidence than the above.

## 10. Future ML design — DOCUMENTATION ONLY

**Do not train these.** 137 fixture-level observations cannot support them, and fitting them now
would produce a model that memorises three days of August.

### Model A — P(market moves toward Wowza)

Classification. Baseline **logistic regression**, and only that, until the sample is in the
thousands; gradient boosting is not justified by 137 observations and would overfit
spectacularly.

Candidate features: `residual`, `abs_residual`, `p_model`, `p_market_entry`, `entry_odds`, odds
band, `model_type`, `league`, `hours_to_kickoff`, `market_prob_std`, `market_prob_range`,
`n_books`, `previous_market_move_pp`, `ece_used`, `n_eff_used`.

Validation must be **chronological**, never random — a random split on 30 correlated snapshots
per fixture would put the same fixture in train and test and report near-perfect accuracy.
Fixture-level grouping is mandatory in any cross-validation.

### Model B — expected signed movement, or expected CLV

Regression on the same features. Preferred to Model A if it can be fit honestly, because
magnitude is what matters economically and a classifier optimising direction would happily learn
to win 0.2pp and lose 2.0pp.

### Prerequisites before either is attempted

1. ≥ 500 fixture-level observations spanning ≥ 8 weeks.
2. Near-kickoff coverage sufficient to populate the time buckets.
3. Unbiased result grading (limitation 3).
4. The placebo margin still holding.

## 11. Future decision engine — DOCUMENTATION ONLY

If, and only if, the phenomenon survives a large prospective sample:

```
V9 football probability
   -> market fair probability
   -> residual
   -> movement model            (does not exist; section 10)
   -> expected CLV
   -> best executable price     (src/best_price.py in Pro)
   -> conservative EV
   -> betting decision
```

This remains future architecture. It is written down so the shape is agreed, not because any
part of it is authorised.

## 12. Files

| file | role |
|---|---|
| `src/movement.py` | arithmetic, quality flags, clustered inference. Pure functions. |
| `scripts/v11_market_movement.py` | driver; produces the detail dataset and five summaries |
| `src/results.py` | match results from football-data.co.uk; league-scoped club resolution |
| `scripts/v11_tests.py` | 141 checks: 68 movement, 30 results, 43 pre-existing |
| `output/v11_market_movement_detail.csv` | one row per eligible snapshot — the durable record |
| `output/v11_movement_summary.csv` | headline, placebo, chronological stability |
| `output/v11_movement_by_residual.csv` | signed and absolute residual buckets |
| `output/v11_movement_by_model.csv` | standard vs new_format |
| `output/v11_movement_by_time.csv` | time-to-kickoff buckets |
| `output/v11_movement_by_league.csv` | per league, with sample-discipline status |

## 13. What was explicitly NOT done

Per the brief's non-goals, and stated so a later reader does not assume an omission was an
oversight:

- The football models were **not** retrained, altered, or given market-movement targets. Their
  independence is the research instrument; training Wowza to mimic future market movement would
  destroy the signal being studied.
- **No** ML model was trained. No XGBoost, no LightGBM, no logistic regression.
- **No** thresholds were tuned, and no threshold anywhere was changed.
- **No** Telegram, no staking, no SNIPER tier, no live deployment, no V12 repository.
- **No** data was deleted. Rows failing quality checks are flagged and retained.
