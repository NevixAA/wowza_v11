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
market probability. Current result: the market moves toward that anchor **46.7%** of the time
versus 54.7% toward the model, an excess of **+8.0pp**. So the headline is not explained by mean
reversion. This control is the single most important number in the output and must be kept.

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
| snapshot observations built | 9,491 over 181 fixtures |
| eligible (`clv_quality == OK`) | 9,265 over 181 fixtures |
| eligible where the price actually moved | 5,919 snapshots / 137 fixtures |
| excluded: `MISSING_KICKOFF` | 182 |
| excluded: `INSUFFICIENT_BOOKS` | 44 |
| **eligible entry span** | **2026-08-17 → 2026-08-19 — 3 distinct days, one ISO week** |

That last row is the most important limitation and is covered in section 8.

## 6. Results

### Headline

| | fixture-level (primary) | snapshot-level (clustered) |
|---|---|---|
| n moved | 137 | 5,919 (181 fixtures) |
| toward Wowza | 54.7% | 52.6% |
| 95% CI | [46.4%, 62.8%] | [45.0%, 60.8%] |
| p vs 50% | 0.267 | 0.515 |
| mean signed move | +0.220pp | +0.204pp |
| 95% CI | [−0.119, +0.559] | [−0.093, +0.522] |
| median signed move | +0.240pp | +0.220pp |
| mean move when correct | +1.535pp | +1.500pp |
| mean move when wrong | −1.370pp | −1.236pp |
| mean \|move\| | 1.460pp | 1.375pp |
| P(\|move\| ≥ 1pp) | 48.9% | 48.5% |
| executable CLV mean | +0.185% | +0.086% |
| executable CLV median | +0.000% | +0.000% |
| positive CLV | 35.8% | 24.2% |
| **placebo (fixed anchor)** | **46.7%** | — |

Both confidence intervals include 50%. Both signed-movement intervals include zero.

### By residual band (fixture-level)

Every band is `INSUFFICIENT_SAMPLE` (< 50 fixtures). The largest is 37.

| band | fixtures | moved | toward | mean signed | mean CLV |
|---|---|---|---|---|---|
| < −10 | 23 | 19 | 63.2% | +0.680 | +0.111 |
| −10:−6 | 18 | 14 | 57.1% | +0.276 | −0.500 |
| −6:−4 | 21 | 19 | 63.2% | +0.615 | +0.257 |
| −4:−2 | 24 | 15 | 73.3% | +0.582 | +0.630 |
| −2:+2 | 37 | 31 | 45.2% | −0.256 | −0.299 |
| +2:+4 | 18 | 13 | 30.8% | −0.589 | −0.159 |
| +4:+6 | 19 | 12 | 50.0% | −0.166 | +0.320 |
| +6:+10 | 15 | 11 | 63.6% | +0.512 | +1.276 |
| > +10 | 6 | 3 | 33.3% | +1.637 | +3.120 |

Note the asymmetry: the **negative** residual bands (model prefers UNDER) look consistently
better than the positive ones. On these sample sizes that is an observation, not a finding — but
it is the kind of asymmetry worth watching, because it would be invisible in the `abs_residual`
view.

### By |residual|

| band | fixtures | toward | mean signed | mean CLV |
|---|---|---|---|---|
| 0–2 | 37 | 45.2% | −0.256 | −0.299 |
| 2–4 | 42 | 53.6% | +0.039 | +0.264 |
| 4–6 | 40 | 58.1% | +0.313 | +0.281 |
| 6–8 | 15 | 61.5% | +0.129 | +0.534 |
| 8–10 | 18 | 58.3% | +0.652 | +0.008 |
| 10+ | 29 | 59.1% | +0.811 | +0.521 |

Monotone-ish in the direction "bigger disagreement, more movement toward us". The brief warns
against assuming larger residual = better signal, and the counter-evidence is in the signed table
above: `> +10` is 33.3% toward on 3 moved observations. Extreme residuals may be identifying
model errors as often as market lag; nothing here separates those.

### By model type

| | fixtures | toward | mean signed | mean CLV | status |
|---|---|---|---|---|---|
| new_format | 119 | 56.5% | +0.306 | +0.165 | EARLY_SIGNAL |
| standard | 62 | 51.1% | +0.046 | +0.227 | EARLY_SIGNAL |

new_format looks better on direction and movement; standard looks marginally better on
executable CLV. Both intervals are wide and overlapping. The brief's caution applies directly:
prediction quality and market-timing quality are not the same thing, and these two tracks may
genuinely differ.

### By time to kickoff (snapshot-level, clustered)

| bucket | fixtures | moved | toward | mean signed | mean CLV |
|---|---|---|---|---|---|
| > 24h | 167 | 5,710 | 53.1% | +0.218 | +0.118 |
| 12–24h | 18 | 101 | 32.7% | −0.370 | −1.437 |
| 6–12h | 14 | 39 | 56.4% | +0.027 | +0.165 |
| 3–6h | 10 | 25 | 40.0% | −0.192 | −0.524 |
| 1–3h | 13 | 38 | 34.2% | −0.069 | −0.736 |
| 30–60m | 10 | 6 | 83.3% | +1.007 | +2.343 |
| 10–30m | 1 | 0 | n/a | n/a | n/a |

**This table is dominated by one bucket and cannot presently answer the entry-timing question.**
96% of moved snapshots are `> 24h`, because the shadow samples twice hourly for days before
kickoff while the near-kickoff buckets have 6–101 observations from 10–18 fixtures. The `30–60m`
row at 83.3% rests on **6 observations** and should not be read as anything. Nearer-kickoff
coverage is the binding constraint on this analysis, and it is the same gap the Pro-side snapshot
coverage monitor found for side markets.

### By league

All 16 leagues are `INSUFFICIENT_SAMPLE`; the largest is USA MLS at 29 fixtures. Two rows are
worth naming only as cautionary examples of what small samples produce: Sweden Allsvenskan 87.5%
toward on 8 moved, and Denmark Superliga **0.0%** toward on 6 moved with CLV −4.53%. Neither is
information about those leagues.

## 7. Answers to the five questions (brief section 29)

**A. Is the directional effect statistically interesting?**
**Marginally, and not significantly.** 54.7% [46.4%, 62.8%], p = 0.267. The CI includes 50%. The
one genuinely encouraging element is the placebo: +8.0pp over an information-free anchor, so
whatever is there is not merely mean reversion. Verdict: *interesting enough to keep collecting,
nowhere near established.*

**B. Is the movement magnitude economically interesting?**
**No, not yet.** Mean signed movement is +0.220pp with a CI of [−0.119, +0.559] that includes
zero. To put the scale in context: +0.22pp of fair probability at ~2.17 average odds is worth
roughly 0.1% in price terms. Wins and losses are near-symmetric (+1.535pp when correct, −1.370pp
when wrong), so the economics rest entirely on the thin directional edge rather than on winning
big and losing small. This is precisely the case the brief's section 4 warns about.

**C. Does it produce positive clean CLV?**
**Not meaningfully.** Executable CLV is +0.185% mean, **+0.000% median**, and only **35.8%** of
observations are positive. A mean pulled above zero by a minority of large moves while the median
is exactly zero and two-thirds are non-positive is not a monetisable price edge. Note also that
signed movement and fair-probability CLV are the same measurement (section 2), so this is the
only CLV number that adds information — and it is ~0.

**D. Is it chronologically stable?**
**Cannot be assessed.** All eligible entries fall in a single ISO week — three days, 2026-08-17
to 08-19. The monthly and weekly rows are emitted as `NOT_ASSESSABLE` rather than as duplicates
of the overall row. The halves split (48.5% first, 56.7% second) is a split *within* those three
days and is as likely to reflect fixture mix as drift. **The entire headline currently rests on
three days of entries.**

**E. Is the sample large enough for deployment?**
**No.** 137 moved fixture-level observations against a graduation bar of 500–1,000 clean ones.
Every residual band and every league is `INSUFFICIENT_SAMPLE`. Three days of data. Not close.

## 8. Limitations

1. **Three days.** The binding limitation. Everything else is secondary to it.
2. **Time-to-kickoff is unusable.** 96% of moved snapshots sit in `> 24h`; the buckets that would
   answer "when should we enter" hold 6–101 observations.
3. **Result/ROI coverage is biased.** v11 grades from v9's public `bets_ledger.csv`, which
   contains only fixtures **v9 actually tipped** — 94 of 336 (28%). The graded subset is
   therefore conditioned on v9 having disagreed with the market enough to fire a tip, which is
   the variable under study. **Movement and CLV are unaffected** (they need only prices, which
   are unconditioned), but any result-based or ROI figure is drawn from a conditioned subset and
   is marked as such. Fixing this requires grading from actual results (football-data.co.uk)
   rather than from v9's tip ledger.
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

- [ ] ≥ 500 clean movement observations at **fixture level** (preferably 1,000+) — currently 137
- [ ] positive mean signed movement with a CI excluding zero — currently [−0.119, +0.559]
- [ ] positive clean executable CLV — currently +0.185% mean / +0.000% median
- [ ] stable across chronological periods — currently unassessable (3 days)
- [ ] not dependent on one small league — currently all 16 leagues insufficient
- [ ] survives reasonable segmentation
- [ ] enough executable-price observations
- [ ] a market-relative effect that remains economically meaningful
- [ ] **still beats the placebo anchor** by a margin that is itself significant

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
| `scripts/v11_tests.py` | 111 checks, 68 of them movement-specific |
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
