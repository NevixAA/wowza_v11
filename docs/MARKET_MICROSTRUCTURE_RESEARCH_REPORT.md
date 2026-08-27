# Market Microstructure Research Report

**Date:** 2026-08-27 · **Repo:** `NevixAA/wowza_v11` · **Status:** research only, nothing deployed

## Bottom line

**No new model. The prospective movement result does not survive its own control.**

The headline was that when Wowza disagrees with the market, the market later moves toward Wowza
~58% of the time (N≈369, p≈0.003). That is reproduced here: 58.00% at N=300.

The control kills it. Ask a deliberately stupider question — did the market move toward a **fixed
anchor**, the median market probability, with no model input whatsoever? Answer: **58.33%**. The
model-free baseline matches the model, very slightly beating it.

| At the earliest entry (N=300) | Rate |
|---|---|
| Market moved toward Wowza | 58.00% (95% CI 52.35–63.45) |
| Market moved toward a fixed anchor (**no model**) | **58.33%** |
| Edge over placebo | **−0.33pp** |

The mechanism is measured, not guessed:

- Wowza's probabilities have **0.564×** the standard deviation of the market's (model sd 0.0520,
  market sd 0.0922). The model is systematically more central.
- **corr(residual, distance-to-anchor) = +0.833**, with **83.0%** sign agreement.

So "the market moved toward Wowza" and "the market moved toward the middle" are largely the same
statement on this sample. Ordinary drift in a noisy opening price reproduces the headline with no
skill involved.

This is the conclusion section 31 said to be willing to reach.

---

## 1. Files changed

**New in `wowza_v11`:**

| File | Role |
|---|---|
| `src/microstructure.py` | Backward-looking price context per snapshot (velocity, momentum, changes, reversals, dispersion pass-through, quality flags) |
| `scripts/v11_microstructure.py` | Research pipeline → one durable dataset + segment tables + the residual-information diagnostic |
| `docs/MARKET_MICROSTRUCTURE_RESEARCH_REPORT.md` | This report |

**New outputs:**

| File | Rows |
|---|---|
| `output/v11_market_microstructure.csv` | 33,094 × 67 — the single durable detail dataset (section 18) |
| `output/v11_microstructure_coverage.csv` | 23 features with coverage/missingness |
| `output/v11_microstructure_segments.csv` | 19 segments, each with its own placebo |

**Not changed:** V9 (frozen, and section 1 requires its independence), V9-Pro, any existing V11
model, threshold, or workflow. No model was trained.

## 2. Existing functionality reused, not rebuilt

Section 26 asks that velocity, dispersion and CLV not acquire three definitions. Reused as-is:

- `src/movement.py` — `eligible`, `fixture_level`, `wilson`, `cluster_bootstrap_mean`,
  `sample_status`, `placebo_toward_rate`, `MIN_MOVE_PP`
- `src/book_consensus.py` — `n_books` and `book_dispersion` are **passed through**, never recomputed
- `v11_market_movement_detail.csv` — forward outcomes (`signed_market_move_pp`, `clv_pct`,
  `toward_wowza`) joined by `snapshot_id`

One rename for honesty: `_attach_momentum` defines `previous_market_move_pp` as
*current − first price*, which is **move-from-open**, not the brief's `last_move_pp` (move since
the previous snapshot). Both now exist under distinct names; conflating them would answer
section 20 with the wrong variable.

## 3–5. New features, coverage, missingness

All strictly backward-looking from each snapshot's own timestamp.

| Feature | Coverage | Missing | Notes |
|---|---|---|---|
| `move_from_open_pp` | 98.5% | 1.5% | NaN on a fixture's first snapshot, never 0.0 |
| `last_move_pp` | 98.5% | 1.5% | move since previous snapshot |
| `n_price_changes` | 100% | 0% | over observed polls only |
| `reversal_count` / `direction_changes` | 100% | 0% | |
| `time_since_last_change_min` | 70.0% | 30.0% | null until a change has been observed |
| `velocity_1h` | 69.1% | 30.9% | pp per hour |
| `velocity_3h` | **84.4%** | 15.6% | best-covered window |
| `velocity_6h` | 73.4% | 26.6% | |
| `market_acceleration` | 63.0% | 37.0% | `velocity_1h − velocity_3h` |
| `largest_move_{1h,3h,6h}_pp`, `range_*` | 69–84% | | |
| `n_books` | 73.5% | 26.5% | passed through |
| `market_prob_std` | 70.7% | 29.3% | passed through |
| **`velocity_30m`** | **0%** | **100%** | **not computable — see below** |

**`velocity_30m` is deliberately empty.** Median inter-snapshot gap is **33.6 minutes** (p25 23.9,
p75 44.3, p95 64.5) — longer than the window. It would be null more often than not and, worse,
non-null precisely for the unrepresentatively densely-sampled fixtures. Present as an explicit
all-NaN column so it reads as *considered and unavailable* rather than forgotten.

**`market_prob_range`** is 0% populated in the existing detail file and remains unavailable; only
`std` is carried upstream.

Two rules that make the rest trustworthy:

- **A window needs a real observation at least halfway back**, else NULL. Without it a "6h
  velocity" from two points 20 minutes apart is extrapolated 18× and dominates any distribution
  it enters. Verified: 0 rows carry `velocity_3h` with a span under 90 minutes.
- **Only 12.9% of consecutive polls show any change** (median move when one happens: 0.31pp). So
  "no change in the last hour" is the normal state and is distinguished from "we did not look";
  `n_polls_seen` travels alongside the change counts.

## 6. Timestamp / leakage safety

Adversarial test: corrupt the **last 30%** of every fixture's prices with random values and
recompute. A backward-looking feature must be bit-identical on the earlier rows.

| Feature | Identical on early rows |
|---|---|
| `velocity_1h` / `3h` / `6h` | 100.00% |
| `market_acceleration` | 100.00% |
| `move_from_open_pp`, `last_move_pp` | 100.00% |
| `n_price_changes`, `reversal_count` | 100.00% |
| `largest_move_3h_pp` | 100.00% |

**PASS — no forward leakage.** Also verified: velocity is time-normalised (pp/hour, not raw
difference), and a fixture's first snapshot yields NaN rather than 0.0 (521 rows, 0 zeros).

## 7–10. Current position

| Metric | Value |
|---|---|
| Fixture-level N (earliest eligible) | **300** |
| Toward-Wowza rate | 58.00% (CI 52.35–63.45) |
| **Placebo (no model)** | **58.33%** |
| Mean signed movement | +0.62pp |
| Mean clean CLV | +0.533% |
| **CLV cluster-bootstrap 95% CI** | **[−0.168, +1.230] — includes zero** |
| Median CLV | +0.000% |
| Positive CLV | 47.7% |

The quoted +0.65% mean CLV is **not significantly positive** once the interval is clustered by
fixture. Median CLV is exactly zero.

## 11. Controlling for previous market momentum (section 20)

**Segmented by prior move** (earliest entry):

| Segment | N | Toward | Placebo | Edge |
|---|---|---|---|---|
| `MARKET_FLAT` | 151 | 55.63% | 55.63% | **0.00pp** |
| `UNKNOWN` (no prior observation) | 131 | 61.07% | 54.96% | +6.11pp |

**Logistic diagnostic** — does the residual say anything once mean reversion is in the equation?

| Specification | N | Term | β | z | Sig. 95% |
|---|---|---|---|---|---|
| `residual` | 300 | residual | +0.2025 | 1.71 | No |
| `reversion` | 300 | reversion | +0.1240 | 1.05 | No |
| `residual + reversion` | 300 | residual | +0.3243 | 1.51 | No |
| | | reversion | −0.1459 | −0.68 | No |
| `+ window + dispersion` (complete cases) | 137 | residual | +0.4781 | 1.37 | No |
| | | `log_window_h` | +0.3714 | **1.88** | No |

**No coefficient is significant in any specification.** In the simple fit only the *intercept*
reaches significance (β=+0.3259, z=2.77) — a constant tilt toward Wowza that does not scale with
how much Wowza disagrees. That is the signature of a structural artifact, not information: if the
residual carried signal, bigger disagreement would predict bigger movement, and it does not
(Q1: **no**).

The strongest term is `log_window_h` — a longer observation window mechanically gives the price
more room to cross toward Wowza. Also mechanical, not informational.

**A structural limit worth stating plainly:** at the earliest snapshot — the only entry where the
effect appears — `velocity_3h` exists for **2 of 300** fixtures, because the earliest snapshot
has no three hours of history behind it. Momentum *cannot* be fully controlled at the exact point
where the effect lives. The `MARKET_FLAT` row above is the best available substitute, and it
shows an edge of exactly 0.00pp.

## 12. By dispersion (section 5, Q2)

Observed terciles, not invented cutoffs:

| Segment | N | Toward | Placebo | Edge | Status |
|---|---|---|---|---|---|
| tight | 46 | 67.39% | 56.52% | +10.87pp | INSUFFICIENT_SAMPLE |
| mid | 45 | 60.00% | 55.56% | +4.44pp | INSUFFICIENT_SAMPLE |
| dispersed | 46 | 58.70% | 54.35% | +4.35pp | INSUFFICIENT_SAMPLE |

The tight-consensus cell is the largest apparent edge in the whole study and it is **n=46**. The
ordering (tight > mid > dispersed) is the kind of monotone pattern that looks like a finding and
is routinely noise at this sample size. **Hypothesis for the confirmation period only** — under
section 22 it must not be selected on and then reported on the same data. Note also dispersion is
only **46% covered** at this entry point.

## 13. By entry time (section 8, Q4)

| Entry | N | Toward | 95% CI | Placebo | Edge | Mean CLV |
|---|---|---|---|---|---|---|
| earliest (~71h) | 300 | 58.00% | 52.4–63.5 | 58.33% | −0.33pp | +0.53% |
| ~24h | 175 | 56.57% | 49.2–63.7 | 53.14% | +3.43pp | −0.18% |
| ~12h | 163 | 50.92% | 43.3–58.5 | 45.40% | +5.52pp | −0.37% |
| ~6h | 151 | 49.67% | 41.8–57.6 | 49.01% | +0.66pp | −0.44% |
| ~3h | 138 | 54.35% | 46.0–62.4 | 53.62% | +0.72pp | −0.39% |
| ~1h | 101 | 54.46% | 44.8–63.8 | 55.45% | −0.99pp | +0.06% |

Every interval spans 50%. The rate **decays to a coin flip by 6–12h** and mean CLV is **negative
at every entry point except the extremes**. The effect lives where the market has barely formed
(~3 days out) and disappears as the market matures — the opposite of what a genuine
price-discovery edge should look like.

## 14. By model type (Q5)

| Model | N | Toward | Placebo | Edge | Mean CLV |
|---|---|---|---|---|---|
| new_format | 178 | 57.87% | 55.06% | +2.81pp | +0.13% |
| standard | 122 | 58.20% | 59.84% | **−1.64pp** | +1.13% |

Neither survives. For `standard`, the placebo **beats** the model. The signal does not survive
separately for either track.

## 15. Bookmaker lead/lag (section 7)

**Not attempted — the data does not support it.** Lead/lag needs per-bookmaker timestamped price
histories. `v11_shadow_snapshots.csv` stores only the *aggregated* consensus per snapshot
(`v11_p_market`, `n_books`, `book_dispersion`); the per-book prices are consumed inside
`book_consensus.build_index` and never persisted per book.

Under section 7's own thresholds every bookmaker would be `INSUFFICIENT_SAMPLE`, so no ranking is
produced. Persisting per-book quotes is listed in §18 as the highest-value collection change.

Also: `price_source` is `exchange` on 73.5% of rows, so a "sharp vs consensus" split has no
documented basis here yet, and section 6 forbids hardcoding one without justification.

## 16. Placebo results (section 21)

Deterministic, seed **20260827**.

| Placebo | Earliest (N=300) | Interpretation |
|---|---|---|
| Fixed anchor (no model) | **58.33%** | matches/beats Wowza's 58.00% |
| Residual shuffled across fixtures | 47.33% | behaves like chance ✓ |
| Residual sign flipped | 42.00% | ≈ 100 − 58 ✓ |

The shuffle and flip behave correctly, which confirms the harness measures what it claims. The
anchor placebo is the one that matters, and Wowza does not beat it.

## 17. Unresolved data limitations

1. **The sample spans 7 DAYS.** 2026-08-17 → 2026-08-24: **2 ISO weeks, 1 calendar month**,
   19 leagues. Section 17 requires multiple weeks, multiple months and different market regimes.
   **Even at N=1000, a single-month sample would not meet the graduation criteria.** Calendar
   spread is now the binding constraint, not row count.
2. **Chronological stability is untestable.** Split-half gives −2.56pp then +4.86pp — but each
   "half" is 3.5 days. This is not evidence either way.
3. **Only 2 of 300 fixture-level rows (0.7%) carry no quality flag.** `SPARSE_HISTORY` 99.3% and
   `MISSING_PREVIOUS_SNAPSHOT` 43.7% are structural at the earliest snapshot; **`INSUFFICIENT_BOOKS`
   at 54.3%** is a real limitation.
4. **Momentum cannot be controlled at the earliest entry** (`velocity_3h` = 2/300). See §11.
5. **Q6/Q7 are definitional, not empirical.** corr(signed move, fair-probability move) = **+1.0000
   exactly**; corr(signed move, `clv_pct`) = +0.858. CLV is computed *from* the same price move, so
   "movement predicts CLV" cannot be evidence for anything. Treat them as one quantity.
6. **`market_prob_range` unavailable** (0% populated); `velocity_30m` not computable.
7. **No per-book history** (§15).
8. **Duplicate column** `p_market_entry` appears twice in `v11_market_movement_detail.csv`. Worked
   around read-only here; worth fixing in the detail builder.

## 18. Recommended for continued collection

Priority order, judged by what would actually change a conclusion:

1. **Per-bookmaker timestamped quotes.** Unlocks lead/lag (§7), sharp-vs-consensus (§6) and real
   dispersion coverage. The single highest-value change, and it must be collected *forward* —
   there is no historical purchase available.
2. **Calendar spread — just keep collecting.** The binding constraint is 1 month, not N=300.
   Nothing needs building.
3. **Book count / dispersion coverage** (currently 73%/71%): a dispersion result cannot be
   evaluated while 54% of the headline sample lacks it.
4. **Denser polling inside 6h of kickoff**, if cheap — would make `velocity_1h` and 30m viable
   where entry timing actually matters.

## 19. Rejected or deferred

| Item | Decision | Why |
|---|---|---|
| `velocity_30m` | **Rejected** | 33.6-min median gap; would be noise |
| `market_prob_range` | Deferred | 0% upstream coverage |
| Bookmaker lead/lag ranking | **Deferred** | no per-book history; all cells INSUFFICIENT_SAMPLE |
| Sharp-vs-consensus tiers | **Deferred** | no documented basis; §6 forbids guessing |
| Models B / C / D | **Rejected for now** | §16–17; N=300 over 1 month, and no signal to model |
| Dispersion entry rule | **Deferred** | n=46; reserved for out-of-time confirmation |
| New football features | Deferred | §30 P3, behind data-quality review |

## 20. Does any evidence justify new ML?

**No.**

Against section 17's graduation threshold:

| Requirement | Status |
|---|---|
| ≥500 clean fixture observations | ✗ 300 |
| Multiple weeks | ~ 2 |
| Multiple months | ✗ 1 |
| Multiple leagues | ✓ 19 |
| Different market regimes | ✗ unassessable in 7 days |

And sample size is the lesser objection. **There is currently no measured signal to model.** The
residual does not beat a model-free anchor, does not scale with disagreement magnitude, reaches
significance in no specification, and its CLV interval includes zero. Fitting Model B/C/D now
would fit the centrality artifact and report it as skill.

**What would change this answer:** the residual beating the anchor placebo by a margin whose CI
excludes zero, in a sample spanning several months, with the effect surviving at an entry time
where the market is actually mature (6–24h). Until then the honest activity is collection, and
per-book quotes are the collection that would teach us most.

---

## What this does *not* say

It does not say Wowza's football model is bad — that is a separate question, measured elsewhere,
and section 1's independence requirement is intact (nothing here touched V9 or fed market data
into it).

It says the *price-discovery* hypothesis is not supported by the current data, and that the
apparent support came from the model being more central than the market rather than from knowing
something the market did not.

The hypothesis is not dead — it is untested at adequate calendar spread, and the instrumentation
to test it properly now exists.
