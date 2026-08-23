# Market Movement Research — Implementation Report

Date: 2026-08-23 · v11 `c7b0b23` · Pro `4709d4a` · v9 `d61fbaf0`

> **RESEARCH ONLY.** No football model was retrained or altered. No threshold changed. No
> Telegram, no staking, no live deployment, no V12. Definitions and caveats:
> `MARKET_MOVEMENT_RESEARCH.md`.

---

## 0. CORRECTION — this report supersedes its own first version

Two claims in the first version of this report were wrong and are corrected below. Both were
found by re-running on a larger snapshot file (25,708 rows vs the 9,971 the first run read) and
by measuring coverage directly rather than inferring it.

**0.1 The placebo verdict has INVERTED. This is now the headline finding.**

| sample | model toward | placebo toward | model's excess |
|---|---|---|---|
| 181 fixtures (first run) | 54.7% | 46.7% | **+8.0pp** |
| **281 fixtures (now)** | **55.7%** | **58.4%** | **−2.7pp** |

I reported the +8.0pp excess as "the one genuinely encouraging element" and "the real good news".
On 100 more fixtures — four extra days — an information-free constant now predicts the market's
direction slightly *better* than Wowza's residual. The encouraging result was noise. Mean
reversion toward the centre is now the better-supported explanation, and since Wowza's
probabilities sit more centrally than the market's, "moved toward Wowza" and "moved toward the
middle" are close to the same sentence.

**0.2 Near-kickoff coverage is NOT the binding constraint.** I named it "the single
highest-value next step". That was wrong. Measured on kicked-off fixtures only:

| bucket | fixtures with a snapshot | | bucket | fixtures |
|---|---|---|---|---|
| 1–3h | 96.5% | | 30–60m | 81.7% |
| 10–30m | 65.5% | | <10m | 16.2% |

89.4% of fixtures have a snapshot within 60 minutes of kickoff; the median last pre-kickoff
snapshot lands **19 minutes** out. What is thin is the count of *moved* observations in late
buckets, and that is **mechanical**: a T-40m entry has only ~34 minutes left before the close, so
there is little distance left to travel.

| bucket | mean window to close | mean \|move\| | **\|move\| per hour** |
|---|---|---|---|
| >24h | 4,408 min | 1.674pp | 0.024 |
| 6–12h | 493 min | 1.023pp | 0.131 |
| 1–3h | 105 min | 0.674pp | 0.405 |
| 30–60m | 35 min | 0.616pp | **1.100** |

Per unit time the market moves ~45× faster near kickoff. Reading the absolute column alone states
the opposite of the truth, so `mean_window_min` and `signed_move_per_hour_pp` are now computed and
the time table prints both with an explicit note.

---

## 1. What already existed (PHASE 0)

`scripts/v11_market_movement.py` (224 lines) already computed:

- residual = `p_model(first) − p_market(first)`, movement = `p_market(close) − p_market(first)`
- sign-agreement rate, residual bands, model-type split, z vs chance, correlation
- close via `closing_snapshot()` — last **pre-kickoff** snapshot, not last row
- flat markets held as a **third state** (NaN, not False)
- **a placebo control for mean reversion** — a fixed information-free anchor

The placebo is the most valuable thing in the pre-existing code and has been preserved and
promoted rather than rewritten. The prior committed run read n=138, toward 55.07%, mean movement
**−0.128pp**.

Absent: signed magnitude in any form, absolute-residual analysis, time-to-kickoff analysis, CLV
linkage, confidence intervals, per-league sample discipline, the durable observation dataset,
quality flags, Pro ingestion, and any treatment of the clustering problem.

## 2. What was added

| file | repo | role |
|---|---|---|
| `src/movement.py` | v11 | arithmetic, quality flags, clustered inference — pure functions (new) |
| `scripts/v11_market_movement.py` | v11 | rewritten driver: detail dataset + 5 summaries |
| `scripts/v11_tests.py` | v11 | +68 movement checks (43 → 111) |
| `MARKET_MOVEMENT_RESEARCH.md` | v11 | hypothesis, definitions, limits, graduation, future ML |
| `src/importers/v11_research.py` | Pro | archives v11 observations with provenance (new) |
| `config/pro_config.py` | Pro | `V11_REPO/RAW_BASE/LOCAL`; canonical table `movement_observations` |
| `src/pipelines/pro_collect.py` | Pro | importer registry merged |
| `src/monitoring/weekly_audit.py` | Pro | new `movement research` audit area |

Outputs: `v11_market_movement_detail.csv` (9,491 rows), `v11_movement_summary.csv`,
`_by_residual.csv`, `_by_model.csv`, `_by_time.csv`, `_by_league.csv`.

No `_by_dispersion.csv`: `n_books` covers 8.3% of snapshots and `book_dispersion` 7.7%. An
optional summary on that coverage would be noise wearing a filename. Brief section 10 stays open.

## 3. Tests run

`python scripts/v11_tests.py` → **111 checks, all passing** (43 pre-existing, 68 new).

Covering every item the brief names: residual, signed movement, toward-classification, magnitude,
time bucketing, residual bucketing, pre-kickoff close selection, post-kickoff exclusion, OVER
direction, UNDER direction, CLV sign convention, missing-close behaviour, duplicate snapshots,
chronological ordering. Plus the fair-probability identity, quality-flag boundaries, and the
clustering regression.

The UNDER cases are constructed so that a bug reading `odds_over25` for both sides produces a
**wrong sign**, not merely a different number — the brief's warning that "direction/sign bugs
could completely invalidate this research" is the reason.

`python -m src.monitoring.weekly_audit` → **23 PASS, 4 INFO, 2 WARN, 0 FAIL**.

## 4. Two bugs found in my own work, both fixed before publishing

**4.1 Pseudo-replicated confidence interval.** The first implementation used a raw Wilson
interval at both units. The same data read:

| unit | toward | 95% CI | p |
|---|---|---|---|
| fixture, n=181 | 54.7% | [46.4%, 62.8%] | 0.267 |
| snapshot, n=5,919 — **wrong** | 52.6% | [51.3%, 53.9%] | **0.0001** |

The second is not more evidence; it is 181 fixtures counted ~30× each with the interval shrunk by
the square root of the replication. Had it shipped beside a correctly clustered magnitude CI, the
direction would have looked established while the magnitude looked uncertain — purely from which
statistic got the right method. A proportion is the mean of a 0/1 variable, so the same cluster
bootstrap applies. Corrected: 52.6% [45.0%, 60.8%], p = 0.515, agreeing with fixture level.

Guarded by a regression test: 12 fixtures × 20 identical snapshots must not narrow the CI.

**4.2 Clobbered provenance.** The Pro importer set a `source_sha` column to v11's commit, but
`season_store.append` does `out["source_sha"] = source_sha or ""` unconditionally and
`pro_collect` passes the collect run's **v9** sha. The importer logged v11 `5dcb7030db9e` while
the stored row said v9 `d61fbaf099dd` — provenance pointing a future investigation at the wrong
repository. Renamed to `calculation_sha`, which cannot collide and pairs with
`calculation_version`. The erroneous partition was **moved to quarantine, not deleted**.

## 5. Current sample

| | |
|---|---|
| snapshot observations | 9,491 over 181 fixtures |
| eligible (`clv_quality == OK`) | 9,265 (97.6%) over 181 fixtures |
| eligible **and** price moved | 5,919 snapshots / **137 fixtures** |
| excluded `MISSING_KICKOFF` / `INSUFFICIENT_BOOKS` | 182 / 44 |
| **eligible entry span** | **2026-08-17 → 08-19 — 3 days, one ISO week** |

## 6. Headline results

| | fixture (primary) | snapshot (clustered) |
|---|---|---|
| n moved | 137 | 5,919 |
| **toward Wowza** | **54.7%** | 52.6% |
| **95% CI** | **[46.4%, 62.8%]** | [45.0%, 60.8%] |
| p vs 50% | 0.267 | 0.515 |
| **mean signed move** | **+0.220pp** | +0.204pp |
| **95% CI** | **[−0.119, +0.559]** | [−0.093, +0.522] |
| median signed move | +0.240pp | +0.220pp |
| when correct / when wrong | +1.535 / −1.370pp | +1.500 / −1.236pp |
| mean \|move\| | 1.460pp | 1.375pp |
| P(≥0.5pp) / P(≥1pp) / P(≥2pp) | 75.2% / 48.9% / 24.8% | 73.9% / 48.5% / 23.1% |
| **clean CLV** n / mean / median | **137 / +0.185% / +0.000%** | 5,919 / +0.086% / +0.000% |
| positive CLV | 35.8% | 24.2% |
| mean entry / close odds | 2.175 / 2.173 | 2.148 / 2.149 |
| **placebo (fixed anchor)** | **46.7%** → excess **+8.0pp** | — |

### By model type (fixture-level)

| | fixtures | toward | mean signed | mean CLV | status |
|---|---|---|---|---|---|
| new_format | 119 | 56.5% | +0.306pp | +0.165% | EARLY_SIGNAL |
| standard | 62 | 51.1% | +0.046pp | +0.227% | EARLY_SIGNAL |

new_format better on direction and movement; standard marginally better on executable CLV. Wide,
overlapping intervals — but consistent with the brief's point that prediction quality and
market-timing quality need not coincide.

### By residual bucket (fixture-level; every bucket INSUFFICIENT_SAMPLE)

| band | fix | moved | toward | mean signed | mean CLV |
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

The brief flagged +6…+10 as promising; here it is 63.6% toward with +1.276% CLV on **11 moved
observations**. Encouraging, and far too small to act on. Note the broader asymmetry: negative
bands (model prefers UNDER) look consistently better than positive ones — invisible in the
`abs_residual` view, and worth watching.

### By time to kickoff (snapshot-level, clustered)

| bucket | fix | moved | toward | mean signed | mean CLV |
|---|---|---|---|---|---|
| > 24h | 167 | 5,710 | 53.1% | +0.218 | +0.118 |
| 12–24h | 18 | 101 | 32.7% | −0.370 | −1.437 |
| 6–12h | 14 | 39 | 56.4% | +0.027 | +0.165 |
| 3–6h | 10 | 25 | 40.0% | −0.192 | −0.524 |
| 1–3h | 13 | 38 | 34.2% | −0.069 | −0.736 |
| 30–60m | 10 | 6 | 83.3% | +1.007 | +2.343 |
| 10–30m | 1 | 0 | n/a | n/a | n/a |

**This cannot answer the entry-timing question yet.** 96% of moved snapshots are `> 24h`. The
`30–60m` row rests on **six observations**. Near-kickoff coverage is the binding constraint.

## 7. The five questions, answered separately (281 fixtures / 219 moved)

**A. Is the directional effect statistically interesting?** **No.** 55.7% [49.1%, 62.1%],
p = 0.091 — includes 50%. And the **placebo beats it** (58.4%). No evidence the residual carries
directional information beyond the market's own centrality.

**B. Is the movement magnitude economically interesting?** **Statistically real, economically
marginal, not attributable to Wowza.** +0.521pp, CI [+0.134, +0.908] excludes zero — a genuine
change from the first run. But given A, mean reversion explains it without a model. At ~2.15
average odds, +0.52pp is worth ~0.25% in price terms even if capturable.

**C. Does it produce positive clean CLV?** **No.** Median **+0.000%** at both units; mean +0.101%
per fixture but **−0.024%** across snapshots; 42.5% / 34.3% positive. The best directional band
(+6:+10, 63.6%) has *negative* CLV (−0.391%).

**D. Is it chronologically stable?** **No — demonstrably not.** Span grew 3 → 7 days, still one
ISO week, so the formal test is unassessable. But the placebo verdict *inverted* across those four
days. That is direct evidence of instability, not merely absence of evidence for stability.

**E. Is the sample large enough for deployment?** **No.** 219 moved fixture-level observations
against a 500–1,000 bar. All 19 leagues insufficient.

## 8. Data-quality limitations

1. **Seven days, one ISO week.** Binding.
2. **Result grading — FIXED, awaiting first live run.** `src/results.py` fetches actual goals from
   football-data.co.uk for all 31 mapped leagues; grading prefers them and keeps v9's tip ledger
   only as a fallback for API-Football-only competitions. Every run prints the provenance split.
   This machine cannot reach `football-data.co.uk` at all (TLS inspection; connection refused, not
   a certificate error), so the fetch validates on the first CI run. Logic is unit-tested against
   synthetic frames built to the real schema; a failed fetch degrades to "nothing newly graded"
   with each unavailable league named, never to wrong results.
3. **Time buckets readable but thin at the near end** — mechanical, not coverage. See section 0.2.
4. **Dispersion analysis impossible** — `n_books` 8.3%, `market_prob_range` not stored.
5. `market_move` uses **consensus**, not best executable price.
6. **One market only** — O/U 2.5.
7. `STALE_*` / `SYNTHETIC_ODDS` declared **not assessable**, not proxied.
8. **Re-ingest overlap in the archive.** v11 recomputes over a growing history, so each ingest is
   a superset: the canonical table holds 34,543 rows describing 25,052 distinct observations.
   Correct for an immutable archive, but any aggregate must dedupe by `snapshot_id` keeping the
   latest `ingested_at`. The audit reports the ratio every run.

## 9. Overall verdict

**The hypothesis is now weakly disfavoured rather than merely unproven.**

The measurement infrastructure is sound, tested (142 checks), archived with versioned provenance,
and audited weekly. What it currently says:

- The directional signal is not distinguishable from chance, **and is beaten by a constant**.
- Mean signed movement is statistically positive — but that is what mean reversion looks like when
  the model's probabilities sit more centrally than the market's.
- Executable CLV is zero at the median and negative across snapshots.
- The one result that looked encouraging four days ago has inverted.

**Keep collecting, change nothing, and do not build the movement model.** Section 10 of
`MARKET_MOVEMENT_RESEARCH.md` sets prerequisites; the placebo criterion is now failing outright,
which is a hard stop rather than a slow one.

The honest reading is the one the brief asked for in advance: *"If the current ~58% effect
disappears with more data, accept the result."* It has not fully disappeared — 55.7% — but the
control that distinguishes skill from centrality has gone the wrong way, and that is the number
that matters. Re-run at ≥500 fixture-level observations spanning ≥8 weeks.
