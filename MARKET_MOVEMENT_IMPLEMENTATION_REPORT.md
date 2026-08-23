# Market Movement Research — Implementation Report

Date: 2026-08-23 · v11 `c7b0b23` · Pro `4709d4a` · v9 `d61fbaf0`

> **RESEARCH ONLY.** No football model was retrained or altered. No threshold changed. No
> Telegram, no staking, no live deployment, no V12. Definitions and caveats:
> `MARKET_MOVEMENT_RESEARCH.md`.

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

## 7. The five questions, answered separately

**A. Is the directional effect statistically interesting?**
Marginally, and **not significantly**. 54.7%, CI [46.4%, 62.8%] includes 50%, p = 0.267. The one
genuinely encouraging element is the placebo: the market moves toward an information-free anchor
only 46.7% of the time, so the +8.0pp excess is **not** mean reversion — which was the most
likely benign explanation and it has been ruled out at this sample size. Worth continuing to
collect. Not established.

**B. Is the movement magnitude economically interesting?**
**No.** +0.220pp mean with CI [−0.119, +0.559] including zero. At ~2.17 average odds that is
worth roughly 0.1% in price terms. Wins and losses are near-symmetric (+1.535 vs −1.370pp), so
the economics rest entirely on the thin directional edge rather than on winning big and losing
small — exactly the case the brief's section 4 warns about.

**C. Does it produce positive clean CLV?**
**Not meaningfully.** +0.185% mean, **+0.000% median**, only **35.8%** positive. A mean lifted
above zero by a minority of large moves, with the median exactly zero and two-thirds
non-positive, is not a monetisable price edge. And note: signed movement and fair-probability CLV
are algebraically the same measurement, so `clv_pct` is the only CLV figure that adds information
— and it is ~0.

**D. Is it chronologically stable?**
**Cannot be assessed.** All eligible entries fall in a single ISO week — three days. Monthly and
weekly rows are emitted as `NOT_ASSESSABLE` rather than as duplicates of the overall row. The
halves split (48.5% / 56.7%) is a split *within* three days and is as likely fixture mix as
drift. **The entire headline rests on three days of entries.**

**E. Is the sample large enough for deployment?**
**No.** 137 moved fixture-level observations against a 500–1,000 bar. Every residual band and
every one of 16 leagues is `INSUFFICIENT_SAMPLE`. Not close.

## 8. Data-quality limitations that matter

1. **Three days of eligible entries.** Everything else is secondary.
2. **Time-to-kickoff unusable** — 96% of moved snapshots in one bucket.
3. **Result/ROI coverage is biased, and the bias is in the variable under study.** v11 grades
   from v9's public `bets_ledger.csv`, which holds only fixtures **v9 tipped** — 94/336 (28%).
   League Two is 0/24, Serie B 0/11, Ireland 0/5. The graded subset is conditioned on v9 having
   disagreed with the market enough to fire a tip. **Movement and CLV are unaffected** (prices
   are unconditioned), but every result-based figure comes from that conditioned subset. Fix:
   grade from actual results (football-data.co.uk covers E0–E3 and the European standard
   leagues) rather than from a tip ledger. **Not yet done — recommended next.**
4. **Dispersion analysis impossible** — `n_books` 8.3%, `market_prob_range` not stored at all
   and left NULL rather than approximated from the std.
5. `market_move` uses **consensus**, not best executable price.
6. **One market only** — O/U 2.5. Nothing extends to BTTS or side markets.
7. `STALE_ENTRY_PRICE`, `STALE_CLOSE_PRICE`, `SYNTHETIC_ODDS` are declared **not assessable**;
   the snapshots carry no per-book quote timestamps or provenance marker, and a proxy would
   misclassify a genuinely settled market.

## 9. Overall verdict

The hypothesis in brief section 24 — *Wowza does not beat the closing market on outcomes, but may
reach the future price first* — **is not refuted, and is not supported either.** What can be said
after this work:

- The directional signal is positive but statistically indistinguishable from chance.
- It **survives its most dangerous placebo**, which is the one real piece of good news: the +8pp
  excess over an information-free anchor means the headline is not just mean reversion.
- The magnitude is economically negligible and its interval includes zero.
- Executable CLV is ~0 at the median with only a third of observations positive.
- Stability is unassessable on three days of data.

So: **keep collecting, change nothing.** The measurement infrastructure now exists, is tested,
archives to canonical storage with versioned provenance, and is audited weekly. The single
highest-value next step is not analysis — it is **near-kickoff snapshot coverage**, because the
entry-timing question is the one that would make this actionable and it currently has 6–101
observations per bucket. Second is unbiased result grading (limitation 3).

Re-run this report at ≥500 fixture-level observations spanning ≥8 weeks. If the 54.7% decays
toward 50% with more data, that is the answer and it should be accepted.
