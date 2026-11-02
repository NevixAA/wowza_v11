# v11 corrections (Data Truth round 2, 2026-11-02)

Corrected values sit BESIDE the originals. No committed history file was rewritten:
`v11_shadow_snapshots.csv`, `v11_momentum_control.csv` and `v11_placebo_table.csv` are untouched.

## v11_p_market_restated.csv

**What changed.** Before 2026-11-02, `v11_p_market` switched estimator between snapshots of the same fixture. On 1,448 of 2,221 fixtures it moved among three estimators:

- an exchange median, with no quality gate;
- a cross-book median;
- one book's de-vig, with a proportional fallback.

**The one estimator.** This file restates every snapshot as of its own time with one estimator, `src/book_consensus.ESTIMATOR` = `power_median_gated`:

- each book's pair is de-vigged with the power method;
- a pair counts only when its overround is inside [1.00, 1.25] (Pro's gate);
- the result is the median across books;
- exchanges count as ordinary books.

**Fewer than 3 books.** The same estimator runs on v9's own pair. The function is `consensus_from_books`, which the live shadow also uses, so the two cannot drift.

**Inputs.** v9 `output/book_odds_snapshots.csv` plus `output/book_odds_snapshots_archive/` at wowza-betting origin/main 8b802bc1. The latest quote per book and side, at or before each snapshot, is used. The value is P(OVER).

**Coverage.** 83,173 snapshots, the full history on main to 2026-11-01 23:41 UTC (round 5 re-run, 2026-11-02, v9 at wowza-betting origin/data-truth-100 6931e604). On 87 of them the exchange anchor had failed the gate (`exchange_anchor_failed_gate`). A full re-run reproduced the 82,831 round-4 rows exactly on every value column. Those rows are kept byte for byte, and the 342 rows collected on main since then were appended (`--append`).

**P(OVER), always.** Before round 5 the shadow stored the CHOSEN side's probability in `v11_p_market`, which was P(UNDER) whenever the UNDER decision ranked higher (on rows whose `v11_side` is `-` too). About 540 of 83,172 published values sit within 0.02 of 1 - P(OVER) rather than of P(OVER). This sidecar is P(OVER) on every row. From round 5 the shadow writes P(OVER) in `v11_p_market` with `v11_p_market_basis = over`, and the side's own value in `v11_p_market_side`. A row keeps its own value in research only when it carries both `p_market_estimator` and `v11_p_market_basis = over`. Every other row takes this sidecar.

**Keeping it current.** CI runs `scripts/v11_restate_p_market.py --v9 "$V9_LOCAL" --append` before the research steps. Append mode keeps every existing row byte for byte. It adds rows only for snapshots that are not in the sidecar and do not carry the one estimator themselves. `output/corrections/` is in the workflow's commit list.

## v11_movement_restated_2026-11-02.csv and v11_microstructure_restated_2026-11-02.csv

`v11_market_movement.py` and `v11_microstructure.py` now read the one-estimator P(OVER) through `src/book_consensus.research_snapshots`, as `v11_momentum_control.py` already did. The committed tables were left untouched. `scripts/v11_restate_research.py` stacks three variants of every table:

- `as_published`: the committed file as it stood;
- `original_estimator_recomputed`: the current code on the published `v11_p_market`;
- `one_estimator_restated`: the current code on the one-estimator P(OVER).

The first two are identical, so the whole difference comes from the estimator. History is 83,173 snapshots, 2,025 eligible fixtures.

| measure | as published | one estimator |
|---|---|---|
| toward Wowza, fixture level (earliest entry) | 56.18% [53.86, 58.47], 1,780 moved | 54.60% [52.25, 56.93], 1,738 moved |
| placebo, fixed anchor | 57.58% | 55.18% |
| toward Wowza, snapshot level | 52.97% [50.24, 55.61] | 50.42% [47.54, 53.32] |
| mean signed move, fixture level | +0.493 pp | +0.337 pp |
| microstructure edge vs placebo, entry earliest / ~24h / ~12h / ~6h / ~3h / ~1h | -1.40 / -1.07 / -0.17 / -0.09 / -0.19 / -1.00 pp | -0.58 / -3.68 / -4.61 / -3.61 / -4.10 / -3.91 pp |
| residual coefficient beside reversion (logistic, z) | 0.137 (1.88) | 0.120 (1.63) |

Under one estimator the model still does not beat the model-free placebo at any entry point. The snapshot-level toward rate is no longer distinguishable from 50%.

## v11_momentum_restated_2026-11-02.csv

The momentum headline is the residual-only coefficient. Window 720/720 min, 200 bootstraps.

| variant | residual only | + full controls | shuffled placebo |
|---|---|---|---|
| as published | 0.0123 [0.0085, 0.0165] | 0.0121 | 0.0052 |
| 86 failed-gate exchange rows removed | 0.0072 [0.0049, 0.0095] | 0.0069 | 0.0028 |
| one estimator (restated) | 0.0040 [0.0018, 0.0060], p=0.02 | 0.0025 [-0.0009, 0.0057], p=0.10 | 0.0007 (p=0.41) |

Under one estimator, the controlled residual coefficient is no longer distinguishable from zero. `scripts/v11_momentum_control.py` now reads the one-estimator value: from the row itself, or from the sidecar for older rows. It drops any row that has neither.
