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

**Coverage.** 82,831 snapshots (to 2026-11-01 15:17 UTC). On 87 of them the exchange anchor had failed the gate (`exchange_anchor_failed_gate`).

**Keeping it current.** Re-run on the full history before this branch merges, because main has collected more snapshots since:

```
python scripts/v11_restate_p_market.py --v9 <wowza-betting checkout>
```

Rows written after the merge carry `p_market_estimator` themselves.

## v11_momentum_restated_2026-11-02.csv

The momentum headline is the residual-only coefficient. Window 720/720 min, 200 bootstraps.

| variant | residual only | + full controls | shuffled placebo |
|---|---|---|---|
| as published | 0.0123 [0.0085, 0.0165] | 0.0121 | 0.0052 |
| 86 failed-gate exchange rows removed | 0.0072 [0.0049, 0.0095] | 0.0069 | 0.0028 |
| one estimator (restated) | 0.0040 [0.0018, 0.0060], p=0.02 | 0.0025 [-0.0009, 0.0057], p=0.10 | 0.0007 (p=0.41) |

Under one estimator, the controlled residual coefficient is no longer distinguishable from zero. `scripts/v11_momentum_control.py` now reads the one-estimator value: from the row itself, or from the sidecar for older rows. It drops any row that has neither.
