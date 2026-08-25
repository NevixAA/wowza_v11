# The research pipeline: what produces what, and how it can go stale

Written because it went stale. On **2026-08-25** v11's raw archives were current to the minute
while every movement summary was two days old, and the workflow was green throughout.

---

## The derivation graph

```
v9 public data (predictions.csv, bets_ledger.csv)
        │
        ▼
  v11_shadow.py ─────────────► v11_shadow_snapshots.csv   APPEND-ONLY history
        │                      v11_shadow_log.csv         latest-per-fixture view
        ▼
  v11_grade.py ──────────────► v11_graded.csv             + v11_scoreboard.csv
        │                      (football-data.co.uk results, v9 ledger as fallback)
        ▼
  v11_fit_evidence.py ───────► v11_evidence.json          N_EFF / ECE, feeds residual
        ▼
  v11_residual.py ───────────► v11_residual.csv           market vs market+Wowza
        ▼
  v11_market_movement.py ────► v11_market_movement_detail.csv   one row per snapshot
                               v11_movement_summary.csv
                               v11_movement_by_residual.csv
                               v11_movement_by_model.csv
                               v11_movement_by_time.csv
                               v11_movement_by_league.csv
        ▼
  src/research_state.py ─────► research_state.json        the clock
                               v11_research_health.json   PASS / WARN / FAIL
        ▼
  commit  ──►  Pro ingests (src/importers/v11_research.py) ──► movement_observations
```

Every step runs in one job, in this order, in `.github/workflows/v11_collect.yml`
(cron `25,55 8-23`, `concurrency: v11-collect` with `cancel-in-progress: false`,
`timeout-minutes: 15`).

## Per-artifact detail

| artifact | producer | inputs | update condition | on failure |
|---|---|---|---|---|
| `v11_shadow_snapshots.csv` | `v11_shadow.py` | v9 predictions | every run, append-only | step fails, run red |
| `v11_shadow_log.csv` | `v11_shadow.py` | v9 predictions | every run, overwritten | step fails, run red |
| `v11_graded.csv` | `v11_grade.py` | shadow log + football-data | every run; shrink refused outside CI | falls back to v9 ledger, warns |
| `v11_scoreboard.csv` | `v11_grade.py` | graded | every run | as above |
| `v11_evidence.json` | `v11_fit_evidence.py` | graded | when enough settled history | writes nothing, fallback stands |
| `v11_residual.csv` | `v11_residual.py` | shadow log + **graded** | every run | step fails, run red |
| the six movement files | `v11_market_movement.py` | snapshots | every run, deterministic | step fails, run red |
| `research_state.json` | `src/research_state.py` | all of the above | every run | step fails, run red |
| `v11_research_health.json` | `src/research_state.py` | as above | every run | step fails, run red |

## How it actually broke

**Root cause 1 — the six movement files were never staged.** `v11_market_movement.py` was
rewritten on 08-23 to emit six files instead of one; the workflow's commit list was not updated.
So CI recomputed all six every 30 minutes and discarded them. Nothing failed: `git add -f` on an
unchanged path is a no-op, `|| true` swallowed the miss, and the list still named
`v11_market_movement.csv` — a file the rewritten script no longer writes. The committed versions
were the ones a human had pushed by hand.

**Root cause 2 — `v11_residual.py` had its own grading path.** It called
`_result_lookup(_load_v9("bets_ledger.csv"))`, re-deriving outcomes from v9's **tipped** bets. The
ledger holds only fixtures v9 bet on, so the experiment was capped at 194 fixtures while
`v11_graded.csv` had 344 — and the sample it used was conditioned on v9 disagreeing with the
market, which is the variable a market-relative test measures. Residual was never *stale*; it was
reading a worse source. `v11_grade.py` moved to real results on 08-23 and this second path was
missed.

Both are the same underlying mistake: **a change was made in one place and its consumers were not
enumerated.**

## The freshness contract

`src/research_state.py` compares each derived artifact's `generated_at` against the newest
observation in its declared sources.

```
NO NEW ELIGIBLE DATA        source did not advance  -> unchanged output is PASS
ANALYSIS FAILED TO REFRESH  source advanced, output did not -> WARN >=12h, FAIL >=30h
```

Both look like an unchanged file from the outside; the comparison is the only thing that separates
them. Thresholds come from the real cadence (runs every 30 min, ~9h overnight gap), so neither can
fire on a single skipped run.

Three details that are load-bearing:

- **`generated_at` is a COLUMN, not a file mtime.** `git checkout` resets mtime, so on a fresh CI
  runner every artifact would look seconds old and the check could never fire.
- **Source timestamps must be WRITE times, never fixture dates.** Keying `v11_graded.csv` on
  `date` reported its newest observation as a week in the future and made every downstream
  artifact look 133h stale. Future-dated values are now discarded.
- **Sample size is deduped on `snapshot_id`, never a raw row count.** The archive holds supersets
  from repeated ingests — 34,543 rows describing 25,052 observations.

## Monotonicity

`n` should not shrink. It legitimately can (methodology change, tightened quality rule, duplicate
correction, quarantine), so a decrease is a **WARN carrying the delta**, never an automatic
failure. Tracked for movement observations, movement fixtures, published fixture n, clean CLV n,
residual n and graded settled count.

`_write_derived` additionally **refuses a local downgrade**: a laptop usually cannot reach
football-data.co.uk, so running `v11_grade.py` locally rewrote `v11_graded.csv` from 344 settled
fixtures to 186. Outside CI a shrink now raises; in CI it is allowed but printed.

## Failure visibility

The workflow's last step reads `v11_research_health.json` and exits non-zero on `FAIL`, emitting a
`::error` annotation. It runs **after** the commit — placed before it, a stale verdict prevented
the freshly collected snapshots from being persisted, and odds snapshots cannot be re-collected.

## Pro's side

`src/importers/v11_research.py` reads `v11_research_health.json` before archiving and tags rows
`V11_RESEARCH_STALE` / `V11_MOVEMENT_STALE` / `V11_CALC_VERSION_MISMATCH` /
`V11_RESEARCH_HEALTH_MISSING` in a `research_staleness` column. Stale rows are still **archived**
— they record what v11 published — but tagged so no aggregate mistakes them for current.
Pro's `weekly_audit` has a `v11 freshness` area reporting v11's verdict, both lags, sample sizes
and monotonicity, using v11's own thresholds so the repos cannot disagree.

## Known remaining issues

- `newformat_odds_dense.csv` has no `kickoff_utc`, so its T-minus buckets are not computable.
- `clv_pct` is a **fraction** in `clv_records.csv` and a **percent** in `bets_ledger.csv`;
  `CLV_PLAUSIBLE_ABS = 25` is therefore inert against the former.
- v9 freshness verdicts are only authoritative **from CI**. Locally the audit measures the local
  clone, which reported production as down twice while it was committing every few minutes; those
  verdicts are now downgraded and labelled with the clone's age.
