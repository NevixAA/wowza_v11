# Future Decision Engine — DESIGN ONLY, NOT DEPLOYED

**Status: documentation.** Nothing described here is built, wired, or scheduled. Brief section 29
asks for the eventual architecture to be written down; sections 16 and 30 forbid training the
models it depends on. Both hold at once: this is the target to aim at, and today's evidence does
not justify aiming yet.

Read `MARKET_MICROSTRUCTURE_RESEARCH_REPORT.md` first. Its conclusion is that the residual does
not beat a model-free baseline, which means **the second stage of this pipeline currently has no
signal to carry.** The design is recorded anyway, because knowing the shape tells you which
measurement would matter next.

## The intended flow

```
V9 FOOTBALL MODEL ──────────► p_wowza          (independent football information)
MARKET ─────────────────────► p_market         (de-vigged consensus)
                                  │
                                  ▼
                            RESIDUAL = p_wowza − p_market
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
  MODEL B                    MODEL C                   MODEL D
  P(move toward Wowza)       E[signed move]            E[clean CLV]
        └─────────────────────────┼─────────────────────────┘
                                  ▼
                        BEST EXECUTABLE PRICE
                                  ▼
                        CONSERVATIVE EV (lower bound)
                                  ▼
                            BET / NO BET
```

## Preconditions, in the order they must be met

**1. V9 independence must survive.** Section 1: V9 is never trained on closing prices, future
market movement, or CLV. The whole construction depends on `p_wowza` and `p_market` being
independently sourced — feed market data into V9 and the residual stops measuring disagreement and
starts measuring the market's own autocorrelation. This is the constraint most easily broken by
accident and the one that would invalidate everything downstream without any error appearing.

**2. There must be a signal.** Model B predicts P(move toward Wowza). Today that target is met at
58.00% by the residual and at **58.33%** by a fixed anchor carrying no model input. Training B now
would fit the fact that V9's probabilities are more central than the market's (sd ratio 0.564,
corr with the mean-reversion term +0.833) and report it as skill. **This is the binding
precondition and it is not close to met.**

**3. Sample and calendar spread.** Section 17: ≥500 clean fixture observations, 1000+ preferred,
spanning multiple weeks, months, leagues and market regimes. Current: **300 fixtures over 7 days
and 1 calendar month**. Calendar spread, not row count, is the tighter constraint — a
single-month sample of 1000 would still fail.

**4. Out-of-time confirmation.** Section 22: freeze the hypothesis on a development period, then
evaluate unchanged on a later period. The candidates worth freezing today are (a) tight
bookmaker consensus (+10.87pp over placebo at n=46) and (b) entry ~24h (+3.43pp at n=175). Both
are hypotheses, both are far too thin to act on, and both must be evaluated on data that did not
generate them.

**5. Model D probably supersedes B and C.** CLV is closest to the economic decision. But note the
measured identity: corr(signed move, fair-probability move) is **+1.0000 exactly**, and corr with
`clv_pct` is +0.858. Movement and CLV are close to one quantity, not three independent targets —
so B, C and D are far less independent than the diagram implies, and fitting all three would
mostly be fitting the same thing three times.

## What each stage would need

| Stage | Target | Blocked on |
|---|---|---|
| Model B | P(move toward Wowza) | a residual that beats the anchor placebo |
| Model C | E[signed move] | same, plus enough spread to estimate magnitude |
| Model D | E[clean CLV] | same; CLV CI currently includes zero |
| Best executable price | max over books | **per-book quotes are not persisted** |
| Conservative EV | lower bound, not point estimate | the three models above |
| BET / NO BET | — | everything above, plus out-of-time confirmation |

Two of these are blocked on data rather than on modelling. **Best executable price cannot be
computed at all today**: per-book prices are consumed inside `book_consensus.build_index` and
never stored, so only the consensus survives. That also blocks bookmaker lead/lag (section 7) and
sharp-vs-consensus (section 6). It is the highest-value collection change available and it must be
collected forward — there is no historical purchase.

## Design commitments worth fixing now

- **Default NO_BET.** The existing `edge_engine` already does this and it should not change: a
  decision engine whose failure mode is "no bet" degrades safely.
- **Conservative EV means a lower bound**, never a point estimate. The current engine already
  applies an uncertainty lower bound before the EV floor.
- **The CLV gate stays.** `BET` requires a segment's clean CLV count ≥ `MIN_CLV_N` (150) and
  positive. No league is anywhere near that (largest clean n = 39), so in practice everything
  would be `PAPER` regardless — which is the correct outcome and worth stating rather than
  treating as a bug.
- **Reuse one definition per quantity.** Section 26: velocity, dispersion and CLV must not
  acquire three implementations across three repos. Dispersion is already passed through from
  `book_consensus` rather than recomputed, and the microstructure module shares `MIN_MOVE_PP`
  with `movement`.
- **Every stage reports its placebo.** The lesson of this round is that a plausible number with
  no model-free control is not evidence. Any future B/C/D must publish its baseline alongside
  its score.

## What would move this forward

Nothing in this document. In order:

1. Keep collecting — calendar spread is the binding constraint and it needs no code.
2. Persist per-book quotes — unblocks three separate sections and the executable-price stage.
3. Re-run `scripts/v11_microstructure.py` when the sample spans several months, and check whether
   the residual beats the anchor placebo with a CI that excludes zero.

If step 3 comes back negative again on a multi-month sample, the price-discovery hypothesis should
be retired rather than re-specified, and this document deleted rather than extended.
