# Results: self/other boundary study -- gpt-5.6-sol smoke test

Subject model: `gpt-5.6-sol` (reasoning_effort='none' on every subject call).
"Other" corpus messages are actually written by `gpt-4o-mini` (reused from the original study's `data/corpus_other.json`); the "other" label sentence names a **mock competitor, "Fable 5"**, per the user's request -- this name does not correspond to any model actually queried in this run.

Reduced-scope smoke test: 6 rounds (not 10), 15 rollouts/cell (not 80), 6x3 noise floor (not 20x8). Wider CIs than the original study; treat as a pilot, not a replication.

**Actual total spend (this run): $1.3247** (kill switch $5.00, hard budget $5.75).

**Invalid-parse rate (main): 0/75**

## Noise floor

6 paraphrase variants, 18 rollouts. range = 1.000, sd = 0.384 (0-10 scale).

## Cell means (bootstrap 95% CI, 10,000 resamples)

| Cell | Mean | 95% CI | n (ok/invalid) |
|---|---|---|---|
| `baseline` | 7.067 | [6.844, 7.300] | 15/0 |
| `self_self` | 6.900 | [6.667, 7.111] | 15/0 |
| `self_other` | 7.222 | [7.011, 7.444] | 15/0 |
| `other_self` | 6.956 | [6.756, 7.167] | 15/0 |
| `other_other` | 7.278 | [7.067, 7.478] | 15/0 |

## Contrasts

| Contrast | Point estimate | Noise floor range |
|---|---|---|
| Label main effect | -0.644 | 1.000 |
| True-source main effect | -0.111 | 1.000 |
| Interaction | 0.000 | 1.000 |
