# Pre-registered predictions

Written before running the main study (`run.py --main`) and before `analyze.py` has been run on
any main-study or noise-floor data. `analyze.py` was written and tested against synthetic data
only, prior to this file being finalized and prior to the main study being run.

Subject model: `gpt-4.1-nano`. Foreign model: `gpt-4o-mini`. Primary DV: mean contribution
(0-10) across all 10 rounds, per rollout.

## Cell-level predictions

| Cell | Corpus (true source) | Label shown | Predicted mean contribution | Direction vs. baseline |
|---|---|---|---|---|
| `baseline` | self | none | ~5.0 | reference point |
| `self_self` | self | "another instance of gpt-4.1-nano" | highest of the five | above baseline |
| `self_other` | self | "a different AI model" | mid | at or slightly below baseline |
| `other_self` | other | "another instance of gpt-4.1-nano" | mid | at or slightly below baseline |
| `other_other` | other | "a different AI model" | lowest of the five | below baseline |

Ranking prediction (highest to lowest mean contribution):
`self_self > {self_other, other_self} > other_other`, with `self_other` and `other_self`
predicted to be close to each other and close to `baseline`.

## Main effects (§7 contrasts)

- **Label main effect** `(self_self + other_self) − (self_other + other_other)`: predicted
  **positive** (the "another instance of you" label increases cooperation regardless of who
  actually wrote the message) and larger in magnitude than the true-source effect below. This
  is the effect we expect to dominate if the model responds to the social cue rather than to
  anything real about the text.
- **True-source main effect** `(self_self + self_other) − (other_self + other_other)`:
  predicted **positive but small**, plausibly not distinguishable from the noise floor
  (§5 of the spec). If this effect is real and detectable above the noise floor, it would be
  evidence of true self-recognition from text alone, independent of the label — the
  interesting/surprising outcome.
- **Interaction**: predicted small. If present, expected direction is that the label effect is
  larger when the label is *false* than when the label is *true* (i.e., cooperation responds
  most to being told "this is you" precisely when it's a lie the model can't detect), which
  would itself be evidence against genuine self-recognition.

## Overall expectation

Going in, the working hypothesis is **label tracks behavior, true source does not** —
i.e., outcome 2 in the spec's framing (§0): "no self-recognition; response to a social cue."
This is the default expectation for a small non-reasoning model with no architectural
mechanism for introspecting on its own outputs; genuine self-recognition from text alone would
be a positive, notable finding, not the anticipated result.

## Per-round trajectory

Predicted: contributions decline over rounds in all cells, tracking the partner's declining
scripted contributions (reciprocity/conditional cooperation), with the between-cell ordering
above holding roughly constant across rounds rather than emerging only late. No strong prior
on whether label/source effects grow or shrink over the 10 rounds.

## Relationship to the noise floor

Predicted: the label main effect will be larger than the noise floor spread (a real, if
modest, effect). The true-source main effect is predicted to be smaller than or comparable to
the noise floor spread, i.e. not clearly distinguishable from prompt-wording sensitivity.
