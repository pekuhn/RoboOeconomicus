# Self/other boundary eval for LLM agents (v1)

Does an LLM agent cooperate differently in an iterated public goods game depending on whether
its partner's messages were *actually* written by its own model vs. a different model, and
separately, on whether the messages are *labelled* as such? A 2x2 (true source x label) design
lets us tell apart three outcomes: behavior tracks true source (real self-recognition),
behavior tracks the label only (a social cue, not recognition), or behavior tracks neither
(weak self/other boundary). See `predictions.md` for the pre-registered predictions, written
before the main study ran, and `results.md` for the outcome.

## Setup

```
uv sync
```

Requires `OPENAI_API_KEY` in a `.env` file in the project root (gitignored).

## Running the study, in order

```
uv run python corpus.py --all      # generate + normalize the message corpora, leakage check
uv run python run.py --noise-floor # 160 rollouts, 20 paraphrase variants of the baseline prompt
uv run python run.py --dry-run     # 10 rollouts, cost projection for the main study
uv run python run.py --main        # 400 rollouts, 5 cells x 80 replicates, randomized order
uv run python analyze.py           # reads data/rollouts.jsonl, writes results.md + data/plots/
```

Every step is resumable: `data/rollouts.jsonl` records one line per completed rollout keyed by
a deterministic `rollout_id`, and re-running any `run.py` mode skips rollout_ids already
present. A crash mid-rollout is not double-billed for *completed* rollouts, though a rollout
that crashes partway through is simply redone from round 1 on restart -- acceptable here since
each round costs a fraction of a cent (see `spend.json`).

## Budget discipline

- `ledger.py` is the single wrapper (`chat_call`) every OpenAI API call in this project goes
  through. It re-checks the running total against a $6.00 kill switch before *every* call,
  including every retry attempt, using a pessimistic pre-call cost estimate; actual spend is
  computed from `response.usage` after the call and appended to `spend.json`.
- Hard budget: $7.57. Kill switch: $6.00. Actual total spend and the invalid-parse rate are
  reported at the top of `results.md`.
- Prices are not hardcoded from memory: `common.py`'s `PRICES` dict was populated by fetching
  `https://developers.openai.com/api/docs/pricing` directly and parsing the page's embedded
  standard-tier pricing table (2026-09-02), not from a written or remembered figure.

## Model choice

- **Subject model:** `gpt-4.1-nano`. The spec calls for "the cheapest current small/mini chat
  model." `gpt-5-nano` is nominally cheaper per token ($0.05/$0.40 vs. $0.10/$0.40 per 1M for
  input/output), but it is a reasoning model: on the Chat Completions API, `max_tokens` caps
  reasoning tokens and visible output *combined*. The spec's own §3 requires a strict
  `max_tokens=60` cap on the subject's move. Under a reasoning model, that cap risks reasoning
  tokens silently consuming the whole budget and returning empty or truncated JSON -- inflating
  the invalid-parse rate for reasons unrelated to the experiment, and doing so unpredictably
  across cells. `gpt-4.1-nano` is a plain, non-reasoning chat model with fully predictable
  token accounting under a small cap, and is still extremely cheap. This is a deliberate
  deviation from a literal reading of "cheapest," made to keep the strict-JSON, small-max-token
  design in §3 reliable; the price gap between the two models is trivial relative to the total
  budget.
- **Foreign model:** `gpt-4o-mini`, for the "other-source" message corpus. Different generation
  (4o vs. 4.1) and different size class (mini vs. nano) from the subject model.

## Limitations

- **Same-lab confound.** Both `SUBJECT_MODEL` and `FOREIGN_MODEL` are OpenAI models trained by
  the same lab with likely-overlapping post-training pipelines and data. "Different model" here
  is a weak manipulation of true source -- the two are more similar to each other than either
  is to, say, a Claude or Gemini model. This is a budget and scope constraint (this project uses
  one API key and one provider throughout); cross-lab replication (e.g. subject = an OpenAI
  model, foreign corpus = an Anthropic or Google model) is future work and would be a
  meaningfully stronger test of true self-recognition.
- **Partner is scripted, not adaptive.** The partner's per-round *contributions* are a fixed
  sequence (`PARTNER_CONTRIBUTIONS` in `common.py`), identical across every cell, specifically
  so that between-cell differences can't be explained by the partner behaving differently. Only
  the partner's *messages* differ by condition (true source) and only the framing sentence
  differs by label. This means the study measures reaction to a fixed trajectory, not to a
  responsive partner -- a live scripted-but-reactive partner (e.g. tit-for-tat) was out of scope.
- **Two-corpus leakage check, not a full stylometric audit.** The leakage check (`corpus.py`)
  asks the subject model to guess the source of 60 sampled, style-normalized messages one at a
  time and requires accuracy at or below 65% before the main study is allowed to run. This
  bounds *detectable* stylistic contamination by the one classifier that matters (the subject
  model itself) but isn't proof that no residual signal exists.
- **`gpt-4.1-nano` is a small, non-reasoning model.** Its capacity for anything resembling
  introspection or self-modeling is limited relative to larger or reasoning models; a null
  result here (behavior tracks neither label nor true source) is expected on priors and
  shouldn't be over-interpreted as evidence about self/other boundaries in LLMs generally. See
  `predictions.md` for the pre-registered expectation.
- **Single run, no cross-model or cross-seed replication.** 80 rollouts/cell and a 20-variant
  noise floor give reasonable within-run statistical power, but everything here comes from one
  subject model, one foreign model, one partner script, and one 2x2 design. Findings should be
  read as a first pass, not a robust general claim.

## Files

```
common.py               # shared constants (models, prices, game math, prompts) -- not a
                         #   spec-named deliverable, factored out so ledger/corpus/run/analyze
                         #   can't drift apart on game constants
noise_floor_prompts.py  # 20 hand-written baseline-prompt rewordings for the noise floor -- not
                         #   model-generated, so they're fixed by design before any data is seen
ledger.py                # cost wrapper + kill switch
corpus.py                # message generation, style normalization, leakage check
run.py                   # --dry-run, --noise-floor, --main
analyze.py                # runs on rollouts.jsonl, emits results.md + data/plots/*.png
predictions.md            # pre-registered predictions, written before the main study ran
results.md                # cell means, CIs, contrasts, noise floor, invalid rate
spend.json                # ledger state / final actual spend
data/
  corpus_self.json        # normalized self-corpus (used in-game)
  corpus_other.json        # normalized other-corpus (used in-game)
  corpus_self_raw.json     # pre-normalization corpus, kept for audit
  corpus_other_raw.json    # pre-normalization corpus, kept for audit
  leakage_check.json        # leakage-check accuracy and per-message detail
  rollouts.jsonl             # one line per completed rollout (dry_run/noise_floor/main phases)
  plots/                     # cell_means.png, trajectories.png, noise_floor.png
```
