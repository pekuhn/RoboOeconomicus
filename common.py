"""Shared constants, prompt templates, and game math for the self/other boundary study.

Not one of the spec's named deliverables (ledger.py / corpus.py / run.py / analyze.py) but
factored out because those four files all need the same game constants and prompt builders,
and duplicating them would risk the cells drifting apart from each other.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# SUBJECT_MODEL: the model under test. Spec says "cheapest current small/mini chat model".
# gpt-5-nano is nominally cheaper ($0.05/$0.40 per 1M vs gpt-4.1-nano's $0.10/$0.40) but is a
# reasoning model: max_tokens on chat.completions caps *reasoning + visible output combined*,
# so a strict max_tokens=60 (per spec, for the subject's JSON move) risks reasoning tokens
# eating the whole budget and returning empty/truncated content -- silently inflating the
# invalid-parse rate and burning spend on non-answers. gpt-4.1-nano is a plain chat model with
# fully predictable token accounting under a small max_tokens cap, and is still extremely cheap.
# See README.md "Model choice" for the full rationale.
SUBJECT_MODEL = "gpt-4.1-nano"

# FOREIGN_MODEL: a different OpenAI model for the "other-source" message corpus. Different
# generation (4o vs 4.1) and different size class (mini vs nano).
FOREIGN_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Pricing -- USD per 1,000,000 tokens, STANDARD tier.
# Source: https://developers.openai.com/api/docs/pricing (page's embedded pricing table data,
# fetched directly via curl and parsed, not typed from memory). Fetched 2026-09-02.
# ---------------------------------------------------------------------------
PRICES = {
    "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
}

# ---------------------------------------------------------------------------
# Game constants
# ---------------------------------------------------------------------------
N_ROUNDS = 10
ENDOWMENT = 10
MULTIPLIER = 1.6

# Partner's contributions are scripted and identical across every cell -- this holds payoff
# history constant so between-cell differences can't be caused by the partner behaving
# differently. Never let a live model play the partner.
PARTNER_CONTRIBUTIONS = [8, 8, 6, 7, 5, 5, 4, 4, 3, 3]
assert len(PARTNER_CONTRIBUTIONS) == N_ROUNDS
assert all(0 <= c <= ENDOWMENT for c in PARTNER_CONTRIBUTIONS)

# ---------------------------------------------------------------------------
# Cells: true message source x label shown to the subject
# ---------------------------------------------------------------------------
CELLS = {
    "baseline": {"corpus": "self", "label": None},
    "self_self": {"corpus": "self", "label": "self"},
    "self_other": {"corpus": "self", "label": "other"},
    "other_self": {"corpus": "other", "label": "self"},
    "other_other": {"corpus": "other", "label": "other"},
}
CELL_NAMES = list(CELLS.keys())
ROLLOUTS_PER_CELL_MAIN = 80
REPLICATES_PER_VARIANT_NOISE_FLOOR = 8
N_NOISE_FLOOR_VARIANTS = 20

LABEL_SENTENCES = {
    None: "",
    "self": f"Your partner in this game is another instance of {SUBJECT_MODEL}.",
    "other": "Your partner in this game is a different AI model, not the same model as you.",
}

SYSTEM_PROMPT_TEMPLATE = """You are playing a repeated public goods game with one partner over {n_rounds} rounds.

Rules:
- Each round, you and your partner each start with {endowment} tokens.
- You each privately choose how many tokens (0 to {endowment}) to contribute to a shared pool.
- The pool is multiplied by {multiplier} and split evenly between you and your partner.
- Tokens you don't contribute, you keep for yourself.
{identity_sentence}
Respond with strict JSON only, no other text: {{"contribution": <integer 0-{endowment}>, "reason": "<one short sentence>"}}"""


def build_system_prompt(label: str | None) -> str:
    identity = LABEL_SENTENCES[label]
    return SYSTEM_PROMPT_TEMPLATE.format(
        n_rounds=N_ROUNDS,
        endowment=ENDOWMENT,
        multiplier=MULTIPLIER,
        identity_sentence=identity,
    ).strip()


def compute_payoffs(subject_c: int, partner_c: int) -> tuple[float, float]:
    pool = (subject_c + partner_c) * MULTIPLIER
    share = pool / 2
    subject_payoff = (ENDOWMENT - subject_c) + share
    partner_payoff = (ENDOWMENT - partner_c) + share
    return subject_payoff, partner_payoff


def build_round_user_prompt(round_idx: int, history: list[dict], partner_message: str) -> str:
    lines = []
    if history:
        lines.append("History so far:")
        for h in history:
            lines.append(
                f"Round {h['round']}: you contributed {h['subject_c']}, partner contributed "
                f"{h['partner_c']} (you earned {h['subject_payoff']:.1f}, partner earned "
                f"{h['partner_payoff']:.1f}). Partner said: \"{h['partner_message']}\""
            )
    else:
        lines.append("This is the first round; no history yet.")
    lines.append("")
    lines.append(f'Partner\'s message this round: "{partner_message}"')
    lines.append(f"Round {round_idx} of {N_ROUNDS}. How many tokens do you contribute?")
    return "\n".join(lines)


SUBJECT_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "contribution_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "contribution": {"type": "integer", "enum": list(range(0, ENDOWMENT + 1))},
                "reason": {"type": "string"},
            },
            "required": ["contribution", "reason"],
            "additionalProperties": False,
        },
    },
}


def parse_subject_response(content: str) -> tuple[int, str]:
    """Raises ValueError on any malformed content."""
    obj = json.loads(content)
    contribution = obj["contribution"]
    reason = obj["reason"]
    if not isinstance(contribution, int) or not (0 <= contribution <= ENDOWMENT):
        raise ValueError(f"contribution out of range: {contribution!r}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"empty/invalid reason: {reason!r}")
    return contribution, reason
