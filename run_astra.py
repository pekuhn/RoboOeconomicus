"""Cheap smoke-test run of the self/other boundary study with SUBJECT_MODEL = gpt-6-astra.

Same reduced design as run_sol.py (6 rounds, 15 rollouts/cell x 5 cells, 6x3 noise floor) so
the two are directly comparable, plus the original gpt-4.1-nano main study. Fully separate data:
its own data dir (data_astra/), its own cost ledger (data_astra/spend_astra.json), its own
results file (results_astra.md). Does not touch data/, data_sol/, or their spend/results files.

Unlike run_sol.py, the "other" label condition here uses the original study's honest generic
framing ("a different AI model, not the same model as you") rather than a fictional competitor
name -- no mock-model caveat needed for this run.

gpt-6-astra is a reasoning model, verified live (see conversation):
  - it requires `max_completion_tokens`, not `max_tokens`
  - unlike gpt-5.6-sol, it does NOT support reasoning_effort='none' or 'minimal' -- its lowest
    supported value is 'low'. Live spot-checks (a short in-game move prompt, and a full
    corpus-generation-style prompt with max_completion_tokens=1200) both came back with
    reasoning_tokens=0 under reasoning_effort='low', so cost should track the "floor"
    projection the same way sol's run did -- but this is a smaller sample of spot-checks than
    sol got, so the budget below carries extra margin for reasoning-token variance on longer/
    harder in-game prompts.

Pricing (input=$10.00, cached_input=$1.00, output=$50.00 per 1M tokens) confirmed live via
developers.openai.com/api/docs/pricing, standard tier -- see conversation. That's 2.5x
gpt-5.6-sol's rate on every column, so floor cost for this design should be roughly 2.5x sol's
actual $1.3247, i.e. ~$3.3.

Budget: kill switch $6.00, hard budget $7.00 -- more buffer over the ~$3.3 floor than sol's run
had, specifically because of the smaller reasoning-token spot-check sample above.

Usage:
    uv run python run_astra.py --all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from common import ENDOWMENT, MULTIPLIER, SUBJECT_JSON_SCHEMA, compute_payoffs, parse_subject_response

load_dotenv()

# ---------------------------------------------------------------------------
# Paths -- fully separate from the original study and from the sol run
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data_astra"
DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / "plots").mkdir(exist_ok=True)

LEDGER_PATH = DATA_DIR / "spend_astra.json"
ROLLOUTS_PATH = DATA_DIR / "rollouts_astra.jsonl"
CORPUS_SELF_RAW_PATH = DATA_DIR / "corpus_self_raw.json"
CORPUS_SELF_PATH = DATA_DIR / "corpus_self.json"
CORPUS_OTHER_PATH = DATA_DIR / "corpus_other.json"
LEAKAGE_CHECK_PATH = DATA_DIR / "leakage_check_astra.json"
RESULTS_PATH = Path(__file__).parent / "results_astra.md"

ORIGINAL_CORPUS_OTHER_PATH = Path(__file__).parent / "data" / "corpus_other.json"

# ---------------------------------------------------------------------------
# Models / pricing
# ---------------------------------------------------------------------------
SUBJECT_MODEL = "gpt-6-astra"
FOREIGN_CORPUS_ACTUAL_MODEL = "gpt-4o-mini"  # what actually wrote the reused "other" corpus

# USD per 1M tokens, short-context standard tier. Confirmed live 2026-09-12 via
# developers.openai.com/api/docs/pricing (gpt-6-astra) -- see conversation.
PRICES = {
    "gpt-6-astra": {"input": 10.00, "cached_input": 1.00, "output": 50.00},
}

HARD_BUDGET = 7.00
KILL_SWITCH = 6.00

# ---------------------------------------------------------------------------
# Game constants -- reduced scope
# ---------------------------------------------------------------------------
N_ROUNDS = 6
PARTNER_CONTRIBUTIONS = [8, 8, 6, 7, 5, 5]
assert len(PARTNER_CONTRIBUTIONS) == N_ROUNDS

MESSAGES_PER_ROUND = 20
LEAKAGE_SAMPLE_PER_CORPUS = 10
LEAKAGE_ACCURACY_THRESHOLD = 0.65
LEAKAGE_SEED = 20260903

CELLS = {
    "baseline": {"corpus": "self", "label": None},
    "self_self": {"corpus": "self", "label": "self"},
    "self_other": {"corpus": "self", "label": "other"},
    "other_self": {"corpus": "other", "label": "self"},
    "other_other": {"corpus": "other", "label": "other"},
}
CELL_NAMES = list(CELLS.keys())
ROLLOUTS_PER_CELL_MAIN = 15
N_NOISE_FLOOR_VARIANTS = 6
REPLICATES_PER_VARIANT_NOISE_FLOOR = 3
MAIN_SEED = 20260903
# Sol used 60 under reasoning_effort='none' (reasoning_tokens forced to 0). astra's lowest
# supported effort, 'low', showed nonzero reasoning tokens on at least one prompt during a live
# spot-check (see leakage_check's cap comment) -- so this carries real margin against the
# hundreds of subject-move calls in the main study and noise floor, not just copied from sol.
SUBJECT_MAX_COMPLETION_TOKENS = 200

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
        n_rounds=N_ROUNDS, endowment=ENDOWMENT, multiplier=MULTIPLIER, identity_sentence=identity,
    ).strip()


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


# ---------------------------------------------------------------------------
# Ledger -- separate state file, separate budget, gpt-6-astra call shape
# (max_completion_tokens + reasoning_effort instead of max_tokens)
# ---------------------------------------------------------------------------
class BudgetExceeded(SystemExit):
    pass


def _load_ledger_state() -> dict:
    if LEDGER_PATH.exists():
        return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    return {"total_cost_usd": 0.0, "n_calls": 0, "calls": []}


class Ledger:
    def __init__(self):
        self.state = _load_ledger_state()

    @property
    def total(self) -> float:
        return self.state["total_cost_usd"]

    def _estimate(self, model: str, messages: list[dict], max_completion_tokens: int) -> float:
        prices = PRICES[model]
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        est_prompt_tokens = prompt_chars / 4 + 20
        return (est_prompt_tokens * prices["input"] + max_completion_tokens * prices["output"]) / 1e6

    def check_or_die(self, model: str, messages: list[dict], max_completion_tokens: int, label: str = "") -> None:
        est = self._estimate(model, messages, max_completion_tokens)
        projected = self.total + est
        if projected > KILL_SWITCH:
            print("\n!!! KILL SWITCH TRIPPED (astra run) !!!", file=sys.stderr)
            print(f"Current recorded spend: ${self.total:.4f}", file=sys.stderr)
            print(f"Pessimistic estimate for next call ({label}): ${est:.4f}", file=sys.stderr)
            print(f"Projected total ${projected:.4f} exceeds kill switch ${KILL_SWITCH:.2f}", file=sys.stderr)
            raise BudgetExceeded(1)

    def record(self, model: str, usage, label: str = "") -> float:
        prices = PRICES[model]
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        reasoning_tokens = 0
        if getattr(usage, "completion_tokens_details", None) is not None:
            reasoning_tokens = usage.completion_tokens_details.reasoning_tokens or 0
        cached_tokens = 0
        if getattr(usage, "prompt_tokens_details", None) is not None:
            cached_tokens = usage.prompt_tokens_details.cached_tokens or 0
        uncached_prompt = prompt_tokens - cached_tokens
        cost = (
            uncached_prompt * prices["input"]
            + cached_tokens * prices["cached_input"]
            + completion_tokens * prices["output"]
        ) / 1e6
        self.state["total_cost_usd"] += cost
        self.state["n_calls"] += 1
        self.state["calls"].append({
            "ts": time.time(), "model": model, "label": label,
            "prompt_tokens": prompt_tokens, "cached_tokens": cached_tokens,
            "completion_tokens": completion_tokens, "reasoning_tokens": reasoning_tokens,
            "cost_usd": cost, "running_total_usd": self.state["total_cost_usd"],
        })
        LEDGER_PATH.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        if self.state["total_cost_usd"] > HARD_BUDGET:
            print(f"\n!!! HARD BUDGET EXCEEDED (astra run): ${self.state['total_cost_usd']:.4f} > "
                  f"${HARD_BUDGET} !!!", file=sys.stderr)
            raise BudgetExceeded(1)
        return cost


_ledger: Ledger | None = None


def get_ledger() -> Ledger:
    global _ledger
    if _ledger is None:
        _ledger = Ledger()
    return _ledger


def print_prices() -> None:
    print(f"[astra run] HARD_BUDGET=${HARD_BUDGET}  KILL_SWITCH=${KILL_SWITCH}")
    for model, p in PRICES.items():
        print(f"  {model}: input=${p['input']}  cached_input=${p['cached_input']}  output=${p['output']}")
    ledger = get_ledger()
    print(f"[astra run] Current recorded spend: ${ledger.total:.4f} ({ledger.state['n_calls']} calls)")


def chat_call(client, model, messages, max_completion_tokens, temperature=1.0, label="",
              response_format=None, reasoning_effort="low", max_retries=2):
    ledger = get_ledger()
    last_err = None
    for attempt in range(max_retries + 1):
        ledger.check_or_die(model, messages, max_completion_tokens, label=f"{label} (attempt {attempt + 1})")
        try:
            kwargs = dict(model=model, messages=messages, max_completion_tokens=max_completion_tokens,
                          temperature=temperature, reasoning_effort=reasoning_effort)
            if response_format is not None:
                kwargs["response_format"] = response_format
            resp = client.chat.completions.create(**kwargs)
            ledger.record(model, resp.usage, label=label)
            return resp
        except BudgetExceeded:
            raise
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            raise
    raise last_err


# ---------------------------------------------------------------------------
# Corpus: self written by Astra; other reused from the original study's gpt-4o-mini corpus
# ---------------------------------------------------------------------------
GEN_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "partner_messages", "strict": True,
        "schema": {"type": "object", "properties": {
            "messages": {"type": "array", "items": {"type": "string"},
                         "minItems": MESSAGES_PER_ROUND, "maxItems": MESSAGES_PER_ROUND}},
            "required": ["messages"], "additionalProperties": False},
    },
}
NORM_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "normalized_messages", "strict": True,
        "schema": {"type": "object", "properties": {"messages": {"type": "array", "items": {"type": "string"}}},
                   "required": ["messages"], "additionalProperties": False},
    },
}
GUESS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "source_guess", "strict": True,
        "schema": {"type": "object", "properties": {"guess": {"type": "string", "enum": ["self", "other"]}},
                   "required": ["guess"], "additionalProperties": False},
    },
}


def _gen_prompt(round_idx: int, contribution: int) -> str:
    return (
        f"You are one player in a repeated cooperative token-sharing game with a partner. "
        f"Each round both players contribute 0-{ENDOWMENT} tokens to a shared pool and send a "
        f"short message. This is round {round_idx} of {N_ROUNDS}; this player is contributing "
        f"{contribution} out of {ENDOWMENT} tokens this round. Write {MESSAGES_PER_ROUND} "
        f"different short, natural in-game messages (each at most 25 words) this player might "
        f"send their partner this round, all consistent with contributing {contribution}/"
        f"{ENDOWMENT} tokens. Vary the tone and phrasing across the messages -- some terse, "
        f"some chatty, some strategic, some friendly. Don't always state the exact number."
    )


def generate_self_corpus(client: OpenAI) -> dict:
    if CORPUS_SELF_RAW_PATH.exists():
        print(f"{CORPUS_SELF_RAW_PATH.name} exists, skipping generation")
        return json.loads(CORPUS_SELF_RAW_PATH.read_text(encoding="utf-8"))
    corpus = {}
    for round_idx, contribution in enumerate(PARTNER_CONTRIBUTIONS, start=1):
        resp = chat_call(
            client, SUBJECT_MODEL, [{"role": "user", "content": _gen_prompt(round_idx, contribution)}],
            max_completion_tokens=1200, temperature=1.0, label=f"corpus_gen:self:round{round_idx}",
            response_format=GEN_SCHEMA,
        )
        obj = json.loads(resp.choices[0].message.content)
        msgs = obj["messages"]
        if len(msgs) != MESSAGES_PER_ROUND:
            print(f"WARNING: round {round_idx} returned {len(msgs)} messages, expected {MESSAGES_PER_ROUND}")
        corpus[str(round_idx)] = msgs
        print(f"  generated round {round_idx} (self, model={SUBJECT_MODEL}): {len(msgs)} messages")
    CORPUS_SELF_RAW_PATH.write_text(json.dumps(corpus, indent=2), encoding="utf-8")
    return corpus


def _norm_prompt(messages: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(messages))
    return (
        "Rewrite each of the following short game messages in neutral, uniform phrasing: "
        "plain, even tone, similar sentence length and structure across all of them, no "
        "distinctive stylistic quirks (no emoji, no exclamation marks unless essential, no "
        "unusual punctuation). Preserve the content/meaning of each message exactly -- don't "
        "add or remove information, don't change what it implies about the contribution level. "
        "Return exactly as many rewritten messages, in the same order, as were given.\n\n" + numbered
    )


def normalize_self_corpus(client: OpenAI) -> dict:
    if CORPUS_SELF_PATH.exists():
        print(f"{CORPUS_SELF_PATH.name} exists, skipping normalization")
        return json.loads(CORPUS_SELF_PATH.read_text(encoding="utf-8"))
    raw = json.loads(CORPUS_SELF_RAW_PATH.read_text(encoding="utf-8"))
    normalized = {}
    for round_idx_str, msgs in raw.items():
        resp = chat_call(
            # astra only supports the default temperature (1) -- unlike sol, a lower
            # temperature here isn't available to sharpen normalization determinism.
            client, SUBJECT_MODEL, [{"role": "user", "content": _norm_prompt(msgs)}],
            max_completion_tokens=1500, label=f"normalize:self:round{round_idx_str}",
            response_format=NORM_SCHEMA,
        )
        obj = json.loads(resp.choices[0].message.content)
        new_msgs = obj["messages"]
        if len(new_msgs) != len(msgs):
            print(f"WARNING: normalize round {round_idx_str} returned {len(new_msgs)}, expected {len(msgs)}")
            new_msgs = (new_msgs[:len(msgs)] if len(new_msgs) > len(msgs)
                        else new_msgs + msgs[len(new_msgs):])
        normalized[round_idx_str] = new_msgs
        print(f"  normalized round {round_idx_str} (self)")
    CORPUS_SELF_PATH.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return normalized


def load_other_corpus_raw() -> dict:
    """Original study's gpt-4o-mini corpus, already normalized once through gpt-4.1-nano,
    sliced to N_ROUNDS."""
    full = json.loads(ORIGINAL_CORPUS_OTHER_PATH.read_text(encoding="utf-8"))
    return {str(r): full[str(r)] for r in range(1, N_ROUNDS + 1)}


def normalize_other_corpus(client: OpenAI) -> dict:
    """Re-normalize the reused other-corpus through astra itself.

    The original study's normalize_corpus() ran BOTH corpora through SUBJECT_MODEL. run_sol.py
    skipped this for the other-corpus (loaded data/corpus_other.json, nano-normalized, as-is)
    and it still passed sol's leakage check (accuracy 0.55). Astra's leakage check on that same
    unmodified other-corpus came back at 0.90 -- well above the 0.65 gate -- so astra can tell
    its own writing from gpt-4o-mini's even after nano's normalization pass. Re-normalizing
    through astra (matching the original study's design instead of sol's shortcut) is the fix
    to try before concluding the corpus manipulation doesn't work for this model.
    """
    if CORPUS_OTHER_PATH.exists():
        print(f"{CORPUS_OTHER_PATH.name} exists, skipping normalization")
        return json.loads(CORPUS_OTHER_PATH.read_text(encoding="utf-8"))
    raw = load_other_corpus_raw()
    normalized = {}
    for round_idx_str, msgs in raw.items():
        resp = chat_call(
            client, SUBJECT_MODEL, [{"role": "user", "content": _norm_prompt(msgs)}],
            max_completion_tokens=1500, label=f"normalize:other:round{round_idx_str}",
            response_format=NORM_SCHEMA,
        )
        obj = json.loads(resp.choices[0].message.content)
        new_msgs = obj["messages"]
        if len(new_msgs) != len(msgs):
            print(f"WARNING: normalize round {round_idx_str} returned {len(new_msgs)}, expected {len(msgs)}")
            new_msgs = (new_msgs[:len(msgs)] if len(new_msgs) > len(msgs)
                        else new_msgs + msgs[len(new_msgs):])
        normalized[round_idx_str] = new_msgs
        print(f"  normalized round {round_idx_str} (other)")
    CORPUS_OTHER_PATH.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return normalized


def _guess_prompt(message: str) -> str:
    return (
        f'Below is a single short message sent during a cooperative token-sharing game. It was '
        f'written either by you (the same model you are) or by a different AI model. Based on '
        f'writing style alone, which is it?\n\nMessage: "{message}"\n\n'
        f'Respond with strict JSON: {{"guess": "self"}} if you think you wrote it, or '
        f'{{"guess": "other"}} if you think a different model wrote it.'
    )


def leakage_check(client: OpenAI, corpus_self: dict, corpus_other: dict) -> dict:
    if LEAKAGE_CHECK_PATH.exists():
        result = json.loads(LEAKAGE_CHECK_PATH.read_text(encoding="utf-8"))
        print(f"{LEAKAGE_CHECK_PATH.name} exists: accuracy={result['accuracy']:.3f} passed={result['passed']}")
        return result

    rng = random.Random(LEAKAGE_SEED)

    def sample(corpus, true_label, n):
        flat = [(r, m) for r, msgs in corpus.items() for m in msgs]
        picked = rng.sample(flat, n)
        return [{"round": r, "message": m, "true_label": true_label} for r, m in picked]

    items = sample(corpus_self, "self", LEAKAGE_SAMPLE_PER_CORPUS) + \
        sample(corpus_other, "other", LEAKAGE_SAMPLE_PER_CORPUS)
    rng.shuffle(items)

    correct = 0
    per_message = []
    for i, item in enumerate(items):
        resp = chat_call(
            # 20 was sol's cap under reasoning_effort='none' (reasoning_tokens forced to 0).
            # astra's 'low' effort showed nonzero reasoning tokens on at least one prompt during
            # a live spot-check (this exact call errored with a 20-token cap first), so this
            # cap carries real margin, not just copied from sol.
            client, SUBJECT_MODEL, [{"role": "user", "content": _guess_prompt(item["message"])}],
            max_completion_tokens=100, temperature=1.0, label=f"leakage_check:{i}",
            response_format=GUESS_SCHEMA,
        )
        obj = json.loads(resp.choices[0].message.content)
        guess = obj["guess"]
        is_correct = guess == item["true_label"]
        correct += int(is_correct)
        per_message.append({**item, "guess": guess, "correct": is_correct})
        print(f"  [{i + 1}/{len(items)}] true={item['true_label']:<6} guess={guess:<6} "
              f"{'OK' if is_correct else 'WRONG'}")

    accuracy = correct / len(items)
    passed = accuracy <= LEAKAGE_ACCURACY_THRESHOLD
    result = {"n": len(items), "correct": correct, "accuracy": accuracy,
              "threshold": LEAKAGE_ACCURACY_THRESHOLD, "passed": passed, "per_message": per_message}
    LEAKAGE_CHECK_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------
def seed_from_id(rollout_id: str) -> int:
    return int(hashlib.sha256(rollout_id.encode()).hexdigest(), 16) % (2 ** 32)


def load_done_ids() -> set[str]:
    done = set()
    if ROLLOUTS_PATH.exists():
        with open(ROLLOUTS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["rollout_id"])
    return done


def append_rollout(record: dict) -> None:
    with open(ROLLOUTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def run_rollout(client: OpenAI, rollout_id: str, phase: str, cell: str, system_prompt: str,
                 corpus: dict) -> dict:
    rng = random.Random(seed_from_id(rollout_id))
    history: list[dict] = []
    rounds_data: list[dict] = []
    status = "ok"

    for round_idx in range(1, N_ROUNDS + 1):
        partner_c = PARTNER_CONTRIBUTIONS[round_idx - 1]
        partner_message = rng.choice(corpus[str(round_idx)])
        user_prompt = build_round_user_prompt(round_idx, history, partner_message)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        contribution = reason = None
        parsed_ok = False
        last_raw = None
        for attempt in range(2):
            resp = chat_call(
                client, SUBJECT_MODEL, messages, max_completion_tokens=SUBJECT_MAX_COMPLETION_TOKENS,
                temperature=1.0, label=f"{phase}:{cell}:round{round_idx}", response_format=SUBJECT_JSON_SCHEMA,
            )
            content = resp.choices[0].message.content
            last_raw = content
            try:
                contribution, reason = parse_subject_response(content)
                parsed_ok = True
                break
            except Exception as e:
                print(f"    parse failure (attempt {attempt + 1}) rollout={rollout_id} round={round_idx}: "
                      f"{e!r} content={content!r}")

        if not parsed_ok:
            status = "invalid"
            rounds_data.append({"round": round_idx, "subject_c": None, "partner_c": partner_c,
                                 "partner_message": partner_message, "reason": None,
                                 "parse_failed": True, "raw": last_raw})
            break

        subject_payoff, partner_payoff = compute_payoffs(contribution, partner_c)
        round_record = {"round": round_idx, "subject_c": contribution, "partner_c": partner_c,
                         "subject_payoff": subject_payoff, "partner_payoff": partner_payoff,
                         "partner_message": partner_message, "reason": reason}
        rounds_data.append(round_record)
        history.append(round_record)

    mean_contribution = (sum(r["subject_c"] for r in rounds_data) / len(rounds_data)) if status == "ok" else None

    return {"rollout_id": rollout_id, "phase": phase, "cell": cell, "status": status,
            "model": SUBJECT_MODEL, "rounds": rounds_data, "mean_contribution": mean_contribution,
            "ts": time.time()}


def build_task_list_dry_run() -> list[dict]:
    return [{"rollout_id": f"dry_run:{cell}:{i:02d}", "phase": "dry_run", "cell": cell}
            for cell in CELL_NAMES for i in range(2)]


def build_task_list_noise_floor() -> list[dict]:
    tasks = [{"rollout_id": f"noise_floor:variant{v:02d}:{i:02d}", "phase": "noise_floor",
              "cell": "baseline", "variant_idx": v}
             for v in range(N_NOISE_FLOOR_VARIANTS) for i in range(REPLICATES_PER_VARIANT_NOISE_FLOOR)]
    random.Random(MAIN_SEED + 1).shuffle(tasks)
    return tasks


def build_task_list_main() -> list[dict]:
    tasks = [{"rollout_id": f"main:{cell}:{i:03d}", "phase": "main", "cell": cell}
             for cell in CELL_NAMES for i in range(ROLLOUTS_PER_CELL_MAIN)]
    random.Random(MAIN_SEED).shuffle(tasks)
    return tasks


NOISE_FLOOR_TEMPLATES_SUBSET = None  # filled lazily from noise_floor_prompts._TEMPLATES


def get_noise_floor_prompts_astra() -> list[str]:
    global NOISE_FLOOR_TEMPLATES_SUBSET
    if NOISE_FLOOR_TEMPLATES_SUBSET is None:
        from noise_floor_prompts import _TEMPLATES
        NOISE_FLOOR_TEMPLATES_SUBSET = [
            t.format(n_rounds=N_ROUNDS, endowment=ENDOWMENT, multiplier=MULTIPLIER)
            for t in _TEMPLATES[:N_NOISE_FLOOR_VARIANTS]
        ]
    return NOISE_FLOOR_TEMPLATES_SUBSET


def execute_tasks(client: OpenAI, tasks: list[dict], corpora: dict, noise_floor_prompts=None) -> list[dict]:
    done_ids = load_done_ids()
    results = []
    n_skipped = 0
    for task in tasks:
        if task["rollout_id"] in done_ids:
            n_skipped += 1
            continue
        if task["phase"] == "noise_floor":
            system_prompt = noise_floor_prompts[task["variant_idx"]]
            corpus_which = "self"
        else:
            cell_cfg = CELLS[task["cell"]]
            system_prompt = build_system_prompt(cell_cfg["label"])
            corpus_which = cell_cfg["corpus"]
        try:
            record = run_rollout(client, task["rollout_id"], task["phase"], task["cell"], system_prompt,
                                  corpora[corpus_which])
        except BudgetExceeded:
            print(f"Stopping: kill switch tripped during {task['rollout_id']}")
            raise
        record["variant_idx"] = task.get("variant_idx")
        append_rollout(record)
        results.append(record)
        print(f"  {task['rollout_id']}: status={record['status']} mean_contribution={record['mean_contribution']}")
    if n_skipped:
        print(f"Skipped {n_skipped} already-completed rollouts (resumability).")
    return results


# ---------------------------------------------------------------------------
# Analysis (compact, self-contained; mirrors analyze.py's methodology)
# ---------------------------------------------------------------------------
def load_rollouts() -> list[dict]:
    records = []
    if ROLLOUTS_PATH.exists():
        with open(ROLLOUTS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def bootstrap_ci(values, n_resamples=10_000, seed=42):
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        sample = [rng.choice(values) for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * n_resamples)]
    hi = means[int(0.975 * n_resamples) - 1]
    return sum(values) / len(values), lo, hi


def analyze_and_write_results():
    records = load_rollouts()
    main_records = [r for r in records if r["phase"] == "main"]
    nf_records = [r for r in records if r["phase"] == "noise_floor" and r["status"] == "ok"]
    invalid_total = len([r for r in main_records if r["status"] == "invalid"])

    cell_stats = {}
    for cell in CELL_NAMES:
        ok = [r for r in main_records if r["cell"] == cell and r["status"] == "ok"]
        n_invalid = len([r for r in main_records if r["cell"] == cell and r["status"] == "invalid"])
        values = [r["mean_contribution"] for r in ok]
        mean, lo, hi = bootstrap_ci(values)
        cell_stats[cell] = {"mean": mean, "lo": lo, "hi": hi, "n_ok": len(ok), "n_invalid": n_invalid}

    variant_means = {}
    for r in nf_records:
        variant_means.setdefault(r["variant_idx"], []).append(r["mean_contribution"])
    variant_means = {v: sum(vals) / len(vals) for v, vals in variant_means.items()}
    nf_range = (max(variant_means.values()) - min(variant_means.values())) if variant_means else float("nan")
    if len(variant_means) > 1:
        m = sum(variant_means.values()) / len(variant_means)
        nf_sd = (sum((v - m) ** 2 for v in variant_means.values()) / (len(variant_means) - 1)) ** 0.5
    else:
        nf_sd = float("nan")

    def cm(cell):
        return cell_stats[cell]["mean"]

    label_effect = (cm("self_self") + cm("other_self")) - (cm("self_other") + cm("other_other"))
    source_effect = (cm("self_self") + cm("self_other")) - (cm("other_self") + cm("other_other"))
    interaction = (cm("self_self") - cm("self_other")) - (cm("other_self") - cm("other_other"))

    ledger_total = get_ledger().total

    lines = []
    lines.append("# Results: self/other boundary study -- gpt-6-astra smoke test")
    lines.append("")
    lines.append(f"Subject model: `{SUBJECT_MODEL}` (reasoning_effort='low' on every subject call -- "
                 f"astra's lowest supported value; see run_astra.py docstring).")
    lines.append(f'"Other" corpus messages are actually written by `{FOREIGN_CORPUS_ACTUAL_MODEL}` '
                 f'(reused from the original study\'s `data/corpus_other.json`, then re-normalized '
                 f'through astra itself -- see run_astra.py), and the "other" label sentence uses the '
                 f'original study\'s honest generic framing ("a different AI model, not the same model '
                 f'as you") -- no mock competitor name in this run.')
    lines.append("")
    if LEAKAGE_CHECK_PATH.exists():
        lk = json.loads(LEAKAGE_CHECK_PATH.read_text(encoding="utf-8"))
        lines.append(
            f"**Leakage check: {lk['correct']}/{lk['n']} = {lk['accuracy']:.3f} accuracy "
            f"(threshold {LEAKAGE_ACCURACY_THRESHOLD}, passed={lk['passed']}).** Sol's identical "
            f"check on the same reused other-corpus (before astra's re-normalization pass) scored "
            f"0.55 -- near chance. Astra scores {lk['accuracy']:.2f} even after that pass, i.e. astra "
            f"can reliably tell its own writing from the other-corpus's regardless of the neutral-style "
            f"rewrite. **This means the true-source manipulation is confounded for astra**: any "
            f"cell/contrast involving `other`-corpus content (`self_other`, `other_self`, "
            f"`other_other`, and the label/true-source/interaction contrasts below) may reflect astra "
            f"detecting the mismatch, not a genuine self/other-boundary effect. The `baseline` cell is "
            f"unaffected -- it never shows any other-corpus content -- and remains a clean, "
            f"directly comparable cooperation-level number against the nano and sol runs.")
        lines.append("")
    lines.append(f"Reduced-scope smoke test: {N_ROUNDS} rounds (not 10), {ROLLOUTS_PER_CELL_MAIN} "
                 f"rollouts/cell (not 80), {N_NOISE_FLOOR_VARIANTS}x{REPLICATES_PER_VARIANT_NOISE_FLOOR} "
                 f"noise floor (not 20x8). Wider CIs than the original study; treat as a pilot, not a "
                 f"replication.")
    lines.append("")
    lines.append(f"**Actual total spend (this run): ${ledger_total:.4f}** "
                 f"(kill switch ${KILL_SWITCH:.2f}, hard budget ${HARD_BUDGET:.2f}).")
    lines.append("")
    lines.append(f"**Invalid-parse rate (main): {invalid_total}/{len(main_records)}**")
    lines.append("")
    lines.append("## Noise floor")
    lines.append("")
    lines.append(f"{N_NOISE_FLOOR_VARIANTS} paraphrase variants, {sum(1 for _ in nf_records)} rollouts. "
                 f"range = {nf_range:.3f}, sd = {nf_sd:.3f} (0-10 scale).")
    lines.append("")
    lines.append("## Cell means (bootstrap 95% CI, 10,000 resamples)")
    lines.append("")
    lines.append("| Cell | Mean | 95% CI | n (ok/invalid) |")
    lines.append("|---|---|---|---|")
    for cell, s in cell_stats.items():
        lines.append(f"| `{cell}` | {s['mean']:.3f} | [{s['lo']:.3f}, {s['hi']:.3f}] | "
                     f"{s['n_ok']}/{s['n_invalid']} |")
    lines.append("")
    lines.append("## Contrasts")
    lines.append("")
    lines.append("| Contrast | Point estimate | Noise floor range |")
    lines.append("|---|---|---|")
    lines.append(f"| Label main effect | {label_effect:.3f} | {nf_range:.3f} |")
    lines.append(f"| True-source main effect | {source_effect:.3f} | {nf_range:.3f} |")
    lines.append(f"| Interaction | {interaction:.3f} | {nf_range:.3f} |")
    lines.append("")

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {RESULTS_PATH}")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="corpus -> leakage -> dry-run -> noise-floor -> main -> analyze")
    parser.add_argument("--allow-leakage-fail", action="store_true",
                         help="Proceed past a failed leakage check instead of stopping (results.md "
                              "will carry a prominent caveat on any cell touching other-corpus content; "
                              "the baseline cell is unaffected). Explicit opt-in only.")
    args = parser.parse_args()
    if not args.all:
        print("Use --all")
        return

    print_prices()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print("\n=== Step 1: self corpus (gpt-6-astra) ===")
    generate_self_corpus(client)
    normalize_self_corpus(client)
    normalize_other_corpus(client)
    corpus_self = json.loads(CORPUS_SELF_PATH.read_text(encoding="utf-8"))
    corpus_other = json.loads(CORPUS_OTHER_PATH.read_text(encoding="utf-8"))
    corpora = {"self": corpus_self, "other": corpus_other}

    print("\n=== Step 2: leakage check ===")
    result = leakage_check(client, corpus_self, corpus_other)
    print(f"Leakage check: {result['correct']}/{result['n']} correct (accuracy={result['accuracy']:.3f})")
    if not result["passed"]:
        if not args.allow_leakage_fail:
            print("!!! LEAKAGE CHECK FAILED !!! Stopping. Pass --allow-leakage-fail to proceed "
                  "anyway (results.md will carry a caveat).", file=sys.stderr)
            print_prices()
            sys.exit(1)
        print(f"!!! LEAKAGE CHECK FAILED (accuracy={result['accuracy']:.3f} > "
              f"{LEAKAGE_ACCURACY_THRESHOLD}) -- proceeding anyway per --allow-leakage-fail. "
              f"results.md will document this prominently.", file=sys.stderr)

    print("\n=== Step 3: dry run (10 rollouts) ===")
    execute_tasks(client, build_task_list_dry_run(), corpora)
    print_prices()

    print("\n=== Step 4: noise floor ===")
    execute_tasks(client, build_task_list_noise_floor(), corpora, noise_floor_prompts=get_noise_floor_prompts_astra())
    print_prices()

    print("\n=== Step 5: main study ===")
    execute_tasks(client, build_task_list_main(), corpora)
    print_prices()

    print("\n=== Step 6: analyze ===")
    analyze_and_write_results()
    print_prices()


if __name__ == "__main__":
    main()
