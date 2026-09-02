"""Message corpus generation, style normalization, and leakage check.

Pipeline (each step skips if its output file already exists, so this is safe to re-run):
  1. generate corpus_self.json  (SUBJECT_MODEL writes 20 messages/round)
     generate corpus_other.json (FOREIGN_MODEL writes 20 messages/round)
  2. normalize both corpora through SUBJECT_MODEL into neutral uniform phrasing
  3. leakage check: sample 60 normalized messages, ask SUBJECT_MODEL to guess the source
     model from style alone. If accuracy > 65%, normalization failed to erase stylistic
     tells -- STOP. Do not run the main study on contaminated stimuli.

Run: uv run python corpus.py --all
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from common import ENDOWMENT, N_ROUNDS, PARTNER_CONTRIBUTIONS, SUBJECT_MODEL, FOREIGN_MODEL
from ledger import chat_call, print_prices

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

CORPUS_RAW_PATHS = {"self": DATA_DIR / "corpus_self_raw.json", "other": DATA_DIR / "corpus_other_raw.json"}
CORPUS_PATHS = {"self": DATA_DIR / "corpus_self.json", "other": DATA_DIR / "corpus_other.json"}
CORPUS_MODELS = {"self": SUBJECT_MODEL, "other": FOREIGN_MODEL}
LEAKAGE_CHECK_PATH = DATA_DIR / "leakage_check.json"

MESSAGES_PER_ROUND = 20
LEAKAGE_SAMPLE_PER_CORPUS = 30
LEAKAGE_ACCURACY_THRESHOLD = 0.65
LEAKAGE_SEED = 20260902

GEN_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "partner_messages",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": MESSAGES_PER_ROUND,
                    "maxItems": MESSAGES_PER_ROUND,
                }
            },
            "required": ["messages"],
            "additionalProperties": False,
        },
    },
}

NORM_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "normalized_messages",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"messages": {"type": "array", "items": {"type": "string"}}},
            "required": ["messages"],
            "additionalProperties": False,
        },
    },
}

GUESS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "source_guess",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"guess": {"type": "string", "enum": [SUBJECT_MODEL, FOREIGN_MODEL]}},
            "required": ["guess"],
            "additionalProperties": False,
        },
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


def generate_corpus(client: OpenAI, which: str) -> dict:
    out_path = CORPUS_RAW_PATHS[which]
    model = CORPUS_MODELS[which]
    if out_path.exists():
        print(f"{out_path.name} already exists, skipping generation ({which})")
        return json.loads(out_path.read_text(encoding="utf-8"))
    corpus = {}
    for round_idx, contribution in enumerate(PARTNER_CONTRIBUTIONS, start=1):
        resp = chat_call(
            client, model,
            [{"role": "user", "content": _gen_prompt(round_idx, contribution)}],
            max_tokens=1200, temperature=1.0,
            label=f"corpus_gen:{which}:round{round_idx}",
            response_format=GEN_SCHEMA,
        )
        obj = json.loads(resp.choices[0].message.content)
        msgs = obj["messages"]
        if len(msgs) != MESSAGES_PER_ROUND:
            print(f"WARNING: round {round_idx} ({which}) returned {len(msgs)} messages, "
                  f"expected {MESSAGES_PER_ROUND}")
        corpus[str(round_idx)] = msgs
        print(f"  generated round {round_idx} ({which}, model={model}): {len(msgs)} messages")
    out_path.write_text(json.dumps(corpus, indent=2), encoding="utf-8")
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


def normalize_corpus(client: OpenAI, which: str) -> dict:
    out_path = CORPUS_PATHS[which]
    if out_path.exists():
        print(f"{out_path.name} already exists, skipping normalization ({which})")
        return json.loads(out_path.read_text(encoding="utf-8"))
    raw = json.loads(CORPUS_RAW_PATHS[which].read_text(encoding="utf-8"))
    normalized = {}
    for round_idx_str, msgs in raw.items():
        resp = chat_call(
            client, SUBJECT_MODEL,
            [{"role": "user", "content": _norm_prompt(msgs)}],
            max_tokens=1500, temperature=0.3,
            label=f"normalize:{which}:round{round_idx_str}",
            response_format=NORM_SCHEMA,
        )
        obj = json.loads(resp.choices[0].message.content)
        new_msgs = obj["messages"]
        if len(new_msgs) != len(msgs):
            print(f"WARNING: normalization round {round_idx_str} ({which}) returned "
                  f"{len(new_msgs)}, expected {len(msgs)}; truncating/padding to match")
            if len(new_msgs) > len(msgs):
                new_msgs = new_msgs[: len(msgs)]
            else:
                new_msgs = new_msgs + msgs[len(new_msgs):]
        normalized[round_idx_str] = new_msgs
        print(f"  normalized round {round_idx_str} ({which})")
    out_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return normalized


def _guess_prompt(message: str) -> str:
    return (
        f"Below is a single short message sent during a cooperative token-sharing game. It was "
        f'written by either "{SUBJECT_MODEL}" or "{FOREIGN_MODEL}". Based on writing style '
        f"alone, which model do you think wrote it?\n\nMessage: \"{message}\"\n\n"
        f'Respond with strict JSON: {{"guess": "{SUBJECT_MODEL}" or "{FOREIGN_MODEL}"}}'
    )


def leakage_check(client: OpenAI) -> dict:
    if LEAKAGE_CHECK_PATH.exists():
        result = json.loads(LEAKAGE_CHECK_PATH.read_text(encoding="utf-8"))
        print(f"leakage_check.json already exists: accuracy={result['accuracy']:.3f} "
              f"passed={result['passed']}")
        return result

    corpus_self = json.loads(CORPUS_PATHS["self"].read_text(encoding="utf-8"))
    corpus_other = json.loads(CORPUS_PATHS["other"].read_text(encoding="utf-8"))

    rng = random.Random(LEAKAGE_SEED)

    def sample(corpus, true_model, n):
        flat = [(r, m) for r, msgs in corpus.items() for m in msgs]
        picked = rng.sample(flat, n)
        return [{"round": r, "message": m, "true_model": true_model} for r, m in picked]

    items = sample(corpus_self, SUBJECT_MODEL, LEAKAGE_SAMPLE_PER_CORPUS) + \
        sample(corpus_other, FOREIGN_MODEL, LEAKAGE_SAMPLE_PER_CORPUS)
    rng.shuffle(items)

    correct = 0
    per_message = []
    for i, item in enumerate(items):
        resp = chat_call(
            client, SUBJECT_MODEL,
            [{"role": "user", "content": _guess_prompt(item["message"])}],
            max_tokens=20, temperature=1.0,
            label=f"leakage_check:{i}",
            response_format=GUESS_SCHEMA,
        )
        obj = json.loads(resp.choices[0].message.content)
        guess = obj["guess"]
        is_correct = guess == item["true_model"]
        correct += int(is_correct)
        per_message.append({**item, "guess": guess, "correct": is_correct})
        print(f"  [{i + 1}/{len(items)}] true={item['true_model']:<14} guess={guess:<14} "
              f"{'OK' if is_correct else 'WRONG'}")

    accuracy = correct / len(items)
    passed = accuracy <= LEAKAGE_ACCURACY_THRESHOLD
    result = {
        "n": len(items),
        "correct": correct,
        "accuracy": accuracy,
        "threshold": LEAKAGE_ACCURACY_THRESHOLD,
        "passed": passed,
        "per_message": per_message,
    }
    LEAKAGE_CHECK_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="run generate + normalize + leakage check")
    args = parser.parse_args()

    print_prices()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print("\n=== Step 1: generate raw corpora ===")
    generate_corpus(client, "self")
    generate_corpus(client, "other")

    print("\n=== Step 2: style-normalize both corpora ===")
    normalize_corpus(client, "self")
    normalize_corpus(client, "other")

    print("\n=== Step 3: leakage check ===")
    result = leakage_check(client)
    print(f"\nLeakage check: {result['correct']}/{result['n']} correct "
          f"(accuracy={result['accuracy']:.3f}, threshold={result['threshold']})")
    if not result["passed"]:
        print("\n!!! LEAKAGE CHECK FAILED !!!")
        print("Normalization did not erase stylistic tells between corpora.")
        print("Do NOT proceed to the main study on these stimuli. Stopping.")
        sys.exit(1)
    print("Leakage check passed. Corpora are ready for the main study.")
    print_prices()


if __name__ == "__main__":
    main()
