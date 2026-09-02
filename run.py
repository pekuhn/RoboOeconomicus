"""Run rollouts for the self/other boundary study.

Order of operations (per spec): ledger -> corpus.py --all -> --noise-floor -> --dry-run -> --main.
corpus.py must have been run successfully (data/leakage_check.json exists and passed) before
any of the modes below will run.

Usage:
    uv run python run.py --dry-run
    uv run python run.py --noise-floor
    uv run python run.py --main
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

from common import (
    CELLS, CELL_NAMES, N_ROUNDS, ROLLOUTS_PER_CELL_MAIN, REPLICATES_PER_VARIANT_NOISE_FLOOR,
    N_NOISE_FLOOR_VARIANTS, PARTNER_CONTRIBUTIONS, SUBJECT_MODEL, PRICES,
    build_system_prompt, build_round_user_prompt, compute_payoffs, parse_subject_response,
    SUBJECT_JSON_SCHEMA,
)
from ledger import chat_call, print_prices, get_ledger, BudgetExceeded
from noise_floor_prompts import get_noise_floor_prompts
from corpus import CORPUS_PATHS, LEAKAGE_CHECK_PATH

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
ROLLOUTS_PATH = DATA_DIR / "rollouts.jsonl"

SUBJECT_MAX_TOKENS = 60
MAIN_SEED = 20260902


def require_corpus_ready() -> None:
    if not CORPUS_PATHS["self"].exists() or not CORPUS_PATHS["other"].exists():
        print("Corpora not found. Run `uv run python corpus.py --all` first.", file=sys.stderr)
        sys.exit(1)
    if not LEAKAGE_CHECK_PATH.exists():
        print("Leakage check not found. Run `uv run python corpus.py --all` first.", file=sys.stderr)
        sys.exit(1)
    result = json.loads(LEAKAGE_CHECK_PATH.read_text(encoding="utf-8"))
    if not result["passed"]:
        print("Leakage check FAILED previously (accuracy above threshold). "
              "Stimuli are contaminated; not proceeding.", file=sys.stderr)
        sys.exit(1)


def load_corpora() -> dict:
    return {
        "self": json.loads(CORPUS_PATHS["self"].read_text(encoding="utf-8")),
        "other": json.loads(CORPUS_PATHS["other"].read_text(encoding="utf-8")),
    }


def load_done_ids() -> set[str]:
    done = set()
    if ROLLOUTS_PATH.exists():
        with open(ROLLOUTS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                done.add(json.loads(line)["rollout_id"])
    return done


def append_rollout(record: dict) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with open(ROLLOUTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def seed_from_id(rollout_id: str) -> int:
    return int(hashlib.sha256(rollout_id.encode()).hexdigest(), 16) % (2**32)


def run_rollout(
    client: OpenAI,
    rollout_id: str,
    phase: str,
    cell: str | None,
    system_prompt: str,
    corpus: dict,
    variant_idx: int | None = None,
) -> dict:
    rng = random.Random(seed_from_id(rollout_id))
    history: list[dict] = []
    rounds_data: list[dict] = []
    status = "ok"

    for round_idx in range(1, N_ROUNDS + 1):
        partner_c = PARTNER_CONTRIBUTIONS[round_idx - 1]
        partner_message = rng.choice(corpus[str(round_idx)])
        user_prompt = build_round_user_prompt(round_idx, history, partner_message)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        contribution = reason = None
        parsed_ok = False
        last_raw = None
        for attempt in range(2):  # initial attempt + 1 retry on parse failure
            resp = chat_call(
                client, SUBJECT_MODEL, messages,
                max_tokens=SUBJECT_MAX_TOKENS, temperature=1.0,
                label=f"{phase}:{cell or f'variant{variant_idx}'}:round{round_idx}",
                response_format=SUBJECT_JSON_SCHEMA,
            )
            content = resp.choices[0].message.content
            last_raw = content
            try:
                contribution, reason = parse_subject_response(content)
                parsed_ok = True
                break
            except Exception as e:
                print(f"    parse failure (attempt {attempt + 1}) rollout={rollout_id} "
                      f"round={round_idx}: {e!r} content={content!r}")

        if not parsed_ok:
            status = "invalid"
            rounds_data.append({
                "round": round_idx, "subject_c": None, "partner_c": partner_c,
                "partner_message": partner_message, "reason": None,
                "parse_failed": True, "raw": last_raw,
            })
            break

        subject_payoff, partner_payoff = compute_payoffs(contribution, partner_c)
        round_record = {
            "round": round_idx, "subject_c": contribution, "partner_c": partner_c,
            "subject_payoff": subject_payoff, "partner_payoff": partner_payoff,
            "partner_message": partner_message, "reason": reason,
        }
        rounds_data.append(round_record)
        history.append(round_record)

    mean_contribution = None
    if status == "ok":
        mean_contribution = sum(r["subject_c"] for r in rounds_data) / len(rounds_data)

    return {
        "rollout_id": rollout_id,
        "phase": phase,
        "cell": cell,
        "variant_idx": variant_idx,
        "status": status,
        "model": SUBJECT_MODEL,
        "rounds": rounds_data,
        "mean_contribution": mean_contribution,
        "ts": time.time(),
    }


def build_task_list_main() -> list[dict]:
    tasks = []
    for cell in CELL_NAMES:
        for i in range(ROLLOUTS_PER_CELL_MAIN):
            tasks.append({
                "rollout_id": f"main:{cell}:{i:03d}",
                "phase": "main", "cell": cell, "replicate": i, "variant_idx": None,
            })
    rng = random.Random(MAIN_SEED)
    rng.shuffle(tasks)  # randomize cell order across the run
    return tasks


def build_task_list_noise_floor() -> list[dict]:
    tasks = []
    for v in range(N_NOISE_FLOOR_VARIANTS):
        for i in range(REPLICATES_PER_VARIANT_NOISE_FLOOR):
            tasks.append({
                "rollout_id": f"noise_floor:variant{v:02d}:{i:02d}",
                "phase": "noise_floor", "cell": "baseline", "replicate": i, "variant_idx": v,
            })
    rng = random.Random(MAIN_SEED + 1)
    rng.shuffle(tasks)
    return tasks


def build_task_list_dry_run() -> list[dict]:
    tasks = []
    for cell in CELL_NAMES:
        for i in range(2):
            tasks.append({
                "rollout_id": f"dry_run:{cell}:{i:02d}",
                "phase": "dry_run", "cell": cell, "replicate": i, "variant_idx": None,
            })
    return tasks


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
            record = run_rollout(
                client, task["rollout_id"], task["phase"], task["cell"], system_prompt,
                corpora[corpus_which], variant_idx=task["variant_idx"],
            )
        except BudgetExceeded:
            print(f"Stopping: budget kill switch tripped during {task['rollout_id']}")
            raise
        append_rollout(record)
        results.append(record)
        print(f"  {task['rollout_id']}: status={record['status']} "
              f"mean_contribution={record['mean_contribution']}")
    if n_skipped:
        print(f"Skipped {n_skipped} already-completed rollouts (resumability).")
    return results


def summarize_dry_run(results: list[dict]) -> None:
    ledger = get_ledger()
    calls = ledger.state["calls"]
    subject_calls = [c for c in calls if c["model"] == SUBJECT_MODEL and c["label"].startswith("dry_run:")]
    if not subject_calls:
        print("No dry-run subject calls recorded; cannot extrapolate.")
        return
    mean_prompt = statistics.mean(c["prompt_tokens"] for c in subject_calls)
    mean_completion = statistics.mean(c["completion_tokens"] for c in subject_calls)
    mean_cost_per_call = statistics.mean(c["cost_usd"] for c in subject_calls)
    n_calls_observed = len(subject_calls)

    n_main_calls = len(CELL_NAMES) * ROLLOUTS_PER_CELL_MAIN * N_ROUNDS
    n_noise_floor_calls = N_NOISE_FLOOR_VARIANTS * REPLICATES_PER_VARIANT_NOISE_FLOOR * N_ROUNDS
    total_future_calls = n_main_calls + n_noise_floor_calls

    projected_main = mean_cost_per_call * n_main_calls
    projected_noise_floor = mean_cost_per_call * n_noise_floor_calls
    projected_total = projected_main + projected_noise_floor

    print("\n=== DRY RUN PROJECTION ===")
    print(f"Observed subject calls (dry run): {n_calls_observed}")
    print(f"Mean prompt tokens/call: {mean_prompt:.1f}")
    print(f"Mean completion tokens/call: {mean_completion:.1f}")
    print(f"Mean cost/call: ${mean_cost_per_call:.6f}")
    print(f"Main study subject calls: {n_main_calls} (400 rollouts x {N_ROUNDS} rounds) "
          f"-> projected ${projected_main:.4f}")
    print(f"Noise floor subject calls: {n_noise_floor_calls} (160 rollouts x {N_ROUNDS} rounds) "
          f"-> projected ${projected_noise_floor:.4f}")
    print(f"PROJECTED TOTAL (main + noise floor, subject calls only): ${projected_total:.4f}")
    print(f"Already spent so far (corpus + leakage + this dry run): ${ledger.total:.4f}")
    print(f"PROJECTED GRAND TOTAL after noise floor + main study: "
          f"${ledger.total + projected_total:.4f}")
    print("Kill switch: $6.00 | Hard budget: $7.57")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--noise-floor", action="store_true")
    mode.add_argument("--main", action="store_true")
    args = parser.parse_args()

    print_prices()
    require_corpus_ready()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    corpora = load_corpora()

    if args.dry_run:
        tasks = build_task_list_dry_run()
        print(f"\n=== DRY RUN: {len(tasks)} rollouts ({N_ROUNDS} rounds each) ===")
        results = execute_tasks(client, tasks, corpora)
        summarize_dry_run(results)
    elif args.noise_floor:
        noise_floor_prompts = get_noise_floor_prompts()
        tasks = build_task_list_noise_floor()
        print(f"\n=== NOISE FLOOR: {len(tasks)} rollouts ({N_ROUNDS} rounds each) ===")
        execute_tasks(client, tasks, corpora, noise_floor_prompts=noise_floor_prompts)
    elif args.main:
        tasks = build_task_list_main()
        print(f"\n=== MAIN STUDY: {len(tasks)} rollouts ({N_ROUNDS} rounds each) ===")
        execute_tasks(client, tasks, corpora)

    print_prices()


if __name__ == "__main__":
    main()
