"""Cost ledger + kill switch. Every OpenAI call in this project must go through chat_call().

Budget: HARD_BUDGET is the absolute ceiling this project may ever spend. KILL_SWITCH is checked
BEFORE every call (including every retry attempt): if the current running total plus a
pessimistic estimate of the upcoming call would exceed it, the call is refused and the process
exits. No retry loop bypasses this check -- check_or_die() runs on every attempt, not just the
first.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from common import PRICES

HARD_BUDGET = 7.57
KILL_SWITCH = 6.00

LEDGER_PATH = Path(__file__).parent / "spend.json"


class BudgetExceeded(SystemExit):
    pass


def _load_state() -> dict:
    if LEDGER_PATH.exists():
        with open(LEDGER_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"total_cost_usd": 0.0, "n_calls": 0, "calls": []}


def _save_state(state: dict) -> None:
    tmp = LEDGER_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    tmp.replace(LEDGER_PATH)


class Ledger:
    def __init__(self):
        self.state = _load_state()

    @property
    def total(self) -> float:
        return self.state["total_cost_usd"]

    def _estimate_call_cost(self, model: str, messages: list[dict], max_tokens: int) -> float:
        """Pessimistic pre-call estimate: assumes the full max_tokens is billed at the output
        price and no cache discount applies. Used only for the kill-switch gate, not for
        recorded spend (recorded spend always comes from actual response.usage)."""
        prices = PRICES[model]
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        est_prompt_tokens = prompt_chars / 4 + 20  # padding for role/formatting overhead
        return (est_prompt_tokens * prices["input"] + max_tokens * prices["output"]) / 1e6

    def check_or_die(self, model: str, messages: list[dict], max_tokens: int, label: str = "") -> float:
        est = self._estimate_call_cost(model, messages, max_tokens)
        projected = self.total + est
        if projected > KILL_SWITCH:
            print("\n!!! KILL SWITCH TRIPPED !!!", file=sys.stderr)
            print(f"Current recorded spend: ${self.total:.4f}", file=sys.stderr)
            print(f"Pessimistic estimate for next call ({label}): ${est:.4f}", file=sys.stderr)
            print(
                f"Projected total ${projected:.4f} exceeds kill switch ${KILL_SWITCH:.2f}",
                file=sys.stderr,
            )
            raise BudgetExceeded(1)
        return est

    def record(self, model: str, usage, label: str = "") -> float:
        prices = PRICES[model]
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
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
        self.state["calls"].append(
            {
                "ts": time.time(),
                "model": model,
                "label": label,
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": cost,
                "running_total_usd": self.state["total_cost_usd"],
            }
        )
        _save_state(self.state)
        if self.state["total_cost_usd"] > HARD_BUDGET:
            print(
                f"\n!!! HARD BUDGET EXCEEDED: ${self.state['total_cost_usd']:.4f} > "
                f"${HARD_BUDGET} !!!",
                file=sys.stderr,
            )
            raise BudgetExceeded(1)
        return cost


_ledger_singleton: Ledger | None = None


def get_ledger() -> Ledger:
    global _ledger_singleton
    if _ledger_singleton is None:
        _ledger_singleton = Ledger()
    return _ledger_singleton


def print_prices() -> None:
    print(f"HARD_BUDGET=${HARD_BUDGET}  KILL_SWITCH=${KILL_SWITCH}")
    print("PRICES (USD / 1M tokens, standard tier, fetched 2026-09-02 from "
          "https://developers.openai.com/api/docs/pricing):")
    for model, p in PRICES.items():
        print(f"  {model}: input=${p['input']}  cached_input=${p['cached_input']}  "
              f"output=${p['output']}")
    ledger = get_ledger()
    print(f"Current recorded spend: ${ledger.total:.4f} ({ledger.state['n_calls']} calls)")


def chat_call(
    client,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float = 1.0,
    label: str = "",
    response_format: dict | None = None,
    max_retries: int = 2,
):
    """The one wrapper every API call in this project must go through."""
    ledger = get_ledger()
    last_err = None
    for attempt in range(max_retries + 1):
        ledger.check_or_die(model, messages, max_tokens, label=f"{label} (attempt {attempt + 1})")
        try:
            kwargs = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
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
                time.sleep(2**attempt)
                continue
            raise
    raise last_err  # pragma: no cover
