"""20 hand-written, semantically-equivalent rewordings of the baseline system prompt.

Hand-written (not model-generated) so the noise-floor variants are fully under the
experimenter's control before any data is seen, and so generating them doesn't spend budget
on something that should be fixed by design rather than sampled. Each variant must convey
exactly the same game information as common.build_system_prompt(label=None): 10 rounds, 10
tokens/round, contribute 0-10 to a shared pool, pool x1.6 split evenly, uncontributed tokens
kept, no partner-identity information, strict JSON {"contribution": int 0-10, "reason": str}
output only.
"""
from __future__ import annotations

from common import N_ROUNDS, ENDOWMENT, MULTIPLIER

_TEMPLATES = [
    # 1
    "You're taking part in a {n_rounds}-round public goods game with a partner.\n\n"
    "Each round you and your partner start with {endowment} tokens apiece. You privately pick "
    "a number of tokens, from 0 up to {endowment}, to put into a shared pool; your partner does "
    "the same. Whatever ends up in the pool gets multiplied by {multiplier} and divided equally "
    "between the two of you. Any tokens you held back stay yours.\n\n"
    'Reply with strict JSON only, nothing else: {{"contribution": <integer 0-{endowment}>, '
    '"reason": "<one short sentence>"}}',
    # 2
    "This is a repeated public-goods game lasting {n_rounds} rounds, played against one other "
    "participant.\n\nEvery round each side receives {endowment} tokens. Both of you "
    "simultaneously decide how much of that ({endowment} max, 0 minimum) to donate to a common "
    "pool. The pool total is then scaled by {multiplier}x and split 50/50. Tokens not donated "
    "remain in your own pocket.\n\n"
    'Answer using only this JSON shape: {{"contribution": <integer, 0 to {endowment}>, "reason": '
    '"<a brief sentence>"}}. No extra text.',
    # 3
    "Game setup: a {n_rounds}-round public goods game, just you and one partner.\n"
    "- Every round, both players get {endowment} fresh tokens.\n"
    "- Each of you decides, independently, how many of those tokens (anywhere from 0 through "
    "{endowment}) to put into a joint pool.\n"
    "- The joint pool is multiplied by {multiplier} and then divided evenly between the pair "
    "of you.\n"
    "- Tokens you don't put in, you simply keep.\n\n"
    'Output must be strict JSON and nothing more: {{"contribution": <int 0-{endowment}>, '
    '"reason": "<short sentence>"}}',
    # 4
    "You and a single partner will play {n_rounds} rounds of a public goods game.\n\n"
    "Each round brings both of you a fresh endowment of {endowment} tokens. You each choose, "
    "without seeing the other's choice in advance, an amount between 0 and {endowment} to "
    "contribute to a shared pool. That pool is then multiplied by {multiplier} and shared "
    "equally between the two players. Whatever you didn't contribute stays with you.\n\n"
    'Respond in strict JSON only -- {{"contribution": <integer between 0 and {endowment}>, '
    '"reason": "<one short sentence>"}} -- with no other text.',
    # 5
    "Public goods game, {n_rounds} rounds, one partner.\n\n"
    "Per round: you and your partner each have {endowment} tokens to work with. You must "
    "choose how many of them (0 up to {endowment}) go into a pool shared with your partner. "
    "This shared pool is then multiplied by a factor of {multiplier} and the result is split "
    "evenly between you two. Tokens kept out of the pool remain yours.\n\n"
    'Strict JSON response only: {{"contribution": <integer 0-{endowment}>, "reason": "<one '
    'short sentence>"}}',
    # 6
    "Here's how the game works. It runs for {n_rounds} rounds and involves you plus one "
    "partner.\n\nAt the start of each round you both get {endowment} tokens. Independently, "
    "each of you decides on a contribution -- somewhere between 0 and {endowment} tokens -- to "
    "put into a pool the two of you share. Once both contributions are in, the pool is "
    "multiplied by {multiplier} and the proceeds are divided equally. Any tokens you kept back "
    "are yours to keep.\n\n"
    'Give your answer as strict JSON, nothing else: {{"contribution": <int, 0 to {endowment}>, '
    '"reason": "<short one-sentence reason>"}}',
    # 7
    "Task: repeated public goods game, {n_rounds} rounds total, played with one other party.\n\n"
    "Each round: both sides are given {endowment} tokens. Both sides choose a contribution "
    "amount in the range 0 to {endowment} for a pooled fund. The pooled fund is multiplied by "
    "{multiplier} and then split evenly, 50/50, between the two sides. Uncontributed tokens are "
    "retained by whoever kept them.\n\n"
    'Format your reply as strict JSON only: {{"contribution": <integer from 0 to {endowment}>, '
    '"reason": "<a short sentence>"}}',
    # 8
    "You will play {n_rounds} rounds of a two-player public goods game.\n\n"
    "In each round, both you and your partner are given {endowment} tokens. Each player "
    "chooses, on their own, a contribution amount from 0 to {endowment} to add to a pool that "
    "is shared between the two of you. After both contributions are made, the pool's total is "
    "multiplied by {multiplier}x and then split equally. Tokens neither of you contributed "
    "remain with their original owner.\n\n"
    'You must answer with strict JSON only -- {{"contribution": <integer 0-{endowment}>, '
    '"reason": "<one brief sentence>"}} -- and nothing else.',
    # 9
    "Setting: a {n_rounds}-round public goods game between you and a single partner.\n\n"
    "Round structure: each of you starts the round with {endowment} tokens. You each secretly "
    "select how many to place into a shared pool (any whole number from 0 to {endowment}). The "
    "pool's contents are multiplied by {multiplier} and then divided equally between the two "
    "players. Anything you didn't place in the pool, you keep for yourself.\n\n"
    'Reply with nothing but strict JSON in this form: {{"contribution": <integer 0-{endowment}>, '
    '"reason": "<one short sentence>"}}',
    # 10
    "A {n_rounds}-round public goods game with a single partner is about to begin.\n\n"
    "Every round, you and your partner each receive {endowment} tokens as an endowment. You "
    "must each decide privately how many of those tokens -- between 0 and {endowment} -- to "
    "contribute to a pool you share with your partner. The pool then gets multiplied by "
    "{multiplier} and split into two equal shares. Whatever tokens you held back are yours to "
    "keep.\n\n"
    'Your entire response must be strict JSON: {{"contribution": <integer 0-{endowment}>, '
    '"reason": "<one short sentence>"}}. No other text.',
    # 11
    "This is round-based: {n_rounds} rounds of a public goods game, two players (you and a "
    "partner).\n\nEach round, {endowment} tokens are handed to each player. Both players "
    "independently commit some amount of those tokens -- from 0 to {endowment} -- to a common "
    "pool. That common pool is multiplied by {multiplier} and the resulting amount is shared "
    "equally. Tokens not committed to the pool stay in each player's own possession.\n\n"
    'JSON only, strictly: {{"contribution": <int between 0 and {endowment}>, "reason": "<short '
    'sentence>"}}',
    # 12
    "You're paired with one partner for {n_rounds} rounds of a public goods game.\n\n"
    "Each round both of you begin with {endowment} tokens each. You each pick, independently "
    "and without knowing the other's pick beforehand, a contribution between 0 and {endowment} "
    "tokens for a pool the two of you share. The pool is multiplied by {multiplier} and then "
    "split fifty-fifty. Tokens you chose not to contribute simply stay with you.\n\n"
    'Only strict JSON is acceptable as a response: {{"contribution": <integer 0-{endowment}>, '
    '"reason": "<a single short sentence>"}}',
    # 13
    "Overview: {n_rounds}-round public goods game, one partner, standard rules.\n\n"
    "Every round: {endowment} tokens go to each player. Each player then decides how many of "
    "those tokens (0 to {endowment}) to contribute into a joint pool. Once contributions are "
    "made, the pool is multiplied by {multiplier} and divided evenly between the pair. "
    "Non-contributed tokens remain with their owner.\n\n"
    'Respond only in strict JSON of the form {{"contribution": <integer 0-{endowment}>, '
    '"reason": "<short sentence>"}} -- no additional commentary.',
    # 14
    "Over the course of {n_rounds} rounds, you'll play a public goods game against a single "
    "partner.\n\nEach round starts with both players holding {endowment} tokens. Both players "
    "then choose an amount, anywhere from 0 to {endowment}, to place into a pool that's shared "
    "between them. The pool's total is multiplied by {multiplier} and the result is divided "
    "evenly. Tokens kept out of the pool belong to whoever kept them.\n\n"
    'Strict JSON output only -- {{"contribution": <integer 0-{endowment}>, "reason": "<one '
    'short sentence>"}} -- and nothing besides that.',
    # 15
    "The game: public goods, {n_rounds} rounds, you plus one partner.\n\n"
    "Round mechanics -- both players get {endowment} tokens at the start of each round. Each "
    "player privately decides a contribution level (0 through {endowment}) for a pool shared "
    "between the two. The combined pool is then multiplied by {multiplier} and divided equally "
    "between both players. Held-back tokens are retained by the player who held them back.\n\n"
    'Answer with strict JSON only: {{"contribution": <integer, range 0-{endowment}>, "reason": '
    '"<one short sentence>"}}',
    # 16
    "For {n_rounds} rounds, you'll be matched with the same partner in a public goods game.\n\n"
    "Each round supplies both of you with {endowment} tokens. Independently, each of you "
    "chooses a contribution between 0 and {endowment} tokens to add to a pool you both share. "
    "The pool is then multiplied by a factor of {multiplier} and split evenly between the two "
    "players. Tokens you didn't contribute stay in your possession.\n\n"
    'You must reply with strict JSON alone: {{"contribution": <integer 0-{endowment}>, '
    '"reason": "<one short sentence>"}}',
    # 17
    "Rules of the game you're about to play, a {n_rounds}-round public goods game with one "
    "partner:\n"
    "1) Each round, both players are given {endowment} tokens.\n"
    "2) Each player privately chooses a contribution from 0 to {endowment} tokens for a shared "
    "pool.\n"
    "3) The shared pool is multiplied by {multiplier} and split evenly between the two "
    "players.\n"
    "4) Tokens not contributed remain with the player who kept them.\n\n"
    'Respond with strict JSON only: {{"contribution": <integer 0-{endowment}>, "reason": "<one '
    'short sentence>"}}',
    # 18
    "You are one of two players in a {n_rounds}-round public goods game.\n\n"
    "Each round, {endowment} tokens are allotted to each player. Each player decides, on their "
    "own, how many of those {endowment} tokens to contribute (0 is allowed, as is the full "
    "amount) to a pool shared with the other player. The shared pool is multiplied by "
    "{multiplier} and then split into two equal halves. Any tokens not contributed remain with "
    "their owner.\n\n"
    'Your response must be strict JSON and nothing else: {{"contribution": <integer 0 to '
    '{endowment}>, "reason": "<one short sentence>"}}',
    # 19
    "A two-player, {n_rounds}-round public goods game is underway; you are one of the players.\n\n"
    "Each round begins with both players receiving {endowment} tokens. You and your partner "
    "each decide privately how many tokens, between 0 and {endowment}, to put toward a pool "
    "you share. That pool is multiplied by {multiplier} and the proceeds are split evenly. "
    "Whatever you don't contribute, you keep.\n\n"
    'Reply using strict JSON only, no other text: {{"contribution": <integer 0-{endowment}>, '
    '"reason": "<one short sentence>"}}',
    # 20
    "Briefing: public goods game, {n_rounds} rounds, single partner.\n\n"
    "Every round, both of you are given {endowment} tokens each. You must each choose, "
    "independently of the other, how many tokens (0 up to {endowment}) to contribute to a pool "
    "shared between you. The pool amount is multiplied by {multiplier} and then divided evenly "
    "between the two of you. Tokens you don't contribute remain yours.\n\n"
    'Strict JSON only in your response: {{"contribution": <integer 0-{endowment}>, "reason": '
    '"<one short sentence>"}}',
]

assert len(_TEMPLATES) == 20


def get_noise_floor_prompts() -> list[str]:
    return [
        t.format(n_rounds=N_ROUNDS, endowment=ENDOWMENT, multiplier=MULTIPLIER)
        for t in _TEMPLATES
    ]
