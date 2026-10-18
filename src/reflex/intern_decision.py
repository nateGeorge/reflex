"""Prompt for Intern-Decision checkpoints (arXiv:2610.22131, InternLM/Intern-Decision).

These checkpoints are trained to read every question in one forward pass: the
assistant turn is a JSON skeleton with one ``<decision>`` token per field, and the
logits at the position before each token score that field's answer symbols. The
text below reproduces ``src/inputs/schema.py`` from that repository (Apache-2.0);
any change to it moves the model off its training distribution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from reflex.prompt import render_text
from reflex.schema import ChoiceQuestion, NoulQuestion, SystemOneRequest

DECISION_TOKEN = "<decision>"
SCHEMA_HEADER = "## Decision schema\n"
ANSWER_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
SYSTEM_PROMPT = (
    "You are a careful decision assistant. Use the state and decision schema in "
    "the user message to make the requested decisions. For every field, choose "
    "exactly one answer symbol (e.g. A, B, C, ...) from its listed options and "
    "return one valid JSON object mapping each field name to its chosen symbol. "
    "Use the field names and symbols exactly as given. Do not include "
    "explanations, Markdown, or extra text."
)
NOUL_YES = "The answer is yes (affirmative, or align with the claim)."
NOUL_NO = "The answer is no (negative, or disagree with the claim)."


@dataclass(frozen=True)
class Field:
    qid: str
    symbols: tuple[str, ...]
    keys: tuple  # readout keys in symbol order: option str, score level int, or noul bool


def decision_token_id(tokenizer) -> int | None:
    """The ``<decision>`` id when the checkpoint was trained with it, else None."""
    added = getattr(tokenizer, "get_added_vocab", None)
    return added().get(DECISION_TOKEN) if added else None


def _options(q) -> tuple[list, list[str]]:
    if isinstance(q, ChoiceQuestion):
        return list(q.criteria), [render_text(v) for v in q.criteria.values()]
    if isinstance(q, NoulQuestion):
        c = q.criteria
        no = render_text(c.false) if c and c.false is not None else NOUL_NO
        yes = render_text(c.true) if c and c.true is not None else NOUL_YES
        return [False, True], [no, yes]
    return list(range(len(q.criteria))), [render_text(v) for v in q.criteria]


def _label(key) -> str:
    if isinstance(key, bool):
        return "yes" if key else "no"
    return str(key)


def build_messages(req: SystemOneRequest) -> tuple[list[dict], list[Field]]:
    """Chat messages plus the field order the decision tokens follow."""
    fields, lines = [], []
    for qid, q in req.questions.items():
        keys, descriptions = _options(q)
        symbols = tuple(ANSWER_SYMBOLS[: len(keys)])
        fields.append(Field(qid, symbols, tuple(keys)))
        lines.append(f"{qid}: {render_text(q.instructions)}")
        lines += [f"    {s} = {_label(k)}: {d}" for s, k, d in zip(symbols, keys, descriptions)]
    state = json.dumps(req.state, ensure_ascii=False, indent=2)
    user = (
        "Return one answer for every field using the supplied answer symbols.\n\n"
        f"## State\n{state}\n" + SCHEMA_HEADER + "\n".join(lines)
    )
    if DECISION_TOKEN in user:
        raise ValueError(f"input contains the reserved token {DECISION_TOKEN}")
    skeleton = json.dumps(
        dict.fromkeys((f.qid for f in fields), DECISION_TOKEN), ensure_ascii=False, indent=4
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": skeleton},
    ]
    return messages, fields
