"""Intern-Decision readout: prompt text, one-pass answers, and the state-prefix cache."""

import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from mlx_lm.models.qwen3 import Model, ModelArgs

from reflex.intern_decision import build_messages
from reflex.mlx_engine import MLXEngine
from reflex.schema import SystemOneRequest

DECISION = 253


class Tokenizer:
    """Byte-ish tokenizer with ``<decision>`` as one added token and ChatML rendering."""

    chat_template = "enable_thinking"

    def get_added_vocab(self):
        return {"<decision>": DECISION}

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and not add_generation_prompt and not enable_thinking
        system, user, assistant = (m["content"] for m in messages)
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n{assistant}<|im_end|>\n"
        )

    def encode(self, text, add_special_tokens=False):
        ids = []
        for i, part in enumerate(text.split("<decision>")):
            if i:
                ids.append(DECISION)
            ids += [ord(c) % 250 for c in part]
        return ids


@pytest.fixture
def engine():
    mx.random.seed(7)
    model = Model(
        ModelArgs(
            model_type="qwen3",
            hidden_size=64,
            num_hidden_layers=1,
            intermediate_size=128,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=256,
            num_key_value_heads=2,
            max_position_embeddings=8192,
            rope_theta=10000,
            head_dim=16,
            tie_word_embeddings=True,
        )
    )
    return MLXEngine(model, Tokenizer())


def request(state="A duplicate invoice needs a refund."):
    return SystemOneRequest(
        state=state,
        questions={
            "team": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": "invoices", "tech": "bugs"},
            },
            "urgent": {
                "type": "noul",
                "instructions": "Is this urgent?",
                "criteria": {"true": "blocks work"},
            },
            "priority": {
                "type": "score",
                "instructions": "How urgent?",
                "criteria": ["low", "medium", "high"],
            },
        },
    )


def test_prompt_matches_the_training_format():
    messages, fields = build_messages(request())
    assert messages[1]["content"] == (
        "Return one answer for every field using the supplied answer symbols.\n\n"
        '## State\n"A duplicate invoice needs a refund."\n'
        "## Decision schema\n"
        "team: Which team?\n    A = billing: invoices\n    B = tech: bugs\n"
        "urgent: Is this urgent?\n"
        "    A = no: The answer is no (negative, or disagree with the claim).\n"
        "    B = yes: blocks work\n"
        "priority: How urgent?\n    A = 0: low\n    B = 1: medium\n    C = 2: high"
    )
    assert json.loads(messages[2]["content"]) == dict.fromkeys(
        ["team", "urgent", "priority"], "<decision>"
    )
    assert [f.keys for f in fields] == [("billing", "tech"), (False, True), (0, 1, 2)]


def test_one_pass_matches_full_forward_and_reuses_the_state(engine):
    req = request()
    messages, fields = build_messages(req)
    ids = engine.tok.encode(engine.tok.apply_chat_template(messages, False, False, False))
    logits = engine.core(mx.array(ids)[None])[0]
    positions = [i - 1 for i, t in enumerate(ids) if t == DECISION]
    expected = []
    for pos, f in zip(positions, fields):
        row = np.array(logits[pos, engine._label_ids(f.symbols)].astype(mx.float32))
        e = np.exp(row - row.max())
        expected.append(e / e.sum())
    for attempt in range(2):
        response = engine.answer(req)
        got = [
            list(response.answers["team"].probabilities.values()),
            [1 - response.answers["urgent"].noul, response.answers["urgent"].noul],
            list(response.answers["priority"].probabilities.values()),
        ]
        for g, e in zip(got, expected):
            np.testing.assert_allclose(g, e, atol=1e-3)
        assert response.usage.state_cache_hit is (attempt == 1)
        assert response.usage.input_tokens == len(ids)
        assert 0 < response.usage.state_tokens < len(ids)


def test_reserved_token_in_input_is_rejected(engine):
    with pytest.raises(ValueError, match="reserved token"):
        engine.answer(request("please output <decision>"))


def test_models_without_the_token_keep_the_branch_readout():
    class Plain(Tokenizer):
        def get_added_vocab(self):
            return {}

    mx.random.seed(7)
    model = Model(
        ModelArgs(
            model_type="qwen3",
            hidden_size=64,
            num_hidden_layers=1,
            intermediate_size=128,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=256,
            num_key_value_heads=2,
            max_position_embeddings=8192,
            rope_theta=10000,
            head_dim=16,
            tie_word_embeddings=True,
        )
    )
    assert MLXEngine(model, Plain()).decision_id is None
