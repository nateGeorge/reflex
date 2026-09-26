"""MLX cache isolation, typed readouts and HTTP input boundaries without model downloads."""

import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient

mx = pytest.importorskip("mlx.core")
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.qwen3 import Model, ModelArgs

from reflex.mlx_engine import MLXEngine, QuestionFirstFormat
from reflex.prompt import build_branches
from reflex.readout import merge_branches, to_answer
from reflex.schema import SystemOneRequest
from reflex.server import create_app, main


class Tokenizer:
    chat_template = "enable_thinking"

    def encode(self, text, add_special_tokens=False):
        return {"Yes": [254], "No": [255]}.get(text, [ord(c) % 254 for c in text])


@pytest.fixture(params=[True, False])
def engine(request):
    """Use a small real Qwen3 model with both tied and separate output weights."""
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
            tie_word_embeddings=request.param,
        )
    )
    return MLXEngine(model, Tokenizer())


def request_for(state="A duplicate invoice needs a refund.", permutations=1):
    return SystemOneRequest(
        state=state,
        permutations=permutations,
        questions={
            "team": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {
                    "billing": "invoices",
                    "tech": "bugs",
                    "other": "everything else",
                },
            },
            "urgent": {"type": "noul", "instructions": "Is this urgent?"},
            "priority": {
                "type": "score",
                "instructions": "How urgent?",
                "criteria": [
                    "low",
                    "medium",
                    "high",
                ],
            },
        },
    )


def test_cache_matches_full_prompt_and_isolates_branches(engine):
    """Reused prefixes match a private cache per branch, across states and permutations."""
    for state in ("A duplicate invoice needs a refund.", "A login error blocks access."):
        req = request_for(state, permutations=3)
        expected = {}
        rng = random.Random(0)
        fmt = QuestionFirstFormat(state=state)
        prefix_ids = engine.tok.encode(fmt.prefix(state))
        for qid, q in req.questions.items():
            results = []
            for branch in build_branches(qid, q, fmt, req.permutations, rng):
                # Reference: a fresh cache per branch, so nothing is shared and
                # nothing is trimmed. A leak between branches shows up here.
                cache = make_prompt_cache(engine.core)
                engine.core(mx.array(prefix_ids)[None], cache=cache)
                ids = engine.tok.encode(branch.text)
                logits = engine.core(mx.array(ids)[None], cache=cache)[0, -1]
                results.append((branch, np.array(logits[engine._label_ids(branch.labels)])))
            expected[qid] = to_answer(q.type, merge_branches(q.type, results, engine.cal), q)
        for attempt in range(2):
            response = engine.answer(req)
            for qid, answer in response.answers.items():
                # The reference projects the head over every branch position at
                # once; the engine projects only the last. Different matmul
                # shapes reassociate floats, which on an untrained model moves
                # probabilities by ~5e-5. A wrong cache offset or a leaked
                # branch would move them by O(0.1), so this stays meaningful.
                if answer.type == "noul":
                    assert answer.noul == pytest.approx(expected[qid].noul, abs=1e-3)
                else:
                    np.testing.assert_allclose(
                        list(answer.probabilities.values()),
                        list(expected[qid].probabilities.values()),
                        atol=1e-3,
                    )
            assert response.usage.output_tokens == 0
            # The state prefix is computed once per request, then reused.
            assert response.usage.state_cache_hit is (attempt == 1)
            assert response.usage.input_tokens == (
                response.usage.state_tokens + response.usage.question_tokens
            )
        assert len(engine.cache) <= engine.cache.max_size
        assert engine.cache.nbytes <= engine.cache.max_bytes


def test_branches_share_one_state_prefix(engine):
    """Branches differ only after the shared state, so they cost one cache entry."""
    state = "A duplicate invoice needs a refund."
    req = request_for(state, permutations=3)
    fmt = QuestionFirstFormat(state=state)
    rng = random.Random(0)
    texts = [
        branch.text
        for qid, q in req.questions.items()
        for branch in build_branches(qid, q, fmt, req.permutations, rng)
    ]
    assert len(texts) > 1
    assert all(not t.startswith("# State") for t in texts)
    assert all(state not in t for t in texts)

    engine.answer(req)
    assert len(engine.cache) == 1, "one state prefix, not one entry per branch"
    # A second state adds exactly one more; the branches still add none.
    engine.answer(request_for("A login error blocks access."))
    assert len(engine.cache) == 2


def test_http_concurrent_requests_and_validation(engine):
    """Concurrent HTTP requests preserve answers and invalid input returns 422."""
    requests = [request_for("refund invoice"), request_for("login problem")]
    with TestClient(create_app(engine)) as client:
        expected = [client.post("/v1/systemone", json=r.model_dump()).json() for r in requests]
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(
                pool.map(
                    lambda r: client.post("/v1/systemone", json=r.model_dump()),
                    requests,
                )
            )
        assert [r.status_code for r in responses] == [200, 200]
        # Both states are new, so the first round misses the cache.
        assert [r["usage"]["state_cache_hit"] for r in expected] == [False, False]
        for response, baseline in zip(responses, expected):
            actual = response.json()
            # Same numbers billed, but the second round reuses each state prefix.
            assert actual["usage"]["state_cache_hit"] is True
            for key in ("input_tokens", "state_tokens", "question_tokens"):
                assert actual["usage"][key] == baseline["usage"][key]
            for qid, answer in actual["answers"].items():
                before = baseline["answers"][qid]
                if answer["type"] == "noul":
                    assert answer["noul"] == pytest.approx(before["noul"], abs=2e-5)
                else:
                    assert answer["probabilities"] == pytest.approx(
                        before["probabilities"],
                        abs=2e-5,
                    )
                    if answer["type"] == "choice":
                        assert answer["choice"] == before["choice"]
        assert client.get("/healthz").json()["device"] == "Device(gpu, 0)"
        assert client.post("/v1/systemone", json={"state": "bad"}).status_code == 422
        image = request_for({"type": "image", "source": "not-fetched"})
        assert client.post("/v1/systemone", json=image.model_dump()).status_code == 422
        engine.max_pack_tokens = 10
        assert client.post("/v1/systemone", json=requests[0].model_dump()).status_code == 422


def test_request_without_permutations_uses_server_default(engine):
    """An omitted permutations field means "the server's setting", not None.

    Regression: the field is optional in the schema, and None used to reach
    distinct_orders, which compared int to None and returned 500.
    """
    body = request_for().model_dump()
    body.pop("permutations")
    with TestClient(create_app(engine)) as client:
        response = client.post("/v1/systemone", json=body)
    assert response.status_code == 200, response.text
    baseline = engine.answer(request_for(permutations=1))
    assert response.json()["usage"]["question_tokens"] == baseline.usage.question_tokens


def test_default_permutations_is_configurable(engine):
    """--permutations sets the default for requests that omit the field; an
    explicit request value still wins."""
    omitted = request_for().model_dump()
    omitted.pop("permutations")
    engine.default_permutations = 3
    three = engine.answer(request_for(permutations=3))
    assert (
        engine.answer(SystemOneRequest(**omitted)).usage.question_tokens
        == three.usage.question_tokens
    )
    one = engine.answer(request_for(permutations=1))
    assert one.usage.question_tokens < three.usage.question_tokens


def test_mlx_dispatch_passes_default_permutations(engine, monkeypatch):
    """The mlx backend accepts --permutations and hands it to the engine."""
    seen = {}
    monkeypatch.setattr(MLXEngine, "load", lambda *args, **kwargs: seen.update(kwargs) or engine)
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)
    main(["--backend", "mlx", "--permutations", "3"])
    assert seen["default_permutations"] == 3


def test_labels_and_branch_limits(engine, monkeypatch):
    """Multi-token labels and oversized question branches fail before inference."""
    monkeypatch.setattr(engine.tok, "encode", lambda *args, **kwargs: [1, 2])
    with pytest.raises(ValueError, match="not a single token"):
        engine._label_ids(["A"])
    monkeypatch.undo()
    req = request_for()
    req.questions["team"].instructions = "x" * 4200
    with pytest.raises(ValueError, match="question branch too long"):
        engine.answer(req)


def test_warmup_precedes_serving(engine, monkeypatch, tmp_path):
    """Startup evaluates the configured request before the server accepts traffic."""
    events = []
    req = request_for()
    warmup = tmp_path / "warmup.json"
    warmup.write_text(req.model_dump_json())
    monkeypatch.setattr(MLXEngine, "load", lambda *args, **kwargs: engine)
    monkeypatch.setattr(engine, "answer", lambda request: events.append(request))
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: events.append("serve"))
    main(["--backend", "mlx", "--warmup", str(warmup)])
    assert events == [req, "serve"]


def test_mlx_rejects_torch_options():
    """The CLI rejects ignored Torch options without loading a model."""
    with pytest.raises(SystemExit) as exc:
        main(["--backend", "mlx", "--device", "cpu"])
    assert exc.value.code == 2
