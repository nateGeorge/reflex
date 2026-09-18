"""MLX cache isolation, typed readouts and HTTP input boundaries without model downloads."""

import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient

mx = pytest.importorskip("mlx.core")
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
    """Cached multi-question readouts match uncached logits across states and permutations."""
    for state in ("A duplicate invoice needs a refund.", "A login error blocks access."):
        req = request_for(state, permutations=3)
        expected = {}
        rng = random.Random(0)
        fmt = QuestionFirstFormat(state=state)
        for qid, q in req.questions.items():
            results = []
            for branch in build_branches(qid, q, fmt, req.permutations, rng):
                ids = engine.tok.encode(branch.text)
                logits = engine.model(mx.array(ids)[None])[0, -1, engine._label_ids(branch.labels)]
                results.append((branch, np.array(logits)))
            expected[qid] = to_answer(q.type, merge_branches(q.type, results, engine.cal), q)
        for _ in range(2):
            response = engine.answer(req)
            for qid, answer in response.answers.items():
                if answer.type == "noul":
                    assert answer.noul == pytest.approx(expected[qid].noul, abs=2e-5)
                else:
                    np.testing.assert_allclose(
                        list(answer.probabilities.values()),
                        list(expected[qid].probabilities.values()),
                        atol=2e-5,
                    )
            assert response.usage.output_tokens == 0
            assert not response.usage.state_cache_hit
            assert response.usage.input_tokens == (
                response.usage.state_tokens + response.usage.question_tokens
            )
        assert len(engine.cache) <= 8
        assert engine.cache.nbytes <= 128 * 1024**2


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
        for response, baseline in zip(responses, expected):
            actual = response.json()
            assert actual["usage"] == baseline["usage"]
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
