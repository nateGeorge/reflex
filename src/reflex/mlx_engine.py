"""Text-only Qwen decisions on Apple Silicon, with bounded question-prefix caching.

Supports `qwen3` and `qwen3_5` architectures (including Qwen3.5 dense models).
Other architectures (Spark, Llama, qwen4_exp/Flash-Next) need prompt-format
and label-token work and are rejected at load.
"""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx_lm import load
from mlx_lm.models.cache import LRUPromptCache, make_prompt_cache

from reflex.images import has_images
from reflex.prompt import PromptFormat, build_branches, render_text
from reflex.readout import Calibration, merge_branches, to_answer
from reflex.schema import SystemOneRequest, SystemOneResponse, Text, Usage


SUPPORTED_MODEL_TYPES = frozenset({"qwen3", "qwen3_5"})


@dataclass
class QuestionFirstFormat(PromptFormat):
    state: Text = ""

    def branch(self, body: str) -> str:
        prefix = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n"
        body += f"\n# State\n{render_text(self.state)}\n\nRespond with only the option label.\n"
        return prefix + super().branch(body)


class MLXEngine:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        model_name="reflex-latest",
        calibration=None,
        max_pack_tokens=8192,
    ):
        if model.model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"MLX backend supports {sorted(SUPPORTED_MODEL_TYPES)} models, "
                f"got {model.model_type!r}"
            )
        # Some mlx-lm architectures (e.g. qwen3_5) wrap the text model in a
        # multimodal container exposing it as ``language_model``. Unwrap once
        # so readout works against the same interface for all archs.
        core = getattr(model, "language_model", model)
        self.core = core
        model.eval()
        # Materialize weights before FastAPI moves inference to worker threads.
        mx.eval(core.parameters())
        self.model = model
        self.tok = tokenizer
        self.model_name = model_name
        self.device = mx.default_device()
        self.cal = calibration or Calibration()
        self.max_pack_tokens = max_pack_tokens
        self.cache = LRUPromptCache(max_size=8, max_bytes=128 * 1024**2)
        self._lock = threading.Lock()

    @classmethod
    def load(cls, model_id, *, calibration_path=None, max_pack_tokens=8192):
        if not mx.metal.is_available():
            raise RuntimeError("MLX backend needs an Apple Silicon GPU")
        mx.set_default_device(mx.gpu)
        model, tokenizer, config = load(model_id, return_config=True)
        model_type = config.get("model_type")
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"MLX backend supports {sorted(SUPPORTED_MODEL_TYPES)} models, "
                f"got {model_type!r}"
            )
        if not config.get("quantization"):
            nn.quantize(model, bits=4, group_size=64)
        return cls(
            model,
            tokenizer,
            model_name=model_id,
            calibration=Calibration.load(calibration_path),
            max_pack_tokens=max_pack_tokens,
        )

    def _label_ids(self, labels):
        ids = [self.tok.encode(label, add_special_tokens=False) for label in labels]
        for label, tokens in zip(labels, ids):
            if len(tokens) != 1:
                raise ValueError(f"label {label!r} is not a single token: {tokens}")
        return mx.array([tokens[0] for tokens in ids])

    def _logits(self, ids, labels):
        label_ids = self._label_ids(labels)
        # Leave a token to evaluate even when the cache already contains the whole prompt.
        cache, rest = self.cache.fetch_nearest_cache(self.model_name, ids[:-1])
        if cache is None:
            cache = make_prompt_cache(self.core)
        hidden = self.core.model(mx.array(rest + ids[-1:])[None], cache=cache)[:, -1:, :]
        if self.core.args.tie_word_embeddings:
            logits = self.core.model.embed_tokens.as_linear(hidden)
        else:
            logits = self.core.lm_head(hidden)
        restricted = logits[0, -1, label_ids].astype(mx.float32)
        mx.eval(restricted, [c.state for c in cache])
        self.cache.insert_cache(self.model_name, ids, cache)
        return np.array(restricted)

    def answer(self, req: SystemOneRequest) -> SystemOneResponse:
        with self._lock:
            return self._answer(req)

    def _answer(self, req):
        if has_images(req.state):
            raise ValueError("state contains images but the loaded model is text-only")
        fmt = QuestionFirstFormat(
            state=req.state,
            no_think="enable_thinking" in (self.tok.chat_template or ""),
        )
        rng = random.Random(0)
        branches = [
            branch
            for qid, q in req.questions.items()
            for branch in build_branches(qid, q, fmt, req.permutations, rng)
        ]
        inputs = [self.tok.encode(b.text, add_special_tokens=False) for b in branches]
        state_tokens = len(self.tok.encode(render_text(req.state), add_special_tokens=False))
        for ids in inputs:
            if len(ids) > self.max_pack_tokens:
                raise ValueError(f"prompt too long: {len(ids)} > {self.max_pack_tokens}")
            if len(ids) - state_tokens > 4096:
                raise ValueError("question branch too long: maximum 4096 tokens including framing")
        per_q = {}
        for branch, ids in zip(branches, inputs):
            logits = self._logits(ids, branch.labels)
            per_q.setdefault(branch.qid, []).append((branch, logits))
        answers = {
            qid: to_answer(q.type, merge_branches(q.type, per_q[qid], self.cal), q)
            for qid, q in req.questions.items()
        }
        total = sum(map(len, inputs))
        state_total = state_tokens * len(branches)
        return SystemOneResponse(
            model=self.model_name,
            answers=answers,
            usage=Usage(
                input_tokens=total,
                state_tokens=state_total,
                question_tokens=total - state_total,
                state_cache_hit=False,
            ),
        )
