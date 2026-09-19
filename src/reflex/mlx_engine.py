"""Text-only decisions on Apple Silicon, with bounded question-prefix caching.

Supports ChatML-marker architectures (`qwen3`, `qwen3_5`, `llama`) plus
`spark2_5`, which uses its own sentence markers. Other architectures
(qwen4_exp/Flash-Next) need prompt-format and label-token work and are
rejected at load.

Loading passes ``trust_remote_code=True``: mlx-community quants such as
Spark ship a custom tokenizer config that transformers refuses to run
without it. Weights are local files; no remote code executes at inference.
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


class _DirectTokenizer:
    """Minimal tokenizer loaded straight from ``tokenizer.json``.

    Fallback for checkpoints whose custom transformers config crashes
    ``AutoTokenizer`` (e.g. Spark's ``configuration_spark.py`` under
    transformers>=5 rope validation). Exposes the ``encode`` /
    ``chat_template`` surface the backend uses.
    """

    def __init__(self, path):
        import json as _json

        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        template_file = path / "chat_template.jinja"
        if template_file.exists():
            self.chat_template = template_file.read_text()
        else:
            self.chat_template = _json.loads(
                (path / "tokenizer_config.json").read_text()
            ).get("chat_template")

    def encode(self, text, add_special_tokens=False):
        return self._tok.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids):
        return self._tok.decode(ids)


SUPPORTED_MODEL_TYPES = frozenset({"qwen3", "qwen3_5", "llama", "spark2_5"})

# Spark sentence markers. Bars are U+FF5C FULLWIDTH VERTICAL LINE, blanks are
# U+2581 LOWER ONE EIGHTH BLOCK. Copy verbatim; verified against the decoded
# chat template output.
SPARK_OPEN = "<｜start▁of▁sentence｜>"
SPARK_CLOSE = "<｜end▁of▁sentence｜>"


def _load_trusted(model_id):
    """Load weights plus tokenizer, tolerating custom-config checkpoints."""
    try:
        return load(
            model_id,
            return_config=True,
            trust_remote_code=True,
            tokenizer_config={"trust_remote_code": True},
        )
    except Exception:
        # Custom transformers configs (Spark) can fail AutoTokenizer
        # validation while mlx-lm loads the same weights fine. Load each side
        # directly instead of failing the whole model.
        from pathlib import Path as _Path

        from huggingface_hub import snapshot_download
        from mlx_lm.utils import load_config, load_model

        path = _Path(snapshot_download(model_id))
        config = load_config(path)
        model, _ = load_model(path, trust_remote_code=True)
        return model, _DirectTokenizer(path), config


@dataclass
class QuestionFirstFormat(PromptFormat):
    state: Text = ""

    system_head: str = "<|im_start|>system\n"
    system_tail: str = "<|im_end|>\n"
    user_head: str = "<|im_start|>user\n"
    assistant_tail: str = "<|im_end|>\n<|im_start|>assistant\n"
    think_close: str = "<think>\n\n</think>\n\n"

    def branch(self, body: str) -> str:
        prefix = (
            f"{self.system_head}{self.system_prompt}{self.system_tail}"
            f"{self.user_head}"
        )
        body += f"\n# State\n{render_text(self.state)}\n\nRespond with only the option label.\n"
        tail = self.assistant_tail
        if self.no_think and self.think_close:
            tail += self.think_close
        return prefix + body + tail


@dataclass
class SparkFirstFormat(QuestionFirstFormat):
    """Spark-X2.5 sentence-marker layout (thinking disabled)."""

    system_head: str = SPARK_OPEN + "<|System|>\nyou are a helpful assistant.\n\n"
    system_tail: str = SPARK_CLOSE
    user_head: str = SPARK_OPEN + "<|User|>\n"
    assistant_tail: str = SPARK_CLOSE + SPARK_OPEN + "<|Bot|>"
    think_close: str = "</think>"


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
        self.model_type = model.model_type
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
        model, tokenizer, config = _load_trusted(model_id)
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
        out = self.core(mx.array(rest + ids[-1:])[None], cache=cache)
        if out.shape[-1] == self.core.args.vocab_size:
            # Some architectures (spark2_5) return logits from __call__.
            logits = out[:, -1:, :]
        elif self.core.args.tie_word_embeddings:
            hidden = hidden[:, -1:, :]
            logits = self.core.model.embed_tokens.as_linear(hidden)
        else:
            hidden = hidden[:, -1:, :]
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
        fmt_cls = SparkFirstFormat if self.model_type == "spark2_5" else QuestionFirstFormat
        fmt = fmt_cls(
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
