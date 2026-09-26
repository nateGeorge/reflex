"""Text-only decisions on Apple Silicon, with bounded question-prefix caching.

Supports ChatML-marker architectures (`qwen3`, `qwen3_5`, `llama`), plus
`spark2_5` (sentence markers), `mistral3`/`ministral3` (`[INST]`), and
`gemma3n` (`<start_of_turn>`). Other architectures
(qwen4_exp/Flash-Next) need prompt-format and label-token work and are
rejected at load.

Loading passes ``trust_remote_code`` through ``tokenizer_config``: mlx-community
quants such as Spark ship a custom tokenizer config that transformers refuses to
run without it. Weights are local files; no remote code executes at inference.
(mlx-lm >= 0.31 dropped the top-level ``trust_remote_code`` argument from
``load``/``load_model``; passing it there raises TypeError.)
"""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx_lm import load
from mlx_lm.models.cache import LRUPromptCache, make_prompt_cache, trim_prompt_cache

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


SUPPORTED_MODEL_TYPES = frozenset({
    "qwen3", "qwen3_5", "llama", "spark2_5",
    "mistral3", "ministral3", "gemma3n",
})

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
        model, _ = load_model(path)
        return model, _DirectTokenizer(path), config


@dataclass
class QuestionFirstFormat(PromptFormat):
    """ChatML layout with the state in the shared prefix (see reflex.prompt).

    Every branch of a request shares the system turn and the ``# State`` block
    and differs only in the question that follows it. Keeping the state first is
    what makes that sharing usable: a prefix cache is only reusable when the
    common text is a prefix, so a layout that puts the state last forces a full
    re-read of the state once per branch.
    """

    state: Text = ""

    system_head: str = "<|im_start|>system\n"
    system_tail: str = "<|im_end|>\n"
    user_head: str = "<|im_start|>user\n"
    assistant_tail: str = "<|im_end|>\n<|im_start|>assistant\n"
    think_close: str = "<think>\n\n</think>\n\n"

    def prefix(self, state: Text | None = None) -> str:
        """Shared prefix: the system turn, then the state under a ``# State`` header.

        ``state`` defaults to the constructor field, so callers can build the
        format up front and still render a prefix per request.
        """
        state = self.state if state is None else state
        return (
            f"{self.system_head}{self.system_prompt}{self.system_tail}"
            f"{self.user_head}# State\n{render_text(state)}\n\n"
        )

    def branch(self, body: str) -> str:
        """Per-branch suffix: the question and options, then the assistant turn."""
        body += "\nRespond with only the option label.\n"
        tail = self.assistant_tail
        if self.no_think and self.think_close:
            tail += self.think_close
        return body + tail


@dataclass
class GemmaFirstFormat(QuestionFirstFormat):
    """Gemma `<start_of_turn>` layout. No system role: system text merges
    into the user turn. Generation starts after `<start_of_turn>model`."""

    system_head: str = "<bos><start_of_turn>user\n"
    system_tail: str = "\n\n"
    user_head: str = ""
    assistant_tail: str = "<end_of_turn>\n<start_of_turn>model\n"
    think_close: str = ""


@dataclass
class MistralFirstFormat(QuestionFirstFormat):
    """Ministral/Mistral `[INST]` layout. Generation starts right after `[/INST]`."""

    system_head: str = "<s>[SYSTEM_PROMPT]"
    system_tail: str = "[/SYSTEM_PROMPT]"
    user_head: str = "[INST]\n"
    assistant_tail: str = "[/INST]"
    think_close: str = ""


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
        cache_entries=16,
        cache_bytes=6 * 1024**3,
        default_permutations=1,
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
        # A request may omit ``permutations``; the field then means "the server's
        # setting", the same contract the torch engine implements.
        self.default_permutations = default_permutations
        # Sized in bytes, because that is the real limit: one 20k-token state
        # is ~2.9 GB of KV. The entry count only keeps the short interactive
        # states that arrive between long ones from evicting them.
        self.cache = LRUPromptCache(max_size=cache_entries, max_bytes=cache_bytes)
        self._lock = threading.Lock()
        self._trunk, self._head = self._split_model()

    @classmethod
    def load(
        cls,
        model_id,
        *,
        calibration_path=None,
        max_pack_tokens=8192,
        cache_entries=16,
        cache_bytes=6 * 1024**3,
        cache_limit_bytes=2 * 1024**3,
        default_permutations=1,
    ):
        if not mx.metal.is_available():
            raise RuntimeError("MLX backend needs an Apple Silicon GPU")
        mx.set_default_device(mx.gpu)
        # MLX's free-buffer cache defaults to the memory limit, which is 1.5x the
        # recommended working set (~61 GiB on a 64 GB Mac). A server that churns
        # multi-gigabyte prompt caches therefore keeps every freed buffer in the
        # allocator instead of returning it, and the machine swaps while the cache
        # holds memory nothing will reuse. Bound it: below the bound buffers are
        # still reused, above it they go back to the OS.
        mx.set_cache_limit(cache_limit_bytes)
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
            cache_entries=cache_entries,
            cache_bytes=cache_bytes,
            default_permutations=default_permutations,
        )

    def _label_ids(self, labels):
        ids = [self.tok.encode(label, add_special_tokens=False) for label in labels]
        for label, tokens in zip(labels, ids):
            if len(tokens) != 1:
                raise ValueError(f"label {label!r} is not a single token: {tokens}")
        return mx.array([tokens[0] for tokens in ids])

    def _split_model(self):
        """Split the model into ``(trunk, head)`` when that is provably equivalent.

        mlx-lm's ``Model.__call__`` is ``head(trunk(inputs, cache))``. Running the
        halves separately keeps the output head -- a ``vocab_size``-wide
        projection -- off the prompt positions. A 20k-token state otherwise
        materializes a ``[1, 20000, 151936]`` logits tensor, several GB, to read
        a single position, and it does that once per branch.

        Checked against the model's own ``__call__`` at load, so an architecture
        with a different layout falls back to the plain forward instead of
        quietly computing something else.
        """
        trunk = getattr(self.core, "model", None)
        if trunk is None:
            return None, None
        head = getattr(self.core, "lm_head", None)
        if head is None:
            head = getattr(getattr(trunk, "embed_tokens", None), "as_linear", None)
        if head is None:
            return None, None
        ids = mx.array([[1, 2, 3]])
        try:
            full = self.core(ids)
            split = head(trunk(ids))
            mx.eval(full, split)
        except Exception:
            return None, None
        if full.shape != split.shape or not mx.allclose(full, split, rtol=1e-2, atol=1e-2):
            return None, None
        return trunk, head

    def _prefill(self, ids, cache):
        """Run ``ids`` into ``cache``. The output is discarded, so skip the head."""
        arr = mx.array(ids)[None]
        if self._trunk is None:
            self.core(arr, cache=cache)
        else:
            self._trunk(arr, cache=cache)

    def _label_logits(self, ids, cache, labels):
        """Readout logits for ``labels`` after appending ``ids`` to ``cache``.

        Only the final position is projected. The caller trims ``ids`` back off
        ``cache`` afterwards, so the shared prefix survives for the next branch.
        """
        arr = mx.array(ids)[None]
        if self._trunk is None:
            logits = self.core(arr, cache=cache)[:, -1:, :]
        else:
            logits = self._head(self._trunk(arr, cache=cache)[:, -1:, :])
        restricted = logits[0, -1, self._label_ids(labels)].astype(mx.float32)
        mx.eval(restricted)
        return np.array(restricted)

    def _state_cache(self, prefix_ids):
        """KV cache covering the shared state prefix, reused by every branch.

        Returns ``(cache, hit)``. The cache is private to the caller: branches
        extend it and trim back. On a miss the caller inserts it into the LRU
        once the branches are done, so a request never pays for a second copy of
        a multi-gigabyte cache.
        """
        cache, rest = self.cache.fetch_nearest_cache(self.model_name, prefix_ids)
        if cache is not None and not rest:
            return cache, True
        if cache is None:
            cache = make_prompt_cache(self.core)
        if rest:
            self._prefill(rest, cache)
            mx.eval([c.state for c in cache])
        return cache, False

    def answer(self, req: SystemOneRequest) -> SystemOneResponse:
        with self._lock:
            return self._answer(req)

    def _answer(self, req):
        if has_images(req.state):
            raise ValueError("state contains images but the loaded model is text-only")
        if self.model_type == "gemma3n":
            fmt_cls = GemmaFirstFormat
        elif self.model_type.startswith("mistral") or self.model_type.startswith("ministral"):
            fmt_cls = MistralFirstFormat
        elif self.model_type == "spark2_5":
            fmt_cls = SparkFirstFormat
        else:
            fmt_cls = QuestionFirstFormat
        fmt = fmt_cls(
            state=req.state,
            no_think="enable_thinking" in (self.tok.chat_template or ""),
        )
        rng = random.Random(0)
        # None is "the server's setting", not zero orders: coalesce exactly the way
        # the torch engine does, or distinct_orders compares int to None.
        permutations = req.permutations or self.default_permutations
        branches = [
            branch
            for qid, q in req.questions.items()
            for branch in build_branches(qid, q, fmt, permutations, rng)
        ]
        # The state is tokenized once, as a prefix shared by every branch, and
        # its KV cache is computed once per request instead of once per branch.
        prefix_ids = self.tok.encode(fmt.prefix(req.state), add_special_tokens=False)
        inputs = [self.tok.encode(b.text, add_special_tokens=False) for b in branches]
        for ids in inputs:
            if len(prefix_ids) + len(ids) > self.max_pack_tokens:
                raise ValueError(
                    f"prompt too long: {len(prefix_ids) + len(ids)} > {self.max_pack_tokens}"
                )
            if len(ids) > 4096:
                raise ValueError("question branch too long: maximum 4096 tokens including framing")
        cache, hit = self._state_cache(prefix_ids)
        per_q = {}
        for branch, ids in zip(branches, inputs):
            logits = self._label_logits(ids, cache, branch.labels)
            per_q.setdefault(branch.qid, []).append((branch, logits))
            # Back to the shared prefix, so the next branch starts at the same
            # offset and reads the same KV entries.
            trim_prompt_cache(cache, len(ids))
        if not hit:
            self.cache.insert_cache(self.model_name, prefix_ids, cache)
        answers = {
            qid: to_answer(q.type, merge_branches(q.type, per_q[qid], self.cal), q)
            for qid, q in req.questions.items()
        }
        total = sum(map(len, inputs))
        state_total = len(prefix_ids) * len(branches)
        return SystemOneResponse(
            model=self.model_name,
            answers=answers,
            usage=Usage(
                input_tokens=state_total + total,
                state_tokens=state_total,
                question_tokens=total,
                state_cache_hit=hit,
            ),
        )
