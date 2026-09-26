"""FastAPI server exposing the TypeSafe-compatible endpoint `POST /v1/systemone`.

    reflex-serve --model Qwen/Qwen3-8B --port 8008

Then any client written for Jev can point at http://localhost:8008 instead.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from reflex.backends import BackendError
from reflex.mps import is_out_of_memory
from reflex.priority import PriorityLock, request_priority
from reflex.schema import SystemOneRequest, SystemOneResponse

log = logging.getLogger("reflex.server")


def create_app(engine, api_key: str | None = None) -> FastAPI:
    """The HTTP surface. `api_key` (or env REFLEX_API_KEY) makes /v1/systemone require
    `Authorization: Bearer <key>`, which you want on any endpoint reachable from the
    internet, e.g. a rented GPU an evaluator calls."""
    app = FastAPI(title="reflex", version="0.1.0")
    api_key = api_key or os.environ.get("REFLEX_API_KEY") or None

    @app.middleware("http")
    async def _auth(request: Request, call_next):
        if (
            api_key
            and request.url.path.startswith("/v1/")
            and request.headers.get("authorization", "") != f"Bearer {api_key}"
        ):
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "Invalid or missing API key",
                        "type": "invalid_api_key",
                    }
                },
            )
        return await call_next(request)

    # One model instance on one accelerator: requests are serialized either way,
    # but cheapest-first, so an interactive 1-question call never waits out a
    # multi-question batch. See reflex.priority.
    gate = PriorityLock()

    @app.get("/healthz")
    @app.get("/health")
    def healthz():
        cache = getattr(engine, "cache", None)
        return {
            "ok": True,
            "status": "healthy",
            "model": engine.model_name,
            "calibration": engine.cal.temperature,
            # MLXEngine is the strategy and names none; torch and SGLang set one.
            "strategy": getattr(engine, "strategy", None),
            "device": str(engine.device),
            "cache": None
            if cache is None
            else {"entries": len(cache), "bytes": cache.nbytes, "max_bytes": cache.max_bytes},
        }

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [{"id": engine.model_name, "object": "model", "owned_by": "reflex"}],
        }

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    def systemone(req: SystemOneRequest):
        # t0 is set inside the gate below, so the latency we report leaves the queue out.
        out_of_memory = False
        queued_at = time.perf_counter()
        wait_ms = 0.0
        try:
            with gate.hold(request_priority(req)):
                wait_ms = (time.perf_counter() - queued_at) * 1000
                t0 = time.perf_counter()
                resp = engine.answer(req)
        except ValueError as e:  # bad labels / too long
            raise HTTPException(status_code=422, detail=str(e))
        except BackendError as e:
            # The inference server behind us failed, not us. A dropped connection under a
            # deep fan-out is transient and worth retrying; say so with 502 rather than
            # letting it surface as an opaque 500.
            log.warning("backend failed: %s", e)
            raise HTTPException(status_code=502, detail=f"backend unavailable: {e}")
        except torch.cuda.OutOfMemoryError:
            out_of_memory = True
        except RuntimeError as e:  # MPS has no OutOfMemoryError of its own
            if not is_out_of_memory(e):
                raise
            out_of_memory = True
        if out_of_memory:
            # Free the memory out here, not inside the except block: there the traceback still
            # holds the frames that own the tensors, so empty_cache() releases nothing and the
            # next request fails too.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            raise HTTPException(status_code=529, detail="overloaded: request too large for GPU")
        ms = (time.perf_counter() - t0) * 1000
        log.info(
            "%d questions, %d state tok (%s), %d q tok, %.0f ms (queued %.0f ms)",
            len(req.questions),
            resp.usage.state_tokens,
            "hit" if resp.usage.state_cache_hit else "miss",
            resp.usage.question_tokens,
            ms,
            wait_ms,
        )
        headers = {
            "x-reflex-latency-ms": f"{ms:.1f}",
            "x-reflex-queue-ms": f"{wait_ms:.1f}",
        }
        # exclude_none: optional fields that are unset stay out of the payload.
        return JSONResponse(resp.model_dump(exclude_none=True), headers=headers)

    return app


def _sglang_backend(args):
    """`--backend sglang`: the same prompt and readout, computed by an SGLang server.

    Only the flags that survive the move are honoured. Anything that needs the weights in
    this process (an adapter, prompt ensembles, a local device) is rejected here rather
    than quietly ignored.
    """
    from transformers import AutoTokenizer

    from reflex.backends.sglang import SGLangBackend
    from reflex.engine import _load_texts
    from reflex.prompt import PromptFormat
    from reflex.readout import Calibration

    for flag, value in (("--adapter", args.adapter), ("--ensemble", args.ensemble)):
        if value:
            raise SystemExit(f"{flag} is not supported by --backend sglang")
    if args.device not in (None, "cuda"):
        raise SystemExit(
            "--device applies to the in-process model; --backend sglang holds no weights "
            "(the device is SGLang's, set when you launch its server)"
        )

    tok = AutoTokenizer.from_pretrained(args.model)
    template = tok.chat_template or ""
    fmt = PromptFormat(
        chat=bool(template),
        no_think="enable_thinking" in template,
        style=args.prompt_style,
        texts=_load_texts(args.prompt_texts),
    )
    backend = SGLangBackend(
        args.sglang_url,
        tokenizer=tok,
        fmt=fmt,
        calibration=Calibration.load(args.calibration),
        model_name=args.served_name or args.model,
        default_permutations=args.permutations,
        max_concurrent_calls=args.sglang_concurrency,
    )
    log.info("sglang backend: %s serving %s", args.sglang_url, backend.model_name)
    return backend


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default=None,
        help="model id; default Qwen/Qwen3.5-4B, or Qwen/Qwen3-4B for --backend mlx",
    )
    ap.add_argument(
        "--adapter", default=None, help="LoRA adapter: a reflex-calibrate output dir or a hub id"
    )
    ap.add_argument(
        "--calibration",
        default=None,
        help="calibration.json (default: the one next to the adapter)",
    )
    ap.add_argument(
        "--api-key", default=None, help="require this bearer key on /v1/* (or env REFLEX_API_KEY)"
    )
    ap.add_argument(
        "--served-name", default=None, help="name reported in responses (default: the model id)"
    )
    ap.add_argument("--prompt-style", default="markdown", choices=["markdown", "compact"])
    ap.add_argument(
        "--prompt-texts",
        default=None,
        help="prompt.json from reflex-optimize (instruction wording)",
    )
    ap.add_argument(
        "--permutations",
        type=int,
        default=1,
        help="default option-order averaging for requests that do not set it "
        "(2 halves letter-position bias at 2x branch cost)",
    )
    ap.add_argument(
        "--stable",
        action="store_true",
        help="take adapter/calibration/prompt defaults from serving/stable.json "
        "(the configuration the `stable` git tag recommends); explicit flags still win",
    )
    ap.add_argument(
        "--ensemble", default=None, help="prompt-ensemble variants json (reflex.ensemble)"
    )
    ap.add_argument(
        "--require-fast-kernels",
        action="store_true",
        help="refuse to start if any op fell back to its reference PyTorch implementation "
        "(a missing flash-linear-attention or causal-conv1d). Off by default; turn it on "
        "for benchmark runs, where a silent fallback costs an order of magnitude",
    )
    ap.add_argument(
        "--backend",
        default="transformers",
        choices=["transformers", "torch", "sglang", "mlx"],
        help="transformers (alias torch): load the model in this process (the default). "
        "sglang: read the same label logits off an SGLang server over HTTP "
        "(reflex.backends.sglang). mlx: load the model on the Apple Silicon GPU "
        "through MLX (reflex.mlx_engine)",
    )
    ap.add_argument(
        "--sglang-concurrency",
        type=int,
        default=8,
        help="most /generate calls in flight against SGLang at once, across all callers. "
        "One reflex request is one batched call, whatever its question count, so this is "
        "the number of requests reflex lets through; the queue belongs here, where it is "
        "visible, rather than as open connections there (default 8)",
    )
    ap.add_argument(
        "--sglang-url",
        default="http://127.0.0.1:30000",
        help="where the SGLang server listens, for --backend sglang",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--max-pack-tokens", type=int, default=8192)
    ap.add_argument(
        "--max-branch-tokens",
        type=int,
        default=4096,
        help="longest single question (instructions plus options) the server will read, "
        "in tokens. Anything longer is refused rather than cut. Raise it for suites that "
        "put a whole document in one question; the ceiling is the model's context window",
    )
    ap.add_argument(
        "--cache-entries", type=int, default=16, help="max cached state prefixes (mlx backend)"
    )
    ap.add_argument(
        "--cache-gb", type=float, default=6.0, help="state-prefix KV budget in GiB (mlx backend)"
    )
    ap.add_argument(
        "--warmup",
        default=None,
        help="SystemOne request JSON to evaluate before accepting traffic",
    )
    ap.add_argument(
        "--dtype",
        default=None,
        choices=["bfloat16", "float16", "float32"],
        help="torch dtype (default bfloat16); mlx reads the checkpoint's own",
    )
    ap.add_argument(
        "--device",
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="torch device_map (default cuda)",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    import uvicorn

    explicit_model = args.model is not None
    if args.model is None:
        # Each backend keeps the default it shipped with.
        args.model = "Qwen/Qwen3-4B" if args.backend == "mlx" else "Qwen/Qwen3.5-4B"

    if args.backend == "mlx":
        if args.adapter or args.dtype or args.device:
            ap.error("--adapter, --dtype and --device apply only to the torch backend")
        # MLXEngine takes no server-side defaults for these, so refuse them rather than
        # answer differently than asked. See reflex.mlx_engine.MLXEngine.load.
        torch_only = {
            "--stable": args.stable,
            "--ensemble": args.ensemble,
            "--prompt-texts": args.prompt_texts,
            "--prompt-style": args.prompt_style != "markdown",
            "--max-branch-tokens": args.max_branch_tokens != 4096,
        }
        given = [flag for flag, is_set in torch_only.items() if is_set]
        if given:
            ap.error(f"{', '.join(given)} not supported by --backend mlx")
        from reflex.mlx_engine import MLXEngine

        engine = MLXEngine.load(
            args.model,
            calibration_path=args.calibration,
            max_pack_tokens=args.max_pack_tokens,
            cache_entries=args.cache_entries,
            cache_bytes=int(args.cache_gb * 1024**3),
            default_permutations=args.permutations,
        )
    elif args.backend == "sglang":
        if args.require_fast_kernels:
            raise SystemExit(
                "--require-fast-kernels applies to the in-process model; --backend sglang "
                "holds no weights here (check the kernels on the SGLang server instead)"
            )
        engine = _sglang_backend(args)
    else:
        from reflex.engine import Engine
        from reflex.kernels import describe, kernel_report, require_fast_kernels

        if args.stable:
            from reflex.serving import engine_kwargs, load_stable

            kw = engine_kwargs(
                load_stable(),
                adapter_path=args.adapter,
                calibration_path=args.calibration,
                prompt_texts=args.prompt_texts,
                prompt_style=args.prompt_style if args.prompt_style != "markdown" else None,
                default_permutations=args.permutations if args.permutations != 1 else None,
            )
            if explicit_model:
                kw["model_id"] = args.model
        else:
            kw = {
                "model_id": args.model,
                "calibration_path": args.calibration,
                "adapter_path": args.adapter,
                "default_permutations": args.permutations,
                "prompt_style": args.prompt_style,
                "prompt_texts": args.prompt_texts,
            }
        engine = Engine.load(
            dtype=getattr(torch, args.dtype or "bfloat16"),
            device=args.device or "cuda",
            max_pack_tokens=args.max_pack_tokens,
            max_branch_tokens=args.max_branch_tokens,
            ensemble=args.ensemble,
            **kw,
        )
        report = kernel_report(engine.model)
        log.info("kernels: %s", describe(report))
        if args.require_fast_kernels:
            require_fast_kernels(report)

    if args.served_name:
        engine.model_name = args.served_name
    if args.warmup:
        engine.answer(SystemOneRequest.model_validate_json(Path(args.warmup).read_text()))
    uvicorn.run(
        create_app(engine, api_key=args.api_key),
        host=args.host,
        port=args.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
