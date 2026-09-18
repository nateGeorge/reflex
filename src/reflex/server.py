"""FastAPI server exposing the TypeSafe-compatible endpoint `POST /v1/systemone`.

    reflex-serve --model Qwen/Qwen3-8B --port 8008

Then any client written for Jev can point at http://localhost:8008 instead.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from reflex.schema import SystemOneRequest, SystemOneResponse

log = logging.getLogger("reflex.server")


def create_app(engine) -> FastAPI:
    app = FastAPI(title="reflex", version="0.1.0")

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "model": engine.model_name, "device": str(engine.device)}

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    def systemone(req: SystemOneRequest):
        t0 = time.perf_counter()
        try:
            resp = engine.answer(req)
        except ValueError as e:  # bad labels / too long
            raise HTTPException(status_code=422, detail=str(e))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise HTTPException(status_code=529, detail="overloaded: request too large for GPU")
        ms = (time.perf_counter() - t0) * 1000
        log.info(
            "%d questions, %d state tok (%s), %d q tok, %.0f ms",
            len(req.questions),
            resp.usage.state_tokens,
            "hit" if resp.usage.state_cache_hit else "miss",
            resp.usage.question_tokens,
            ms,
        )
        return JSONResponse(resp.model_dump(), headers={"x-reflex-latency-ms": f"{ms:.1f}"})

    return app


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir from reflex-calibrate")
    ap.add_argument("--calibration", default=None, help="calibration.json (temperatures)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--max-pack-tokens", type=int, default=8192)
    ap.add_argument("--backend", choices=["torch", "mlx"], default="torch")
    ap.add_argument("--warmup", help="SystemOne request JSON to evaluate before accepting traffic")
    ap.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default=None, help="torch device_map (cuda, mps, cpu)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    import uvicorn

    if args.backend == "mlx":
        if args.adapter or args.dtype or args.device:
            ap.error("--adapter, --dtype and --device apply only to the torch backend")
        from reflex.mlx_engine import MLXEngine

        engine = MLXEngine.load(
            args.model or "Qwen/Qwen3-4B",
            calibration_path=args.calibration,
            max_pack_tokens=args.max_pack_tokens,
        )
    else:
        from reflex.engine import Engine

        engine = Engine.load(
            args.model or "Qwen/Qwen3.5-4B",
            device=args.device or "cuda",
            dtype=getattr(torch, args.dtype or "bfloat16"),
            calibration_path=args.calibration,
            adapter_path=args.adapter,
            max_pack_tokens=args.max_pack_tokens,
        )
    if args.warmup:
        engine.answer(SystemOneRequest.model_validate_json(Path(args.warmup).read_text()))
    uvicorn.run(create_app(engine), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
