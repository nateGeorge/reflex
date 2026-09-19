"""Standard Reflex decision-model benchmark.

Workloads (accuracy + latency each):
  smoke    14 auto-think routine/deep prompts x option orders (choice, 2 options)
  banking  Banking77 sample: 1 true intent + 25 distractors (choice, 26 options)
  sst2     SST-2 sample (noul, positive/negative)
  stars    Yelp review sample, 1-5 stars (score, 5 ordered levels)
  batch    1 state x 8 mixed questions in ONE request (multi-question throughput)

Datasets load from raw parquet via snapshot_download. HF_HUB_DISABLE_XET=1
works around xet-backed repos. No `datasets` dependency.

Usage:
  uv run python bench/suite.py --model $HOME/.local/share/reflex-mlx/models/qwen3-4b-4bit \
      --out runs/bench-qwen3-4b.json [--n 200] [--backend mlx] [--permutations 2]

Comparisons across models need identical --n/--seed/--permutations and the
same dataset revisions. Fit calibration on train splits, report on test splits.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from reflex.schema import SystemOneRequest

BATCH_QUESTIONS = {
    "need": {"type": "noul", "instructions": "Does the request need deep reasoning?"},
    "team": {
        "type": "choice",
        "instructions": "Which team handles this?",
        "criteria": {
            "billing": "charges, refunds, invoices",
            "technical": "bugs, outages, errors",
            "sales": "pricing, upgrades, plans",
            "trust": "spam, fraud, abuse",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["can wait a week", "handle today", "blocked right now"],
    },
    "angry": {"type": "noul", "instructions": "Is the customer angry?"},
    "refund": {"type": "noul", "instructions": "Is a refund requested?"},
    "bug": {"type": "noul", "instructions": "Is a software bug reported?"},
    "channel": {
        "type": "choice",
        "instructions": "Which channel is this for?",
        "criteria": {"self": "customer can self-serve", "agent": "needs a human agent"},
    },
    "complexity": {
        "type": "score",
        "instructions": "How complex is the request?",
        "criteria": ["single fact", "short task", "multi-step project"],
    },
}

BATCH_STATES = [
    "I was charged twice for my subscription this month, please fix it now, this is ridiculous.",
    "Where do I update my credit card expiry date?",
    "Our webhook endpoint returns 500s after the deploy. Error rate is 40% and climbing.",
]


def _parquet_rows(repo: str, pattern: str):
    """Download one parquet glob from a dataset repo, return list of dicts."""
    import glob as _glob

    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    base = snapshot_download(repo, repo_type="dataset", allow_patterns=[pattern])
    files = sorted(_glob.glob(f"{base}/{pattern}"))
    rows = []
    for f in files:
        rows.extend(pq.read_table(f).to_pylist())
    return rows


def _banking_label_names():
    import json as _json

    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    base = snapshot_download(
        "banking77", repo_type="dataset", allow_patterns=["data/test*.parquet"]
    )
    import glob as _glob

    schema = pq.read_schema(sorted(_glob.glob(f"{base}/data/test*.parquet"))[0])
    meta = _json.loads(schema.metadata[b"huggingface"])
    return meta["info"]["features"]["label"]["names"]


def load_engine(model: str, backend: str):
    if backend == "mlx":
        from reflex.mlx_engine import MLXEngine

        return MLXEngine.load(model)
    from reflex import Engine

    return Engine.load(model)


def timed(engine, req: SystemOneRequest):
    start = time.perf_counter()
    resp = engine.answer(req)
    return resp, (time.perf_counter() - start) * 1000


def run_smoke(engine, perms: int):
    import sys

    sys.path.insert(0, str(Path.home() / ".local/share/reflex-mlx"))
    from smoke import CASES, request_for

    rows = []
    engine.answer(SystemOneRequest(**request_for("Warmup: show the current branch name.")))
    for reverse in (False, True):
        for name, expected, text in CASES:
            for attempt in ("first", "repeat"):
                body = request_for(text, reverse)
                req = SystemOneRequest(**body, permutations=perms)
                resp, ms = timed(engine, req)
                ans = resp.answers["thinking_need"]
                rows.append({"case": name, "expected": expected, "reverse": reverse,
                             "attempt": attempt, "choice": ans.choice,
                             "correct": ans.choice == expected, "ms": round(ms, 1)})
    return rows


def run_banking(engine, n: int, seed: int, perms: int):
    rows = _parquet_rows("banking77", "data/test*.parquet")
    labels = _banking_label_names()
    rng = random.Random(seed)
    idx = rng.sample(range(len(rows)), min(n, len(rows)))
    out = []
    for i in idx:
        item = rows[i]
        true = labels[item["label"]]
        distract = rng.sample([l for l in labels if l != true], 25)
        options = [true] + distract
        rng.shuffle(options)
        criteria = {o: o.replace("_", " ") for o in options}
        req = SystemOneRequest(
            state=item["text"],
            questions={"intent": {
                "type": "choice",
                "instructions": "What is the customer's intent? Respond with only the letter.",
                "criteria": criteria}},
            permutations=perms,
        )
        resp, ms = timed(engine, req)
        probs = resp.answers["intent"].probabilities or {}
        got = max(probs, key=probs.get) if probs else None
        out.append({"expected": true, "got": got, "correct": got == true,
                    "ms": round(ms, 1)})
    return out


def run_sst2(engine, n: int, seed: int, perms: int):
    rows = _parquet_rows("stanfordnlp/sst2", "data/validation*.parquet")
    rng = random.Random(seed)
    idx = rng.sample(range(len(rows)), min(n, len(rows)))
    out = []
    for i in idx:
        item = rows[i]
        expected = bool(item["label"])
        req = SystemOneRequest(
            state=item["sentence"],
            questions={"pos": {
                "type": "noul",
                "instructions": "Is this review positive?",
                "criteria": {"true": "positive sentiment", "false": "negative sentiment"}}},
            permutations=perms,
        )
        resp, ms = timed(engine, req)
        out.append({"expected": expected, "got": resp.answers["pos"].noul,
                    "correct": (resp.answers["pos"].noul >= 0.5) == expected,
                    "ms": round(ms, 1)})
    return out


def run_stars(engine, n: int, seed: int, perms: int):
    rows = _parquet_rows("yelp_review_full", "yelp_review_full/test*.parquet")
    rng = random.Random(seed)
    idx = rng.sample(range(len(rows)), min(n, len(rows)))
    levels = ["1 star: very bad", "2 stars: bad", "3 stars: okay",
              "4 stars: good", "5 stars: excellent"]
    out = []
    for i in idx:
        item = rows[i]
        expected = int(item["label"])
        req = SystemOneRequest(
            state=str(item["text"])[:1000],
            questions={"rating": {
                "type": "score",
                "instructions": "What star rating is this review? Respond with only the letter.",
                "criteria": levels}},
            permutations=perms,
        )
        resp, ms = timed(engine, req)
        ans = resp.answers["rating"]
        got = ans.score if ans.score is not None else -1
        out.append({"expected": expected, "got": got,
                    "correct": round(got) == expected, "ms": round(ms, 1)})
    return out


def run_batch(engine):
    rows = []
    for state in BATCH_STATES:
        req = SystemOneRequest(state=state, questions=BATCH_QUESTIONS)
        resp, ms = timed(engine, req)
        t0 = time.perf_counter()
        for qid, q in BATCH_QUESTIONS.items():
            engine.answer(SystemOneRequest(state=state, questions={qid: q}))
        separate_ms = (time.perf_counter() - t0) * 1000
        rows.append({"state": state[:40], "n_questions": len(BATCH_QUESTIONS),
                     "one_request_ms": round(ms, 1),
                     "separate_requests_ms": round(separate_ms, 1),
                     "ms_per_question_batched": round(ms / len(BATCH_QUESTIONS), 1)})
    return rows


def summarize(rows, key="correct"):
    acc = sum(r[key] for r in rows) / len(rows) if rows else 0
    ms = sorted(r["ms"] for r in rows if "ms" in r)
    med = statistics.median(ms) if ms else 0
    p95 = ms[min(len(ms) - 1, int(len(ms) * 0.95))] if ms else 0
    return {"n": len(rows), "accuracy": round(acc, 4),
            "median_ms": round(med, 1), "p95_ms": round(p95, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--permutations", type=int, default=1)
    ap.add_argument("--backend", default="mlx", choices=["mlx", "torch"])
    ap.add_argument("--skip", default="")
    args = ap.parse_args()
    skip = set(args.skip.split(",")) if args.skip else set()

    engine = load_engine(args.model, args.backend)
    out = {"model": args.model, "backend": args.backend, "n": args.n,
           "seed": args.seed, "permutations": args.permutations}
    for name, fn in [("smoke", lambda: run_smoke(engine, args.permutations)),
                     ("banking", lambda: run_banking(engine, args.n, args.seed, args.permutations)),
                     ("sst2", lambda: run_sst2(engine, args.n, args.seed, args.permutations)),
                     ("stars", lambda: run_stars(engine, args.n, args.seed, args.permutations)),
                     ("batch", lambda: run_batch(engine))]:
        if name in skip:
            continue
        print(f"[{name}]...", flush=True)
        rows = fn()
        out[name] = {"rows": rows}
        if name != "batch":
            out[name]["summary"] = summarize(rows)
            print(f"[{name}] {out[name]['summary']}", flush=True)
        else:
            print(f"[{name}] {rows}", flush=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        if args.backend == "mlx":
            # MLX Metal buffers grow to a high-water mark and are not
            # returned promptly. Release between legs so long suites plus
            # anything else on the machine do not pile up into swap.
            import mlx.core as mx

            mx.metal.clear_cache()
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
