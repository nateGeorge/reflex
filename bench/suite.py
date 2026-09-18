"""Standard Reflex decision-model benchmark.

Workloads (accuracy + latency each):
  smoke    14 auto-think routine/deep prompts x option orders (choice, 2 options)
  banking  Banking77 sample: 1 true intent + 9 distractors (choice, 10 options)
  sst2     SST-2 sample (noul, positive/negative)
  stars    Amazon reviews sample, 1-5 stars (score, 5 ordered levels)
  batch    1 state x 8 mixed questions in ONE request (multi-question throughput)

Usage:
  uv run python bench/suite.py --model $HOME/.local/share/reflex-mlx/models/qwen3-4b-4bit \
      --out runs/bench-qwen3-4b.json [--n 200] [--backend mlx]

Comparisons across models need identical --n and the same dataset revisions.
Fit calibration on train splits, report on test/validation splits.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

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


def run_smoke(engine):
    import sys

    sys.path.insert(0, str(Path.home() / ".local/share/reflex-mlx"))
    from smoke import CASES, request_for

    rows = []
    engine.answer(SystemOneRequest(**request_for("Warmup: show the current branch name.")))
    for reverse in (False, True):
        for name, expected, text in CASES:
            for attempt in ("first", "repeat"):
                resp, ms = timed(engine, SystemOneRequest(**request_for(text, reverse)))
                ans = resp.answers["thinking_need"]
                rows.append({"case": name, "expected": expected, "reverse": reverse,
                             "attempt": attempt, "choice": ans.choice,
                             "correct": ans.choice == expected, "ms": round(ms, 1)})
    return rows


def run_banking(engine, n: int, seed: int):
    from datasets import load_dataset

    ds = load_dataset("banking77", split="validation")
    rng = random.Random(seed)
    labels = ds.features["label"].names
    idx = rng.sample(range(len(ds)), min(n, len(ds)))
    rows = []
    for i in idx:
        item = ds[int(i)]
        true = labels[item["label"]]
        distract = rng.sample([l for l in labels if l != true], 9)
        options = [true] + distract
        rng.shuffle(options)
        criteria = {o: o.replace("_", " ") for o in options}
        req = SystemOneRequest(state=item["text"], questions={
            "intent": {"type": "choice", "instructions": "What is the customer's intent?",
                       "criteria": criteria}})
        resp, ms = timed(engine, req)
        got = resp.answers["intent"].choice
        # choice answer is the winning option key; resolve via probabilities order
        probs = resp.answers["intent"].probabilities or {}
        got_key = max(probs, key=probs.get) if probs else got
        rows.append({"expected": true, "got": got_key, "correct": got_key == true,
                     "ms": round(ms, 1)})
    return rows


def run_sst2(engine, n: int, seed: int):
    from datasets import load_dataset

    ds = load_dataset("glue", "sst2", split="validation")
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), min(n, len(ds)))
    rows = []
    for i in idx:
        item = ds[int(i)]
        expected = item["label"] == 1
        req = SystemOneRequest(state=item["sentence"], questions={
            "pos": {"type": "noul", "instructions": "Is this review positive?",
                    "criteria": {"true": "positive sentiment", "false": "negative sentiment"}}})
        resp, ms = timed(engine, req)
        rows.append({"expected": expected, "got": resp.answers["pos"].noul,
                     "correct": (resp.answers["pos"].noul >= 0.5) == expected,
                     "ms": round(ms, 1)})
    return rows


def run_stars(engine, n: int, seed: int):
    from datasets import load_dataset

    ds = load_dataset("amazon_reviews_multi", "en", split="test")
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), min(n, len(ds)))
    levels = ["1 star: very bad", "2 stars: bad", "3 stars: okay",
              "4 stars: good", "5 stars: excellent"]
    rows = []
    for i in idx:
        item = ds[int(i)]
        expected = int(item["stars"]) - 1
        req = SystemOneRequest(state=item["review_body"][:1000], questions={
            "rating": {"type": "score", "instructions": "What star rating is this review?",
                       "criteria": levels}})
        resp, ms = timed(engine, req)
        ans = resp.answers["rating"]
        got = ans.score if ans.score is not None else -1
        rows.append({"expected": expected, "got": got,
                     "correct": round(got) == expected, "ms": round(ms, 1)})
    return rows


def run_batch(engine):
    rows = []
    for state in BATCH_STATES:
        req = SystemOneRequest(state=state, questions=BATCH_QUESTIONS)
        resp, ms = timed(engine, req)
        per_q = ms / len(BATCH_QUESTIONS)
        # baseline: same questions as separate requests
        t0 = time.perf_counter()
        for qid, q in BATCH_QUESTIONS.items():
            engine.answer(SystemOneRequest(state=state, questions={qid: q}))
        separate_ms = (time.perf_counter() - t0) * 1000
        rows.append({"state": state[:40], "n_questions": len(BATCH_QUESTIONS),
                     "one_request_ms": round(ms, 1),
                     "separate_requests_ms": round(separate_ms, 1),
                     "ms_per_question_batched": round(per_q, 1)})
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
    ap.add_argument("--backend", default="mlx", choices=["mlx", "torch"])
    ap.add_argument("--skip", default="")
    args = ap.parse_args()
    skip = set(args.skip.split(",")) if args.skip else set()

    engine = load_engine(args.model, args.backend)
    out = {"model": args.model, "backend": args.backend, "n": args.n, "seed": args.seed}
    for name, fn in [("smoke", lambda: run_smoke(engine)),
                     ("banking", lambda: run_banking(engine, args.n, args.seed)),
                     ("sst2", lambda: run_sst2(engine, args.n, args.seed)),
                     ("stars", lambda: run_stars(engine, args.n, args.seed)),
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
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
