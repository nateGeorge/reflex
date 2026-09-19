"""Fit per-kind temperature calibration on train splits.

Temperature scaling from stored probabilities is exact up to an additive
constant: z = log(p) recovers logits, and softmax is translation-invariant.
Grid-searches t in [0.25, 4.0] minimizing NLL per question kind, writes
`evidence/cal-<name>.json` in the `Calibration.save` format.

Usage:
  uv run python bench/calibrate.py --model .../models/gemma-3n-e4b-4bit \
      --out evidence/cal-gemma.json [--n 200] [--permutations 1]
"""

from __future__ import annotations

import argparse
import json
import math
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from suite import (
    _banking_label_names,
    _parquet_rows,
    load_engine,
    timed,
)

from reflex.schema import SystemOneRequest


def collect(engine, workload: str, n: int, seed: int, perms: int):
    """Return list of (kind, prob_vector, expected_index)."""
    import random

    rng = random.Random(seed)
    out = []
    if workload == "banking":
        rows = _parquet_rows("banking77", "data/train*.parquet")
        labels = _banking_label_names()
        for i in rng.sample(range(len(rows)), min(n, len(rows))):
            item = rows[i]
            true = labels[item["label"]]
            options = [true] + rng.sample([l for l in labels if l != true], 25)
            rng.shuffle(options)
            req = SystemOneRequest(
                state=item["text"],
                questions={"intent": {
                    "type": "choice",
                    "instructions": "What is the customer's intent? Respond with only the letter.",
                    "criteria": {o: o.replace("_", " ") for o in options}}},
                permutations=perms,
            )
            resp, _ = timed(engine, req)
            probs = resp.answers["intent"].probabilities or {}
            keys = list(probs.keys())
            out.append(("choice", [probs[k] for k in keys], keys.index(true)))
    elif workload == "sst2":
        rows = _parquet_rows("stanfordnlp/sst2", "data/train*.parquet")
        for i in rng.sample(range(len(rows)), min(n, len(rows))):
            item = rows[i]
            expected = bool(item["label"])
            req = SystemOneRequest(
                state=item["sentence"],
                questions={"pos": {
                    "type": "noul",
                    "instructions": "Is this review positive?",
                    "criteria": {"true": "positive sentiment",
                                 "false": "negative sentiment"}}},
                permutations=perms,
            )
            resp, _ = timed(engine, req)
            p = resp.answers["pos"].noul
            out.append(("noul", [p, 1 - p], 0 if expected else 1))
    elif workload == "stars":
        rows = _parquet_rows("yelp_review_full", "yelp_review_full/train*.parquet")
        levels = ["1 star: very bad", "2 stars: bad", "3 stars: okay",
                  "4 stars: good", "5 stars: excellent"]
        for i in rng.sample(range(len(rows)), min(n, len(rows))):
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
            resp, _ = timed(engine, req)
            ans = resp.answers["rating"]
            probs = ans.probabilities or {}
            keys = sorted(probs.keys())
            out.append(("score", [probs[k] for k in keys], keys.index(str(expected))))
    return out


def nll(probs, idx, t):
    z = [math.log(max(p, 1e-12)) / t for p in probs]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    return math.log(sum(e)) - math.log(e[idx])


def fit(samples):
    grid = [round(0.25 + 0.05 * i, 2) for i in range(76)]
    by_kind: dict[str, list] = {}
    for kind, probs, idx in samples:
        by_kind.setdefault(kind, []).append((probs, idx))
    temps = {}
    for kind, rows in by_kind.items():
        best = min(grid, key=lambda t: sum(nll(p, i, t) for p, i in rows))
        base = sum(nll(p, i, 1.0) for p, i in rows) / len(rows)
        fit_nll = sum(nll(p, i, best) for p, i in rows) / len(rows)
        temps[kind] = {"temperature": best, "n_train": len(rows),
                       "nll_t1": round(base, 4), "nll_fit": round(fit_nll, 4)}
    return temps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--permutations", type=int, default=1)
    ap.add_argument("--backend", default="mlx", choices=["mlx", "torch"])
    ap.add_argument("--workloads", default="banking,sst2,stars")
    args = ap.parse_args()
    engine = load_engine(args.model, args.backend)
    samples = []
    for w in args.workloads.split(","):
        print(f"[{w}] collecting train...", flush=True)
        samples.extend(collect(engine, w, args.n, args.seed, args.permutations))
    result = {"model": args.model, "temperature": {},
              "detail": (detail := fit(samples))}
    result["temperature"] = {k: v["temperature"] for k, v in detail.items()}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
