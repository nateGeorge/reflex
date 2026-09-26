# Reflex benchmark suite

Standard decision-model benchmark. Every candidate model runs the same
workloads with the same `--n` and `--seed`, so numbers compare directly.

## Workloads

| Name      | Type                 | Data                                                              |
| --------- | -------------------- | ----------------------------------------------------------------- |
| `smoke`   | `choice`, 2 options  | 14 auto-think routine/deep prompts x option orders x first/repeat |
| `banking` | `choice`, 10 options | Banking77 sample (1 true intent + 9 distractors)                  |
| `sst2`    | `noul`               | SST-2 validation sample                                           |
| `stars`   | `score`, 5 levels    | Amazon reviews (en) test sample, 1-5 stars                        |
| `batch`   | mixed, 8 questions   | 1 state x 8 questions in ONE request vs separate requests         |
| `prefix`  | mixed, 8 questions   | 1 long state (~20k tok) x 8 questions, cold vs warm cache reuse   |

Each workload reports accuracy plus median/p95 latency. `batch` measures
multi-question throughput: the reason decision models beat chat models.
`prefix` measures the other half of that argument -- one long state prefix is
read once and reused by every branch and by the next request. Its row reports
`cold_ms`, `warm_ms`, `speedup`, `state_tokens`, `cache_entries`, and
`answers_identical`; `answers_identical` must be `true`, and a `false` there is a
state-prefix cache bug, not a threshold to tune.

## Run

```bash
uv run python bench/suite.py --model $HOME/.local/share/reflex-mlx/models/qwen3-4b-4bit \
    --out runs/bench-qwen3-4b.json
```

Options: `--n 200` (dataset sample size), `--seed 7`, `--backend mlx|torch`,
`--prefix-tokens 20000`, `--max-pack-tokens 32768` (match the server's flag),
`--skip banking,sst2` (e.g. for quick smoke-only runs).

## Rules

- Fit calibration on train splits, report on test/validation splits.
- The smoke suite stays the source of truth for the auto-think workload;
  public datasets measure general decision accuracy, not routine-vs-deep.
- Record the evidence JSON path alongside any reported number.
