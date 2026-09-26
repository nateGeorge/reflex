# Reflex benchmark suite

Standard decision-model benchmark. Every candidate model runs the same
workloads with the same `--n` and `--seed`, so numbers compare directly.

## Workloads

| Name      | Type                 | Data                                                              |
| --------- | -------------------- | ----------------------------------------------------------------- |
| `smoke`   | `choice`, 5 options  | 14 auto-think routine/deep prompts x option orders x first/repeat |
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
uv run python bench/suite.py --model Qwen/Qwen3-4B \
    --out runs/bench-qwen3-4b.json
```

Options: `--n 200` (dataset sample size), `--seed 7`, `--backend mlx|torch`,
`--prefix-tokens 20000`, `--max-pack-tokens 32768` (match the server's flag),
`--skip banking,sst2` (e.g. for quick smoke-only runs).

## Rules

- Fit calibration on train splits, report on test/validation splits.
- The smoke suite stays the source of truth for the auto-think workload;
  public datasets measure general decision accuracy, not routine-vs-deep.
- That workload lives in `bench/smoke.py` and is a copy of the question
  pi-auto-think sends: the same instruction, the same criteria, and the five
  levels `choiceLevels()` offers. It used to live outside the repo, which is why
  the smoke leg could not be re-run after that directory was deleted. Re-copy it
  when the extension's wording changes -- the wording moves the score, so a
  stale copy measures a question nobody asks.
- A `deep` case counts as correct only at `high`; `medium` there is the
  under-rating this workload exists to catch, not a near miss.
- For `smoke`, read the `production` summary, not `summary`. Only the forward
  rows are the criteria order pi-auto-think sends; the reversed rows probe order
  robustness and score lower (routine prompts drift up a level when the criteria
  run high-to-low), so the average understates production.
- Do not reorder the criteria to chase that probe. Six orders were measured on
  this workload and the shipped cheapest-to-most-expensive order scored best
  (13/14); `high` first, `off` last, and dropping `off` were all worse.
- Record the evidence JSON path alongside any reported number.
