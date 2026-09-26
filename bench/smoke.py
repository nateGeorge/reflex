"""The auto-think workload: the question pi-auto-think actually sends.

This is the one workload that measures the product decision, so it mirrors
`src/levels.ts` in nateGeorge/pi-auto-think rather than a public dataset:
`INSTRUCTION` and `CRITERIA` are copied from `LEVEL_INSTRUCTION` and
`LEVEL_CRITERIA`, and `LEVELS` is what `choiceLevels()` offers a model with no
`thinkingLevelMap`. Re-copy both when the extension's wording changes -- that
wording is load-bearing, not cosmetic. On this workload the anchored wording
moved deep prompts from 0/28 correct to 26/28.

This file used to live outside the repo, at
`~/.local/share/reflex-mlx/smoke.py`, so the smoke leg could not be re-run once
that directory was gone. It is vendored here on purpose: the workload is the
suite's source of truth and belongs in git.
"""

INSTRUCTION = (
    "Pick the thinking level that matches the step. Do not pick a lower level just to save tokens."
)

CRITERIA = {
    "off": "No thinking needed. A value already known, a greeting, or a result already in hand.",
    "minimal": "One mechanical step with no decisions: run a command, open a file.",
    "low": (
        "A small, local, easily reversed change or lookup: read a file, list a "
        "directory, rename, fix a typo."
    ),
    "medium": (
        "Several dependent steps, or new code in one place: write a function, add "
        "a test, add a flag, follow a known pattern."
    ),
    "high": (
        "Any step where being wrong is costly or the cause is unknown: debugging, "
        "root-causing, designing, refactoring existing code, changing shared or "
        "security-sensitive logic, migrations."
    ),
}

# The five levels the extension offers a model with no thinkingLevelMap, in the
# order it sends them (cheapest first).
LEVELS = ("off", "minimal", "low", "medium", "high")

QUESTION = {
    "type": "choice",
    "instructions": INSTRUCTION,
    "criteria": {level: CRITERIA[level] for level in LEVELS},
}

# What counts as a right answer. Routine work has to stay cheap; a deep prompt
# has to reach `high`, because parking a race condition on `medium` is exactly
# the under-rating this workload exists to catch.
ACCEPT = {
    "routine": {"off", "minimal", "low"},
    "deep": {"high"},
}

CASES = [
    ("read", "routine", "Read the README and show the installation command."),
    ("status", "routine", "Show git status and the last three commits."),
    ("files", "routine", "List the Python files in the tests directory."),
    (
        "lookup",
        "routine",
        "What port does the local server listen on? Check the config.",
    ),
    (
        "summary",
        "routine",
        "Summarize this sentence: The job runs daily and writes a CSV file.",
    ),
    (
        "search",
        "routine",
        "Find the definition of calculate_total and show its file path.",
    ),
    (
        "race",
        "deep",
        (
            "Diagnose a race condition where two workers charge the same invoice. "
            "Design an idempotent fix and tests."
        ),
    ),
    (
        "architecture",
        "deep",
        (
            "Design a multi-tenant job scheduler with fairness, retries, and crash "
            "recovery. Explain the consistency tradeoffs."
        ),
    ),
    (
        "memory",
        "deep",
        (
            "Investigate why our service memory grows after each request. Form "
            "hypotheses, trace object retention, and fix the leak."
        ),
    ),
    (
        "refactor",
        "deep",
        (
            "Refactor the payment state machine without breaking retries or refunds. "
            "Plan the migration and regression tests."
        ),
    ),
    (
        "flaky",
        "deep",
        (
            "The integration test passes alone but fails in parallel. Find the "
            "shared-state bug and implement a robust fix."
        ),
    ),
    (
        "rootcause",
        "deep",
        (
            "Requests sometimes disappear after a leader failover. Reconstruct the "
            "failure sequence and fix the data-loss bug."
        ),
    ),
    (
        "longread",
        "routine",
        (
            "Only extract the port and cache size from this configuration summary. "
            "Do not diagnose or change anything. "
            + "The service logs errors and supports retries. " * 35
            + "Port: 8008. Cache size: 256."
        ),
    ),
    (
        "longdebug",
        "deep",
        (
            "Diagnose and fix this intermittent data loss. "
            + "Worker A starts a write. Worker B takes over after a timeout. Both "
            "report success, but the row is absent. "
            * 20
            + "Find the race, propose a safe transaction design, and add a "
            "regression test."
        ),
    ),
]


def request_for(text, reverse=False):
    """Build the request body for one case, optionally with the options reversed.

    Reversing is a position-bias check: a model that only reads the first
    criterion should answer differently here. `QUESTION` itself is never
    mutated.
    """
    question = dict(QUESTION)
    if reverse:
        question["criteria"] = dict(reversed(list(QUESTION["criteria"].items())))
    return {"state": text, "questions": {"thinking_need": question}}
