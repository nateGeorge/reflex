"""Guards on the auto-think workload and the smoke leg that runs it.

`bench/smoke.py` is the only workload that measures the real product decision,
and it is a copy of what pi-auto-think sends. These tests pin the shape so the
copy cannot drift silently: a workload that asks a different question than
production measures nothing. `run_smoke` is exercised with a fake engine, so the
leg that used to import a module outside the repo is covered without a GPU.
"""

import importlib.util
from pathlib import Path

_BENCH = Path(__file__).resolve().parents[1] / "bench"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = _load(_BENCH / "smoke.py", "smoke_workload")
suite = _load(_BENCH / "suite.py", "bench_suite")

ACCEPT = smoke.ACCEPT
CASES = smoke.CASES
CRITERIA = smoke.CRITERIA
INSTRUCTION = smoke.INSTRUCTION
LEVELS = smoke.LEVELS
QUESTION = smoke.QUESTION


class _Answer:
    def __init__(self, choice):
        self.choice = choice


class _Response:
    def __init__(self, choice):
        self.answers = {"thinking_need": _Answer(choice)}


class _Engine:
    """Records the requests it is given and answers with a fixed level."""

    def __init__(self, choice):
        self.choice = choice
        self.requests = []

    def answer(self, request):
        self.requests.append(request)
        return _Response(self.choice)


def test_question_is_the_five_level_choice_production_sends():
    assert QUESTION["type"] == "choice"
    assert tuple(QUESTION["criteria"]) == LEVELS
    assert len(LEVELS) == 5
    assert QUESTION["instructions"] == INSTRUCTION
    # The instruction is the retuned one: it must not talk the model down.
    assert "Do not pick a lower level" in INSTRUCTION
    assert "higher level for caution" not in INSTRUCTION


def test_every_level_has_a_criterion():
    assert set(CRITERIA) >= set(LEVELS)
    for level in LEVELS:
        assert CRITERIA[level].strip()


def test_cases_cover_both_classes():
    assert len(CASES) == 14
    assert {expected for _, expected, _ in CASES} == set(ACCEPT)
    for name, expected, text in CASES:
        assert name and text.strip()
        assert expected in ACCEPT


def test_accept_sets_keep_routine_cheap_and_deep_expensive():
    assert ACCEPT["routine"] == {"off", "minimal", "low"}
    assert ACCEPT["deep"] == {"high"}
    # A deep case parked on medium is the under-rating this workload catches.
    assert "medium" not in ACCEPT["deep"]


def test_request_for_reverses_only_the_option_order():
    forward = smoke.request_for("do the thing")
    backward = smoke.request_for("do the thing", reverse=True)

    assert forward["state"] == backward["state"] == "do the thing"
    assert forward["questions"]["thinking_need"]["criteria"] == QUESTION["criteria"]
    assert list(backward["questions"]["thinking_need"]["criteria"]) == list(reversed(LEVELS))
    assert set(backward["questions"]["thinking_need"]["criteria"]) == set(LEVELS)
    # Building a reversed copy must not mutate the shared question.
    assert list(QUESTION["criteria"]) == list(LEVELS)


def test_run_smoke_runs_every_case_in_both_orders_twice():
    engine = _Engine("high")
    rows = suite.run_smoke(engine, 1)

    assert len(rows) == len(CASES) * 2 * 2
    # One warmup request, then every case.
    assert len(engine.requests) == 1 + len(CASES) * 2 * 2
    assert {r["reverse"] for r in rows} == {False, True}
    assert {r["attempt"] for r in rows} == {"first", "repeat"}
    assert {r["case"] for r in rows} == {name for name, _, _ in CASES}
    assert all(r["ms"] >= 0 for r in rows)


def test_run_smoke_scores_against_the_accept_sets():
    deep = _Engine("high")
    deep_rows = [r for r in suite.run_smoke(deep, 1) if r["expected"] == "deep"]
    assert deep_rows and all(r["correct"] for r in deep_rows)

    routine = _Engine("high")
    routine_rows = [r for r in suite.run_smoke(routine, 1) if r["expected"] == "routine"]
    assert routine_rows and not any(r["correct"] for r in routine_rows)


def test_production_rows_are_the_forward_order_half():
    rows = suite.run_smoke(_Engine("low"), 1)
    production = suite.production_rows(rows)

    assert len(production) == len(rows) // 2
    assert all(r["reverse"] is False for r in production)
    assert {r["case"] for r in production} == {name for name, _, _ in CASES}
    # The reversed rows probe order robustness and must not be in the headline.
    assert len(production) < len(rows)


def test_run_smoke_sends_the_vendored_question_to_the_engine():
    engine = _Engine("medium")
    suite.run_smoke(engine, 1)

    # The warmup plus the first case, which is sent in forward order.
    first = engine.requests[1].questions["thinking_need"]
    assert first.instructions == INSTRUCTION
    assert list(first.criteria) == list(LEVELS)

    # ...and a reversed case really is reversed by the time it reaches the engine.
    reversed_orders = [
        list(r.questions["thinking_need"].criteria)
        for r in engine.requests[1:]
        if list(r.questions["thinking_need"].criteria) != list(LEVELS)
    ]
    assert reversed_orders
    assert all(order == list(reversed(LEVELS)) for order in reversed_orders)
