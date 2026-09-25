"""Priority lock tests. Pure Python -- no model, no GPU, no accelerator needed."""

from __future__ import annotations

import threading
import time

import pytest

from reflex.priority import PriorityLock, _Waiter, request_priority
from reflex.schema import NoulAnswer, SystemOneRequest, SystemOneResponse, Usage
from reflex.server import create_app


def _req(questions: int, state_chars: int = 100) -> SystemOneRequest:
    return SystemOneRequest(
        state="x" * state_chars,
        questions={
            f"q{i}": {
                "type": "choice",
                "instructions": "Pick one.",
                "criteria": {"a": "first", "b": "second"},
            }
            for i in range(questions)
        },
    )


# --------------------------------------------------------------------------- priority


def test_priority_separates_interactive_from_batch():
    interactive = request_priority(_req(1, 200))
    batch = request_priority(_req(8, 200_000))
    assert interactive == 1
    assert batch > interactive * 100, "an 8-question long-state batch must rank far behind"


def test_priority_grows_with_questions_and_state():
    assert request_priority(_req(4, 100)) > request_priority(_req(1, 100))
    assert request_priority(_req(1, 4096)) > request_priority(_req(1, 10))
    # Sub-step state differences must not reorder anything.
    assert request_priority(_req(1, 10)) == request_priority(_req(1, 4095))


def test_priority_tolerates_non_string_state():
    req = SystemOneRequest(
        state={"note": "structured state"},
        questions={"q0": {"type": "noul", "instructions": "Is it true?"}},
    )
    assert request_priority(req) >= 1


# ------------------------------------------------------------------------------- lock


def test_lock_admits_one_holder_at_a_time():
    lock = PriorityLock()
    guard = threading.Lock()
    inside = 0
    peak = 0

    def run():
        nonlocal inside, peak
        for _ in range(20):
            with lock.hold(1):
                with guard:
                    inside += 1
                    peak = max(peak, inside)
                time.sleep(0.001)
                with guard:
                    inside -= 1

    threads = [threading.Thread(target=run) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)

    assert peak == 1, "two requests ran concurrently on a single-accelerator lock"


def test_lock_serves_the_cheapest_waiter_first():
    lock = PriorityLock()
    lock.acquire(0)  # main thread holds it, so both workers queue behind us

    served: list[str] = []

    def run(name: str, priority: int):
        with lock.hold(priority):
            served.append(name)

    big = threading.Thread(target=run, args=("big", 100))
    big.start()
    time.sleep(0.05)  # let "big" queue first
    small = threading.Thread(target=run, args=("small", 1))
    small.start()
    time.sleep(0.05)  # let "small" queue second

    lock.release()
    big.join(5)
    small.join(5)

    assert served == ["small", "big"], "a cheap call should jump an already-waiting batch"


def test_lock_keeps_equal_priorities_fifo():
    lock = PriorityLock()
    lock.acquire(0)

    served: list[int] = []

    def run(n: int):
        with lock.hold(7):
            served.append(n)

    threads = []
    for n in range(4):
        t = threading.Thread(target=run, args=(n,))
        t.start()
        threads.append(t)
        time.sleep(0.03)  # force a deterministic arrival order

    lock.release()
    for t in threads:
        t.join(5)

    assert served == [0, 1, 2, 3]


def test_aging_promotes_a_long_waiting_batch():
    lock = PriorityLock(aging_seconds=0.05)
    lock.acquire(0)

    served: list[str] = []

    def run(name: str, priority: int):
        with lock.hold(priority):
            served.append(name)

    big = threading.Thread(target=run, args=("big", 50))
    big.start()
    time.sleep(0.1)  # big waits out its whole aging window while we hold the lock

    small = threading.Thread(target=run, args=("small", 1))
    small.start()
    time.sleep(0.05)

    lock.release()
    big.join(5)
    small.join(5)

    assert served == ["big", "small"], "an aged batch must outrank a fresh cheap call"


def test_aging_promotes_and_floors_at_zero():
    lock = PriorityLock(aging_seconds=1.0)
    waiter = _Waiter(priority=4, seq=0, enqueued=0.0)
    assert lock._effective(waiter, 0.0) == 4
    assert lock._effective(waiter, 0.5) == 2
    assert lock._effective(waiter, 1.0) == 0
    assert lock._effective(waiter, 100.0) == 0, "aging must floor at zero, never go negative"


def test_hold_releases_the_lock_on_exception():
    lock = PriorityLock()
    with pytest.raises(RuntimeError), lock.hold(1):
        raise RuntimeError("boom")

    done = threading.Event()

    def run():
        with lock.hold(1):
            done.set()

    threading.Thread(target=run).start()
    assert done.wait(5), "the lock stayed held after an exception"


def test_aging_seconds_must_be_positive():
    with pytest.raises(ValueError):
        PriorityLock(aging_seconds=0)


# ----------------------------------------------------------------------------- server


class _FakeEngine:
    """Stands in for a real engine: no model, no GPU, just overlap bookkeeping."""

    model_name = "fake"
    device = "cpu"

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self.inside = 0
        self.peak = 0
        self.order: list[int] = []

    def answer(self, req: SystemOneRequest) -> SystemOneResponse:
        with self._guard:
            self.inside += 1
            self.peak = max(self.peak, self.inside)
            self.order.append(len(req.questions))
        time.sleep(0.05)
        with self._guard:
            self.inside -= 1
        return SystemOneResponse(
            model=self.model_name,
            answers={qid: NoulAnswer(noul=0.5) for qid in req.questions},
            usage=Usage(input_tokens=1),
        )


def _systemone_handler(app):
    return next(r.endpoint for r in app.routes if getattr(r, "path", None) == "/v1/systemone")


def test_server_routes_requests_through_the_priority_gate():
    engine = _FakeEngine()
    handler = _systemone_handler(create_app(engine))

    batch = threading.Thread(target=handler, args=(_req(8, 200_000),))
    batch.start()
    time.sleep(0.01)  # the batch is mid-answer, still holding the gate
    rest = [threading.Thread(target=handler, args=(_req(1, 100),)) for _ in range(3)]
    for t in rest:
        t.start()

    batch.join(10)
    for t in rest:
        t.join(10)

    assert engine.peak == 1, "the server let two requests overlap"
    assert len(engine.order) == 4
    assert engine.order[0] == 8, "the request that arrived first still ran first"
