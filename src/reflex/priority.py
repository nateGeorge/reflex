"""Priority serialization for classification requests.

One model instance on one accelerator means requests run one at a time. Arrival
order is the wrong order for the traffic this server actually sees:

* interactive calls -- one question over a short state, ~10ms of work;
* batch calls -- up to eight questions over a long state, seconds to minutes.

Under a plain lock, an interactive call that arrives while a batch is running
waits out the entire batch. Its caller allows a sub-second deadline, so it times
out and silently degrades -- even though the work it wanted would have cost
10ms. ``PriorityLock`` orders *waiters* by estimated cost, so the cheap call
goes next.

Reordering waiters is free: a waiter has not started, so there is no partial
work to discard. The request that holds the lock is never interrupted.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass

from reflex.schema import SystemOneRequest

__all__ = ["PriorityLock", "request_priority"]

# State characters per priority step. Deliberately coarse: the estimate only has
# to separate "interactive" from "batch", not predict milliseconds.
STATE_CHARS_PER_STEP = 4096


def _state_chars(state: object) -> int:
    """Character count of a state, whatever shape it arrived in."""
    if isinstance(state, str):
        return len(state)
    if isinstance(state, (dict, list)):
        return len(str(state))
    return 1


def request_priority(req: SystemOneRequest) -> int:
    """Estimated cost of a request. Lower runs first.

    Cost is roughly ``questions x state``: every question branch re-reads the
    shared state, so the two factors multiply. The estimate only has to rank
    requests against each other.
    """
    questions = max(1, len(req.questions))
    return questions * (1 + _state_chars(req.state) // STATE_CHARS_PER_STEP)


@dataclass
class _Waiter:
    priority: int
    seq: int
    enqueued: float


class PriorityLock:
    """Serialize access, serving the cheapest pending request first.

    Waiters age: a waiter's priority decays to zero over ``aging_seconds`` of
    waiting, so any waiter outranks a freshly arrived cheap call once it has
    waited that long. Without aging, a steady stream of cheap calls could starve
    an expensive one indefinitely.
    """

    def __init__(self, aging_seconds: float = 5.0) -> None:
        if aging_seconds <= 0:
            raise ValueError("aging_seconds must be positive")
        self._aging = aging_seconds
        self._cv = threading.Condition()
        self._busy = False
        self._seq = 0
        self._waiters: list[_Waiter] = []

    def _effective(self, waiter: _Waiter, now: float) -> int:
        """Priority after aging: decays to zero over ``aging_seconds`` of waiting.

        Proportional rather than one step per interval, so the bound is "a
        waiter wins after ``aging_seconds``" regardless of how expensive it is.
        """
        waited = now - waiter.enqueued
        if waited >= self._aging:
            return 0
        return int(waiter.priority * (1.0 - waited / self._aging))

    def _next(self, now: float) -> _Waiter | None:
        if not self._waiters:
            return None
        # Cheapest first, then oldest first, so equal-cost requests stay FIFO.
        return min(self._waiters, key=lambda w: (self._effective(w, now), w.seq))

    def acquire(self, priority: int = 0) -> None:
        """Block until this caller is the best waiter, then take the lock."""
        with self._cv:
            waiter = _Waiter(priority, self._seq, time.monotonic())
            self._seq += 1
            self._waiters.append(waiter)
            while True:
                if not self._busy and self._next(time.monotonic()) is waiter:
                    self._busy = True
                    self._waiters.remove(waiter)
                    return
                # The timeout bounds how long an aging promotion goes unnoticed.
                self._cv.wait(timeout=self._aging)

    def release(self) -> None:
        with self._cv:
            self._busy = False
            self._cv.notify_all()

    @contextlib.contextmanager
    def hold(self, priority: int = 0) -> Iterator[None]:
        self.acquire(priority)
        try:
            yield
        finally:
            self.release()
