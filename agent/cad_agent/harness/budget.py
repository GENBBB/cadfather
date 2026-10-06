"""Call counters and the absolute ceiling.

This lives in the harness on purpose: if the scaffold imposed its own budget, the
limit would be mutable and therefore meaningless. Exceeding a limit raises an
exception rather than silently truncating: a quietly cut-short run looks honest and
spoils the measurement.

Call kinds are kept apart because they cost differently: an image costs the agent
many times more than text, and a single counter would make the visual channel free.

Besides calls, this module holds the **per-part wall-clock** ceiling: it is of the
same nature (imposed by the harness, not owned by the scaffold, exceeded by an
exception). See `check_wall`.

There are deliberately no tokens here. The budget counts **calls**, because ceilings
are set in calls; only the transport knows tokens, and they are recorded in the log
(`tech.tokens`, with a separate pair of fields for the visual channel). A `tokens`
field nobody wrote to would travel into every per-figure record as zeros and look
like a real counter.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


class BudgetExceeded(RuntimeError):
    """Call ceiling exhausted. The figure is counted as a failure."""


class DeadlineExceeded(BudgetExceeded):
    """Per-part wall-clock ceiling exhausted. The rollout stops; what was found is kept.

    A separate class rather than a string inside `BudgetExceeded`, because the
    outcomes differ and must not be mixed. The call ceiling means "the part asked for
    more than it was allotted": a failure, zero in the table. The wall ceiling means
    "the part ran too long": the rollout is cut off, but the best prefix found so far
    must still reach the record. One field for two events gives a plausible report
    with the wrong diagnosis.

    Inheriting from `BudgetExceeded` is intentional: wrappers that already let the
    call ceiling propagate will keep propagating this one and will not swallow the
    deadline by oversight.
    """


# Outcome of a rollout stopped by the wall ceiling. One string shared by all
# wrappers and the harness: the report uses it to tell "the part ran too long"
# from a failure, and diverging spellings would make that impossible.
WALL_STOP_REASON = "part wall-time cap exhausted"

CALL_KINDS = ("vlm", "agent_text", "agent_visual", "agent_repair", "opt", "det", "exec")


@dataclass
class Budget:
    """Per-figure counter. A ceiling of None means no limit."""

    limits: dict[str, int | None] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=lambda: {kind: 0 for kind in CALL_KINDS})
    # Per-part wall-clock ceiling and the point it is counted from. `started` is set
    # from outside (`run_figure`) so the deadline runs from the start of the part,
    # not from the creation of the counter: GT warm-up lies between them and is
    # also part time.
    wall_sec: float | None = None
    started: float = field(default_factory=time.monotonic)
    _lock: Any = field(default_factory=threading.Lock, repr=False)

    def spend(self, kind: str, amount: int = 1) -> None:
        if kind not in self.counts:
            raise ValueError(f"Unknown call kind: {kind!r}")
        with self._lock:
            self.counts[kind] += amount
            limit = self.limits.get(kind)
            if limit is not None and self.counts[kind] > limit:
                raise BudgetExceeded(
                    f"Call limit for {kind} exceeded: {self.counts[kind]} > {limit}"
                )
            total_limit = self.limits.get("total")
            if total_limit is not None and self.total > total_limit:
                raise BudgetExceeded(f"Total call limit exceeded: {self.total} > {total_limit}")

    def check_calls(self, kind: str, amount: int = 1) -> None:
        """Check that the remaining budget covers a call BEFORE it is made.

        `spend` counts and only then raises: by that time the call has already
        happened and been paid for, and the counter honestly shows the overrun. That
        suffices for loop actions, whose legality is checked before launch. But a
        policy has a channel it calls itself (`state.ask`), and a post-check holds
        nothing there: a policy that catches `BudgetExceeded` with a broad
        `except Exception` asks again and again, and the counter goes arbitrarily far
        past the ceiling.

        The fix is not politeness on the policy's part but refusing BEFORE the call
        once the ceiling is exhausted: then a retry costs nothing, however it is caught.
        """
        with self._lock:
            limit = self.limits.get(kind)
            if limit is not None and self.counts.get(kind, 0) + amount > limit:
                raise BudgetExceeded(
                    f"Call limit for {kind} exceeded: "
                    f"{self.counts.get(kind, 0)} + {amount} > {limit}"
                )
            total_limit = self.limits.get("total")
            if total_limit is not None and self.total + amount > total_limit:
                raise BudgetExceeded(
                    f"Total call limit exceeded: {self.total} + {amount} > {total_limit}"
                )

    @property
    def elapsed_sec(self) -> float:
        return time.monotonic() - self.started

    def check_wall(self, where: str) -> None:
        """Soft wall-clock ceiling: checked at points, not by a watchdog thread.

        Soft means checked before a new unit of work, not interrupting one already
        started. A watchdog thread is unnecessary and harmful here: everything the
        part launches externally is already isolated in a fork with its own timeout
        (`execution.timeout_sec`, `det_timeout_sec`), so it cannot hang forever. All
        that is needed is a place to ask the time, and those are the points where
        **candidates are generated**: `propose_steps`, `algo_rebuild` and `optimize`.
        Hence the cost of the measure: an overshoot of at most one unit of work.

        An overshoot by one started unit is valid and is not truncated. Its size
        equals the longest call timeout that could start last, and it is bounded
        there, not here: `execution.opt_timeout_sec`, `det_timeout_sec`.

        `where` goes into the exception text: the part's record should show where it
        was stopped, not just that it was.
        """
        if self.wall_sec is None:
            return
        elapsed = self.elapsed_sec
        if elapsed >= self.wall_sec:
            raise DeadlineExceeded(
                f"Part wall-time cap exhausted before {where}: "
                f"{elapsed:.0f} s >= {self.wall_sec:.0f} s"
            )

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def to_dict(self) -> dict[str, Any]:
        return {"calls": dict(self.counts), "total": self.total}
