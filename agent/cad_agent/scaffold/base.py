"""Types of the seam between the harness and the orchestration.

The experiment contract requires the orchestration to arrive from outside as an
object and to receive its capabilities as a set of functions:

    result = scaffold.run(obs, resources)

This module describes exactly that junction. Everything here is stable; only the
implementation of `run` is mutated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class Observation:
    """What the scaffold knows about the part at the start of a rollout."""

    figure_id: str
    gt_mesh_path: Path
    work_dir: Path
    prefix_code: str
    difficulty: dict[str, Any] = field(default_factory=dict)


@dataclass
class Reconstruction:
    """What the scaffold must return.

    `code` and `mesh_path` are the **best** prefix found, not the last one:
    the return rule is separate from the stopping rule.
    """

    figure_id: str
    code: str | None = None
    mesh_path: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    stop_reason: str = ""
    # Which exit the policy finished the part through (`harness.search_types.DONE_BY_*`);
    # `None` if the policy did not finish it.
    done_by: str | None = None
    n_steps: int = 0
    journal: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


class Scaffold(Protocol):
    """The policy. The only method the harness calls."""

    def run(self, obs: Observation, res: Any) -> Reconstruction: ...
