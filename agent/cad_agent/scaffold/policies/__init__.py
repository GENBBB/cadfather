"""Registry of search policies.

A policy is an object with two methods, `plan` and `select`. The run config names
the policy (`experiment.scaffold.policy`) and carries only harness settings:
caps, log level, dataset; the search knobs belong to the policy.

This repository has one policy, `dialogue_lean`: at every turn the assistant
picks a tool by function call, sees images of the target and the best
candidates, and itself declares the part finished.
"""

from __future__ import annotations

from typing import Any, Callable

from cad_agent.scaffold.policies.dialogue_lean import DialogueLeanPolicy

POLICIES: dict[str, Callable[[], Any]] = {
    "dialogue_lean": lambda: DialogueLeanPolicy(objective="iou"),
}

POLICY_NAMES = tuple(sorted(POLICIES))


def build(name: str) -> Any:
    """Build a policy by name. An unknown name is an error, not a default."""
    factory = POLICIES.get(name)
    if factory is None:
        raise KeyError(f"Policy {name!r} not found. Available: {list(POLICY_NAMES)}")
    return factory()
