"""CAD reconstruction pipeline package.

Three layers:

- ``capabilities``: stable functions that propose a DSL step, execute code,
  measure, render, build deterministically and optimize parameters.
- ``harness``: run plumbing: the part queue, worker processes, call counters
  and the budget cap, artifacts and reporting. It is deliberately outside the
  policy's reach: otherwise the budget limit would be mutable and meaningless.
- ``scaffold``: orchestration of a single part: when to call the agent, how
  many samples to draw, when to roll back, when to stop. Prompts live here too,
  since they change together with the logic.

The package is put on ``sys.path`` through the ``agent/`` directory: running
``python agent/run_experiment.py`` does this by itself, so no ``sys.path``
edits are needed. The only exception is ``dsl_runtime``, which attaches the
vendored snapshots from ``vendor/``.
"""
