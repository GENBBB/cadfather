#!/usr/bin/env python3
"""CLI for parsing the `launch` config section: a thin wrapper over `cad_agent.launch_plan`.

The parsing itself lives in the package (`agent/cad_agent/launch_plan.py`), because
the name `tools` is generic and may clash with another package of that name when
launched from a different root. This file stays only as a shell entry point:
`run_system.sh`, `run_exec_bench.sh`, `run_tests.sh` and `preflight.py` call it by
path, as a script.

    eval "$(python agent/tools/launch_plan.py --config configs/dialogue_lean.yaml)"
"""

from __future__ import annotations

import sys
from pathlib import Path

# A file run as a script puts its own directory (`agent/tools`) on `sys.path`,
# not `agent/`, so the package has to be added explicitly, as `preflight` does.
AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from cad_agent.launch_plan import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
