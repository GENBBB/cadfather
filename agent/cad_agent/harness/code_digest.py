"""Fingerprint of the code that produces a result.

A hash over the contents of the harness, policy and capability files. It is written
to each run's `provenance.json`; `tools/compare_runs.py` uses it to check whether two
runs were produced by the same code. The module imports nothing from the package: it
is loaded by file path (`importlib.util.spec_from_file_location`) so that
`cad_agent/__init__` is not executed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# Code that takes part in producing a result. Paths are relative to the repository
# root; a directory is taken whole via `**/*.py`, a file individually.
CODE_TREES: tuple[str, ...] = (
    "agent/cad_agent/harness",
    "agent/cad_agent/scaffold",
    "agent/cad_agent/capabilities",
)
CODE_FILES: tuple[str, ...] = (
    "agent/run_experiment.py",
    "agent/cad_agent/dsl_runtime.py",
)


def _digest(chunks: list[str]) -> str:
    sha = hashlib.sha256()
    for chunk in chunks:
        sha.update(chunk.encode("utf-8"))
        sha.update(b"\0")
    return sha.hexdigest()[:16]


def _file_digest(path: Path) -> str:
    """Hash of a file's contents; a missing file is also a state."""
    if not path.is_file():
        return "absent"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def code_digest(repo_root: Path) -> str:
    """Hash of our code that produces a result.

    Computed over file CONTENTS, not modification time or commit: an uncommitted
    edit changes results exactly as a committed one does.
    """
    parts: list[str] = []
    for tree in CODE_TREES:
        root = repo_root / tree
        # A typo in a directory path would give an empty walk and SILENTLY drop a
        # whole package from the key, so the cache would keep hitting across edits.
        # A missed name must fail rather than narrow the check.
        if not root.is_dir():
            raise RuntimeError(f"code_digest: no directory {tree!r}; the key would be computed without it")
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            parts.append(f"{path.relative_to(repo_root)}:{_file_digest(path)}")
    for name in CODE_FILES:
        path = repo_root / name
        parts.append(f"{name}:{_file_digest(path)}")
    return _digest(parts)


