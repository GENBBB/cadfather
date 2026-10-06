"""Run provenance: which code, which config, which servers.

Why a separate file. `config.json` answers "what the author specified", and pairs
of runs are compared with `diff` on it: one differing line means one knob. Time,
host and PID would produce a difference in every pair, so everything about
**circumstances** rather than intent lives here, in `provenance.json`.

What is written:

- `code.digest` and `code.files`: the same `code_digest.code_digest` and the hash
  of every file that goes into it. The node has no `.git`, so the commit cannot be
  recovered there; the per-file hashes make it possible to find it locally;
- `git`: commit and dirty paths, when the run is started where `.git` exists;
- `source_config`: path, sha256 and YAML text the run started from. The config in
  the repository gets edited after a run and stops reproducing it;
- `servers`: `server_metrics.identity` of each role at the start and at the end;
  a restart or weight swap in the middle of a run shows up as a mismatch;
- `code_end`: the digest at the end and the files that changed on disk during the
  run. A fix uploaded on top of a running run would otherwise have to be traced
  through `ctime`.

Collection never fails a run: what cannot be determined is simply absent from the file.
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.util
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from cad_agent.harness import server_metrics

REPO_ROOT = Path(__file__).resolve().parents[3]


def _code_digest_module(repo: Path):
    """Load `code_digest` as a file rather than an import: importing a submodule
    would execute `cad_agent/__init__` with everything it pulls in."""
    path = repo / "agent/cad_agent/harness/code_digest.py"
    spec = importlib.util.spec_from_file_location("_code_digest_provenance", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def code_files(repo: Path) -> dict[str, str]:
    """Path -> hash for every file that goes into `code_digest`."""
    key = _code_digest_module(repo)
    out: dict[str, str] = {}
    for tree in key.CODE_TREES:
        for path in sorted((repo / tree).rglob("*.py")):
            if "__pycache__" not in path.parts:
                out[str(path.relative_to(repo))] = key._file_digest(path)
    for name in key.CODE_FILES:
        out[name] = key._file_digest(repo / name)
    return out


def code_state(repo: Path) -> dict[str, Any]:
    try:
        key = _code_digest_module(repo)
        return {"digest": key.code_digest(repo), "files": code_files(repo)}
    except Exception as exc:  # noqa: BLE001 — not knowing the digest beats not starting the run
        return {"error": f"{type(exc).__name__}: {exc}"}


def _git(repo: Path, *args: str) -> str | None:
    if not (repo / ".git").exists():
        return None
    try:
        result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def git_state(repo: Path) -> dict[str, Any] | None:
    head = _git(repo, "rev-parse", "HEAD")
    if head is None:
        return None
    status = _git(repo, "status", "--porcelain", "--untracked-files=no") or ""
    return {"commit": head.strip(), "dirty": [line[3:] for line in status.splitlines() if line.strip()]}


def _source(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        return {"path": str(path), "error": str(exc)}
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "text": data.decode(errors="replace"),
    }


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def collect_start(
    config: dict[str, Any],
    source_config: str | Path | None = None,
    repo: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Everything known at the moment the run starts."""
    return {
        "repo": str(repo),
        "started_at": _now(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "argv": list(sys.argv),
        "python": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cache_epoch": os.environ.get("CAD_EVO_CACHE_EPOCH"),
        "code": code_state(repo),
        "git": git_state(repo),
        "source_config": _source(source_config),
        "servers": server_metrics.identities(config),
    }


def collect_end(start: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Add to the start snapshot whatever may have changed during the run."""
    out = dict(start)
    out["finished_at"] = _now()
    end = code_state(Path(start.get("repo") or REPO_ROOT))
    before = (start.get("code") or {}).get("files") or {}
    after = end.get("files") or {}
    out["code_end"] = {
        "digest": end.get("digest"),
        "changed": sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p)),
    }
    out["servers_end"] = server_metrics.identities(config)
    return out
