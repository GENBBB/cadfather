"""Run scratch space: where candidate meshes live while they are needed.

A candidate mesh is needed only for the duration of a part's rollout: the next
step renders an image from it, `choose_point` measures distances on it, and the
metrics read it right after export. All of this happens within one part and
within about a minute.

Writing candidates to the run directory (NFS) and reading them back would cost a
network round trip per candidate, with most results thrown away. Instead they are
written here, to tmpfs (RAM, `/dev/shm`), and only what was explicitly requested
(`logging.save_meshes`) reaches the run directory. OCC cannot export geometry
straight to memory (`Shape.export` takes only a path), but tmpfs has the same effect
without touching a disk.

Layout: ``<root>/<run id>/<part>/``. A part directory is removed at the end of its
rollout, the run directory at the end of the run: a worker may die before its own
cleanup, and the run then cleans up after it.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# tmpfs exists on any modern Linux; if it is missing or not writable, fall back to
# the ordinary temp directory. On a node with an NFS home, /tmp is usually a local
# disk, which is still much closer than the run directory.
SHM_ROOT = Path("/dev/shm")


def default_root() -> Path:
    """Scratch root: tmpfs if writable, otherwise the system temp directory."""
    candidate = SHM_ROOT / f"cad_agent-{os.getuid()}"
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / f".probe.{os.getpid()}"
        probe.touch()
        probe.unlink()
        return candidate
    except OSError:
        fallback = Path(tempfile.gettempdir()) / f"cad_agent-{os.getuid()}"
        logger.warning("tmpfs %s is unavailable, scratch files will go to %s", SHM_ROOT, fallback)
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def run_dir(run_id: str, root: str | Path | None = None) -> Path:
    """Scratch directory of a run."""
    base = Path(root) if root else default_root()
    path = base / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def figure_dir(run_scratch: str | Path, figure_id: str) -> Path:
    """Scratch directory of a part. A `figure_id` like `group/name` is a subdirectory."""
    path = Path(run_scratch) / figure_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup(path: str | Path) -> None:
    """Remove a scratch directory. Silent: cleanup must never fail a run."""
    shutil.rmtree(Path(path), ignore_errors=True)
