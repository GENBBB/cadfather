from __future__ import annotations

import os

THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def apply_worker_thread_limits(threads: int = 1) -> None:
    """Limit native math runtimes before heavy geometry/scientific imports."""
    value = str(threads)
    for key in THREAD_LIMIT_ENV_VARS:
        os.environ[key] = value


def apply_occt_thread_limit(threads: int = 1) -> None:
    """Limit OpenCASCADE's process-local thread pool before CadQuery geometry."""
    from OCP.OSD import OSD_ThreadPool

    OSD_ThreadPool.DefaultPool_s().Init(threads)
