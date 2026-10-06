"""How many processes and threads live under a run, and whether the cores are oversubscribed.

Why. The pipeline multiplies on three levels at once: `n_workers` part processes,
inside each an execution pool (`pool_size` shims forking grandchildren), and inside
those native libraries start their own threads (OCC, BLAS, trimesh). The total
thread count is written nowhere, yet it decides whether a run is bound by cores or
sits idle. Without measurement, tuning `n_workers` and `pool_size` is guesswork.

Counted by hand, without `psutil`: it is not in every environment, while `/proc` on
Linux gives exactly the same. On systems without `/proc` the measurement silently
turns off; a run should not be lost because of it.

What is sampled:

- `n_processes` / `n_threads`: the whole process subtree of the run, i.e. workers,
  pool shims and their forks together;
- `threads_here`: threads of the sampling process itself;
- `rss_mb`: total resident memory of the subtree;
- `loadavg_1` and `n_cpu`: what it runs into;
- `threads_per_cpu`: the main figure; clearly above one means oversubscription,
  threads fighting for cores instead of working;
- `cpu_throttled_*`: how much CPU time was **taken away** from the run by hitting
  the cgroup quota.

Cores are counted by quota, not by machine. `os.cpu_count()` answers "how many cores
does the hardware have", while under a CFS quota work proceeds at the speed of
`cpu.max`; these are different numbers (e.g. 192 vs 48). Computing oversubscription
from the machine number understates it several times over and never shows the warning.

Layout. The run writes `load.jsonl` (one line per sample) and an aggregate in
`summary.json`; a part puts its own snapshot in `tech.json`. Knobs: `logging.load`
and `logging.load_interval_sec`.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROC = Path("/proc")
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096

# Variables that limit native library threads. Captured once per run: if they are
# not set, each library decides on its own, and "where did 600 threads come from"
# becomes a question without an answer.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    # Pool of the Mesa software rasterizer. Listed alongside the others because it is
    # the same kind of per-core pool, only started by rendering.
    "LP_NUM_THREADS",
)

# Graphics stack: not about threads but about what the picture is drawn with. Here
# for the same reason: "where does a CUDA context in the run come from" should be a
# question with an answer in the report, not a reason to poke at nvidia-smi mid-run.
GRAPHICS_ENV_VARS = (
    "VTK_DEFAULT_OPENGL_WINDOW",
    "VTK_SMP_BACKEND_IN_USE",
    "PYVISTA_OFF_SCREEN",
    "CUDA_VISIBLE_DEVICES",
)


def available() -> bool:
    """Whether `/proc` is present in the form we read it."""
    return PROC.is_dir() and (PROC / "self" / "stat").is_file()


def _read_stat(pid: str) -> tuple[int, int, int] | None:
    """`(ppid, threads, rss in pages)` from `/proc/<pid>/stat`.

    The process name is in parentheses and may contain spaces and parentheses
    themselves, so we split on the **last** `)`, not the first.
    """
    try:
        raw = (PROC / pid / "stat").read_text()
    except (OSError, ValueError):
        return None  # the process died between listing and reading
    try:
        fields = raw[raw.rindex(")") + 2:].split()
        return int(fields[1]), int(fields[17]), int(fields[21])
    except (ValueError, IndexError):
        return None


def snapshot(root_pid: int | None = None) -> dict[str, Any]:
    """Snapshot of the process subtree rooted at `root_pid` (default: this process)."""
    root_pid = os.getpid() if root_pid is None else root_pid
    result: dict[str, Any] = {
        "pid": root_pid,
        # Three numbers, not one: `n_cpu` is what oversubscription is computed from,
        # and the machine count and the quota sit next to it because their divergence
        # is the main trap of this measurement.
        "n_cpu": n_cpu(),
        "n_cpu_machine": os.cpu_count() or 0,
        "n_cpu_quota": cpu_quota(),
        "ts": round(time.time(), 3),
    }
    for name, value in cpu_stat().items():
        result[f"cpu_{name}"] = value

    try:
        result["loadavg_1"] = round(os.getloadavg()[0], 2)
    except (OSError, AttributeError):
        pass

    if not available():
        return result

    # Snapshot of the whole process table: the tree cannot be built one `ppid` at a
    # time, since children are born and die faster than we walk `/proc`.
    children: dict[int, list[int]] = {}
    info: dict[int, tuple[int, int]] = {}
    for entry in PROC.iterdir():
        if not entry.name.isdigit():
            continue
        stat = _read_stat(entry.name)
        if stat is None:
            continue
        ppid, n_threads, rss_pages = stat
        pid = int(entry.name)
        info[pid] = (n_threads, rss_pages)
        children.setdefault(ppid, []).append(pid)

    seen: set[int] = set()
    queue = [root_pid]
    n_threads = 0
    rss_pages = 0
    while queue:
        pid = queue.pop()
        if pid in seen or pid not in info:
            continue
        seen.add(pid)
        threads, rss = info[pid]
        n_threads += threads
        rss_pages += rss
        queue.extend(children.get(pid, ()))

    result["n_processes"] = len(seen)
    result["n_threads"] = n_threads
    result["threads_here"] = info.get(os.getpid(), (0, 0))[0]
    result["rss_mb"] = round(rss_pages * PAGE_SIZE / (1024 * 1024), 1)
    if result["n_cpu"]:
        result["threads_per_cpu"] = round(n_threads / result["n_cpu"], 2)
    return result


def thread_env() -> dict[str, str]:
    """Native-library thread limits: what is actually set."""
    return {name: os.environ[name] for name in THREAD_ENV_VARS if name in os.environ}


def graphics_env() -> dict[str, str]:
    """Render graphics stack: what is actually set.

    `CUDA_VISIBLE_DEVICES` is included even when empty: an empty string is a working
    state ("cards hidden"), and telling it from "variable absent" matters more than
    anything else in this dict.
    """
    return {name: os.environ[name] for name in GRAPHICS_ENV_VARS if name in os.environ}


CGROUP = Path("/sys/fs/cgroup")

# cgroup v2 CPU counters. `throttled_usec` is the time the kernel took away from
# tasks that hit the quota: the work was ready to run and was not allowed to. In our
# summary this is the only cost item not visible in any single stage: it is smeared
# over all of them.
CPU_STAT_FIELDS = ("usage_usec", "throttled_usec", "nr_periods", "nr_throttled")

_quota_cache: list[float | None] = []


def cpu_quota() -> float | None:
    """How many CPUs the **pod** is allowed, or `None` if there is no quota.

    Read once: the quota does not change during a run, while `snapshot()` is also
    called for every part.

    Order: v2, then v1. On the cluster it is v2 (`cpu.max` like `4800000 100000`,
    i.e. 48 cores), but the run's environment need not match the analysis environment.
    A read error means "the quota is not visible", not "there is no quota": both lead
    to the machine number, but failing silently here is not allowed, as load
    measurement must not bring a run down.
    """
    if _quota_cache:
        return _quota_cache[0]

    quota: float | None = None
    try:
        raw = (CGROUP / "cpu.max").read_text().split()
        if len(raw) == 2 and raw[0] != "max" and int(raw[1]) > 0:
            quota = int(raw[0]) / int(raw[1])
    except (OSError, ValueError, IndexError):
        try:
            limit = int((CGROUP / "cpu" / "cpu.cfs_quota_us").read_text())
            period = int((CGROUP / "cpu" / "cpu.cfs_period_us").read_text())
            if limit > 0 and period > 0:
                quota = limit / period
        except (OSError, ValueError):
            quota = None

    _quota_cache.append(quota)
    return quota


def n_cpu() -> float:
    """Cores the run can actually count on.

    The minimum of quota and machine: a quota above the core count adds no work, and
    the absence of a quota does not make the machine bigger.
    """
    machine = float(os.cpu_count() or 0)
    quota = cpu_quota()
    if quota is None:
        return machine
    if not machine:
        return round(quota, 2)
    return round(min(quota, machine), 2)


def cpu_stat() -> dict[str, int]:
    """CPU counters from cgroup: how much was spent and how much taken by the quota.

    The counters are **cumulative and shared by the whole pod**, not by our run: they
    include everything running in the pod, vLLM servers among it. Hence the difference
    between the start and the end of the run is reported (see :meth:`LoadSampler.summary`),
    and even that is about the pod, not only about us. It is enough for the question
    "was CPU taken from us", not for "how much exactly is ours".
    """
    try:
        raw = (CGROUP / "cpu.stat").read_text()
    except OSError:
        return {}
    result: dict[str, int] = {}
    for line in raw.splitlines():
        name, _, value = line.partition(" ")
        if name in CPU_STAT_FIELDS:
            try:
                result[name] = int(value)
            except ValueError:
                continue
    return result


class LoadSampler:
    """Background load sampling for the whole run.

    Lives in the main process and samples the **whole subtree**, so it sees workers,
    pool shims and their forks, i.e. exactly the load the run creates on the machine.

    The thread is a daemon and falls silent after a single error: load measurement
    must neither delay the run nor bring it down.
    """

    def __init__(
        self,
        run_dir: str | Path | None = None,
        interval_sec: float = 5.0,
        enabled: bool = True,
    ):
        # `run_dir=None` means measurement without a file: this is how a part counts
        # its peaks without producing a `load.jsonl` in every figure directory.
        self.path = Path(run_dir) / "load.jsonl" if run_dir is not None else None
        self.interval_sec = max(0.5, float(interval_sec))
        self.enabled = bool(enabled) and available()
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._root_pid = os.getpid()

    def __enter__(self) -> "LoadSampler":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        # The first sample is taken immediately: for a short part the loop may never
        # wake up, and the peak would stay empty.
        self.samples.append(snapshot(self._root_pid))
        self._thread = threading.Thread(target=self._loop, name="load-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=self.interval_sec + 1.0)
        self._thread = None

    def _loop(self) -> None:
        try:
            stream = open(self.path, "a", encoding="utf-8") if self.path is not None else None
            try:
                while not self._stop.is_set():
                    sample = snapshot(self._root_pid)
                    self.samples.append(sample)
                    if stream is not None:
                        stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        stream.flush()
                    self._stop.wait(self.interval_sec)
            finally:
                if stream is not None:
                    stream.close()
        except Exception:
            logger.warning("Load measurement stopped because of an error", exc_info=True)

    def summary(self) -> dict[str, Any]:
        """Peak and mean over the samples plus the thread environment.

        The peak matters more than the mean: the machine hits it, and it explains why a
        run thought parallel went at the speed of a sequential one.
        """
        result: dict[str, Any] = {
            "enabled": self.enabled,
            "n_samples": len(self.samples),
            "interval_sec": self.interval_sec,
            "n_cpu": n_cpu(),
            "n_cpu_machine": os.cpu_count() or 0,
            "n_cpu_quota": cpu_quota(),
            "thread_env": thread_env(),
            "graphics_env": graphics_env(),
        }
        if not self.samples:
            return result
        result.update(self._cpu_pressure())

        for key in ("n_processes", "n_threads", "rss_mb", "loadavg_1"):
            values = [sample[key] for sample in self.samples if sample.get(key) is not None]
            if not values:
                continue
            result[f"{key}_max"] = max(values)
            result[f"{key}_mean"] = round(sum(values) / len(values), 2)

        if result.get("n_cpu") and result.get("n_threads_max"):
            result["threads_per_cpu_max"] = round(result["n_threads_max"] / result["n_cpu"], 2)
        return result

    def _cpu_pressure(self) -> dict[str, Any]:
        """How much CPU time the pod spent and how much was taken from it.

        By the difference between the first and the last sample, not by the absolute
        counters: those accumulate since the pod was born (weeks for us) and say nothing
        about the run. The difference is about the run window but still about the whole
        pod: these counters cannot separate our fork from a neighboring vLLM.

        A single sample is not enough: its difference is zero, and presenting that as
        "no throttling" would be a lie. Then the section is simply absent.
        """
        if len(self.samples) < 2:
            return {}
        first, last = self.samples[0], self.samples[-1]
        delta: dict[str, Any] = {}
        for name in CPU_STAT_FIELDS:
            key = f"cpu_{name}"
            if first.get(key) is None or last.get(key) is None:
                return {}
            delta[key] = last[key] - first[key]

        usage_sec = delta["cpu_usage_usec"] / 1e6
        throttled_sec = delta["cpu_throttled_usec"] / 1e6
        result: dict[str, Any] = {
            "cpu_usage_sec": round(usage_sec, 1),
            "cpu_throttled_sec": round(throttled_sec, 1),
            "cpu_periods": delta["cpu_nr_periods"],
            "cpu_periods_throttled": delta["cpu_nr_throttled"],
        }
        # A share of what was spent rather than of some ceiling: "for every second of
        # work the pod lost this much". A ceiling cannot be computed here, since the
        # counters are shared by the pod.
        if usage_sec > 0:
            result["cpu_throttled_share"] = round(throttled_sec / usage_sec, 3)
        if delta["cpu_nr_periods"] > 0:
            result["cpu_periods_throttled_share"] = round(
                delta["cpu_nr_throttled"] / delta["cpu_nr_periods"], 3
            )
        return result
