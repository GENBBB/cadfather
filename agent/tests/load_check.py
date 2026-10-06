#!/usr/bin/env python3
"""Check that load is measured against the quota, not against the machine.

Reason. On the cluster `os.cpu_count()` returns 192 while `cpu.max` is 48: under
a CFS quota work runs at the speed of the quota. While oversubscription was
computed from the machine count, "1.2 threads per core" in the report was really
4.8, and the oversubscription warning never fired. Time taken away by the quota is
not visible in any single stage: it is smeared across all of them and looks like
"got slower".

The cgroup is substituted with a directory: the real one cannot be read, since it
differs per environment and the test would either check the machine instead of
the code or not run where there is no quota.

What is checked:

1. `cpu.max` is parsed: a quota, `max` (no quota), garbage;
2. cgroup v1 (`cpu.cfs_quota_us`) is read when v2 is absent;
3. `n_cpu()` does not exceed the machine count and drops to the quota;
4. `cpu.stat` counters reach the sample;
5. the summary holds the **difference** over the run, not the pod's accumulated counter;
6. with a single sample, a zero difference is not reported as "no throttling";
7. the oversubscription warning counts processes against the quota.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_ROOT))

from cad_agent.harness import config as config_mod  # noqa: E402
from cad_agent.harness import load as load_mod  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if condition else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def fake_cgroup(tmp: Path, name: str, files: dict[str, str]) -> Path:
    """Directory substituted for `/sys/fs/cgroup`."""
    root = tmp / name
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def use(root: Path | None) -> None:
    """Substitute the cgroup and reset the quota cache (it is read once)."""
    load_mod.CGROUP = root if root is not None else Path("/nonexistent-cgroup")
    load_mod._quota_cache.clear()


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="load_check_"))
    real_cgroup = load_mod.CGROUP
    machine = os.cpu_count() or 0

    print("1. cpu.max is parsed")
    use(fake_cgroup(tmp, "v2", {"cpu.max": "4800000 100000\n"}))
    check("quota 48 from 4800000/100000", load_mod.cpu_quota() == 48.0, str(load_mod.cpu_quota()))
    use(fake_cgroup(tmp, "v2_max", {"cpu.max": "max 100000\n"}))
    check("`max` means no quota", load_mod.cpu_quota() is None)
    use(fake_cgroup(tmp, "v2_junk", {"cpu.max": "garbage\n"}))
    check("garbage does not break the measurement", load_mod.cpu_quota() is None)
    use(None)
    check("no cgroup means no quota", load_mod.cpu_quota() is None)

    print("\n2. cgroup v1 when there is no v2")
    use(fake_cgroup(tmp, "v1", {
        "cpu/cpu.cfs_quota_us": "1600000\n",
        "cpu/cpu.cfs_period_us": "100000\n",
    }))
    check("quota 16 from cfs_quota_us", load_mod.cpu_quota() == 16.0, str(load_mod.cpu_quota()))
    use(fake_cgroup(tmp, "v1_off", {
        "cpu/cpu.cfs_quota_us": "-1\n",
        "cpu/cpu.cfs_period_us": "100000\n",
    }))
    check("-1 means no quota", load_mod.cpu_quota() is None)

    print("\n3. n_cpu() is the minimum of the quota and the machine")
    use(fake_cgroup(tmp, "small", {"cpu.max": "400000 100000\n"}))
    check("quota below the machine: the quota is used", load_mod.n_cpu() == 4.0, str(load_mod.n_cpu()))
    use(fake_cgroup(tmp, "huge", {"cpu.max": f"{(machine + 100) * 100000} 100000\n"}))
    check("a quota above the machine adds no cores", load_mod.n_cpu() == float(machine),
          f"{load_mod.n_cpu()} vs {machine}")
    use(None)
    check("without a quota: the machine count", load_mod.n_cpu() == float(machine))

    print("\n4. The cpu.stat counters reach the measurement")
    # The quota is deliberately smaller than any machine: otherwise `n_cpu()`
    # would hit the machine count and the machine would be tested, not quota parsing.
    use(fake_cgroup(tmp, "stat", {
        "cpu.max": "400000 100000\n",
        "cpu.stat": "usage_usec 1000000\nthrottled_usec 250000\n"
                    "nr_periods 100\nnr_throttled 20\nsome_other 5\n",
    }))
    sample = load_mod.snapshot()
    check("quota and machine are recorded next to n_cpu",
          sample["n_cpu"] == 4.0 and sample["n_cpu_quota"] == 4.0
          and sample["n_cpu_machine"] == machine, str(sample.get("n_cpu")))
    check("counters were read", sample.get("cpu_throttled_usec") == 250000
          and sample.get("cpu_usage_usec") == 1000000)
    check("extra cpu.stat fields are not carried", "cpu_some_other" not in sample)
    check("oversubscription is computed against available cores",
          sample.get("threads_per_cpu") == round(sample["n_threads"] / 4.0, 2))

    print("\n5. The summary has the run difference, not the pod counter")
    sampler = load_mod.LoadSampler(run_dir=None, enabled=False)
    sampler.samples = [
        {"cpu_usage_usec": 1_000_000, "cpu_throttled_usec": 100_000,
         "cpu_nr_periods": 100, "cpu_nr_throttled": 10},
        {"cpu_usage_usec": 3_000_000, "cpu_throttled_usec": 700_000,
         "cpu_nr_periods": 300, "cpu_nr_throttled": 110},
    ]
    summary = sampler.summary()
    check("used is a difference", summary.get("cpu_usage_sec") == 2.0, str(summary.get("cpu_usage_sec")))
    check("throttled is a difference", summary.get("cpu_throttled_sec") == 0.6, str(summary.get("cpu_throttled_sec")))
    check("share of the spent time", summary.get("cpu_throttled_share") == 0.3,
          str(summary.get("cpu_throttled_share")))
    check("share of throttled periods", summary.get("cpu_periods_throttled_share") == 0.5,
          str(summary.get("cpu_periods_throttled_share")))
    check("the quota reached the summary", summary.get("n_cpu_quota") == 4.0,
          str(summary.get("n_cpu_quota")))

    print("\n6. A single sample is not enough")
    sampler.samples = sampler.samples[:1]
    summary = sampler.summary()
    check("a zero difference is not reported for \"there was none\"", "cpu_throttled_share" not in summary)
    sampler.samples = [{"n_threads": 10}, {"n_threads": 12}]
    summary = sampler.summary()
    check("without counters there is no section", "cpu_throttled_sec" not in summary)

    print("\n7. Oversubscription is computed against the quota")
    # The quota is smaller than the machine, otherwise the machine is the limiter and
    # there is nothing to say about the quota (exactly what is checked on a separate line below).
    use(fake_cgroup(tmp, "warn", {"cpu.max": "400000 100000\n"}))
    records: list[str] = []

    class Catcher(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    logger = logging.getLogger("cad_agent.harness.config")
    handler = Catcher(level=logging.WARNING)
    logger.addHandler(handler)
    try:
        # 32 workers with a pool of 3 shims: 128 processes under a quota of 4.
        config_mod._check_workers(
            {"n_workers": 32, "execution": {"backend": "proxy_pool", "pool_size": 3}}
        )
        check("128 processes with a quota of 4 give a warning",
              any("oversubscription" in text for text in records), "; ".join(records)[:120])
        check("the warning names the quota",
              any("cgroup quota" in text for text in records), "; ".join(records)[:160])
        records.clear()
        # 2 workers with a pool of 1: 4 processes under a quota of 4, there is headroom.
        config_mod._check_workers(
            {"n_workers": 2, "execution": {"backend": "proxy_pool", "pool_size": 1}}
        )
        check("stays silent with headroom", not records, "; ".join(records)[:120])
    finally:
        logger.removeHandler(handler)

    load_mod.CGROUP = real_cgroup
    load_mod._quota_cache.clear()
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} — {', '.join(FAILURES)}")
        raise SystemExit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
