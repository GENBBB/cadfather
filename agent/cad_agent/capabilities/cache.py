"""Cache hit counter shared by all pipeline caches.

A dedicated type instead of a pair of ints per class: there are several caches, in
different layers and processes, and the question asked of each is the same: "does it
work at all?". Identical field names let them be merged into one table in `tech.json`
and compared without remembering how each cache named its counters.

Caches counted this way:

``gt_image``     rendered GT of a part (field of `FigureRenderer`);
``render_pred``  on-disk PNGs of predictions (`MeshRenderCache`);
``gt_points``    GT point cloud used for point selection (field of `StepProposer`);
``exec_gt``      normalized GT and its points inside the execution fork;
``det_gt``       GT skeleton of the deterministic branch;
``det_gt_repair``  repair of a holey GT; it survives the fork only as a file.

About `exec_gt`: it survives a task **only** in the `proxy_pool` backend, where the
shim warms up the GT and the forked grandchild inherits it. `serial_fork` and
`ephemeral_pool` have nothing to inherit from, so the hit rate there must be zero.
This number is therefore a direct check that the backend choice delivers what it
was made for, not just a convenience for analysis.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CacheStats:
    """Hits, misses and what the cache did beyond reading."""

    hits: int = 0
    misses: int = 0
    # Cache writes. Usually equal to misses, but not always: a write can fail
    # (no space, IO error), and then the next miss repeats.
    writes: int = 0
    # LRU evictions. A nonzero value means the cache is too small for the run:
    # a part pays for a render it has already done.
    evictions: int = 0

    def hit(self, count: int = 1) -> None:
        self.hits += count

    def miss(self, count: int = 1) -> None:
        self.misses += count

    def write(self, count: int = 1) -> None:
        self.writes += count

    def evict(self, count: int = 1) -> None:
        self.evictions += count

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float | None:
        """Hit rate. `None` if there were no lookups, not zero.

        The difference matters: zero means "the cache never hit", while no lookups
        means the path was not used in this run and no conclusion can be drawn.
        """
        return self.hits / self.lookups if self.lookups else None

    def to_dict(self) -> dict[str, int | float | None]:
        payload: dict[str, int | float | None] = {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": None if self.hit_rate is None else round(self.hit_rate, 4),
        }
        if self.writes:
            payload["writes"] = self.writes
        if self.evictions:
            payload["evictions"] = self.evictions
        return payload

    def merge(self, other: "CacheStats") -> None:
        self.hits += other.hits
        self.misses += other.misses
        self.writes += other.writes
        self.evictions += other.evictions


def merge_stats(
    target: dict[str, CacheStats],
    source: dict[str, CacheStats],
) -> dict[str, CacheStats]:
    """Merge counter sets by cache name (for the per-run summary)."""
    for name, stats in source.items():
        target.setdefault(name, CacheStats()).merge(stats)
    return target


def as_dict(stats: dict[str, CacheStats]) -> dict[str, dict[str, int | float | None]]:
    return {name: value.to_dict() for name, value in sorted(stats.items())}
