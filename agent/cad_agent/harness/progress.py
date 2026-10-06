"""Per-part progress bar. It computes nothing and affects nothing.

A separate module rather than three lines in `pool.py`: the bar must be able to
**not exist**, for three different reasons at once: `tqdm` may be missing from
the run environment, the output may not be a terminal (on a server the run is
redirected to a file, and `\\r` would turn the log into megabytes of junk), and
the config may turn it off. Three conditions in a hot loop are three places to
end up with a bar that writes into the log, or a run that crashes over decoration.

Rule: **the bar can neither crash the run nor change its result.** Hence
`Progress` with an empty default implementation: callers always call
`advance()` and never ask whether a bar exists.

By default the bar is on only in a terminal: `run_system.sh` runs in the
foreground and inherits the terminal, so an interactive run gets the bar and a
redirect to a file does not. This is `null` in the config; `true`/`false` set it
explicitly, and an explicit value beats the terminal (otherwise there would be
no way to ask for a bar under `tee` or to remove it on a live terminal).
"""

from __future__ import annotations

import sys
from typing import Any

# Config key: `experiment.logging.progress`. `null` means follow the terminal.
AUTO = None


class Progress:
    """A bar that does not exist. The base class, and the fallback when tqdm is unavailable."""

    def advance(self, record: dict[str, Any] | None = None) -> None:
        """One part is done. `record` is its per-part record or `None`."""

    def note(self, text: str) -> None:
        """Short summary to the right of the bar. Decoration; it may be absent."""

    def close(self) -> None:
        """Idempotent: closing the bar twice is fine."""

    def __enter__(self) -> "Progress":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class _TqdmProgress(Progress):
    """A tqdm bar with a running postfix.

    The postfix shows the mean score and the number of failures, which is what one
    watches a live run for. They are computed here from the same records that go
    into the report, but **are not exposed**: `run_eval` builds the aggregates, and
    a second implementation for decoration would silently drift from the first.
    """

    def __init__(self, bar: Any):
        self._bar = bar
        self._done = 0
        self._score_sum = 0.0
        # Counted separately from `_done - _failed`: a part may arrive with no record
        # at all (`advance(None)`); it is then neither a failure nor a success and
        # simply stays out of the mean. Via subtraction it would lower the mean
        # for every such part.
        self._scored = 0
        self._failed = 0
        self._closed = False

    def advance(self, record: dict[str, Any] | None = None) -> None:
        self._done += 1
        if record is not None:
            # A failure is a record without metrics or with a zero score: exactly what
            # the contract counts as zero. The exact breakdown (execution error vs
            # non-watertight) lives in the report, not here.
            score = record.get("score")
            if score is None or record.get("metrics") is None:
                self._failed += 1
            else:
                self._score_sum += float(score)
                self._scored += 1
        try:
            self._bar.set_postfix_str(
                f"score {self._score_sum / max(self._scored, 1):.3f}, "
                f"failures {self._failed}",
                refresh=False,
            )
            self._bar.update(1)
        except Exception:  # noqa: BLE001
            # Decoration must not crash the run: if tqdm trips over a closed stream or
            # a foreign terminal, carry on without the bar.
            self._closed = True
            self._bar = _Silent()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._bar.close()
        except Exception:  # noqa: BLE001
            pass


class _Silent:
    """Stand-in for a broken bar: silently swallows everything."""

    def set_postfix_str(self, *args: Any, **kwargs: Any) -> None: ...

    def update(self, *args: Any, **kwargs: Any) -> None: ...

    def close(self) -> None: ...


class _PlainBar(Progress):
    """A bar without per-part semantics: a step is a step, the caller sets the postfix.

    A separate class rather than a flag on `_TqdmProgress`: that one computes the
    mean score and failures from the fields of a PART RECORD. For a stage with no
    such record at all (building a benchmark set), every built part would read as
    a failure: a plausible postfix with the wrong meaning.
    """

    def __init__(self, bar: Any):
        self._bar = bar
        self._closed = False

    def advance(self, record: dict[str, Any] | None = None) -> None:
        try:
            self._bar.update(1)
        except Exception:  # noqa: BLE001 - decoration must not crash the run
            self._closed = True
            self._bar = _Silent()

    def note(self, text: str) -> None:
        try:
            self._bar.set_postfix_str(text, refresh=False)
        except Exception:  # noqa: BLE001
            self._closed = True
            self._bar = _Silent()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._bar.close()
        except Exception:  # noqa: BLE001
            pass


def _open_bar(total: int, desc: str, unit: str) -> Any | None:
    """The tqdm bar itself, or `None` if the environment has no tqdm.

    It goes to stderr, not stdout: the log `StreamHandler` writes there too, and
    the bar and the logs must share one stream, otherwise redirecting one of them
    would interleave them in the terminal. This also keeps the bar compatible with
    `command | tee file`: stdout is redirected there while the terminal stays on
    stderr, so the bar remains visible.
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        file=sys.stderr,
        # The bar must not stay in the scrollback after the stage: the report
        # prints the result, not the bar.
        leave=False,
        dynamic_ncols=True,
    )


def bar(total: int, desc: str, unit: str = "pcs", wanted: bool | None = AUTO) -> Progress:
    """A bar of `total` steps; the caller decides what goes into the postfix.

    `wanted` is a tri-state, like the config key: `None` follows the terminal,
    `True`/`False` decide explicitly (and explicit beats the terminal, otherwise
    there is no way to ask for a bar under `tee` or remove it on a live terminal).
    """
    if total <= 0 or not enabled({"progress": wanted}):
        return Progress()
    made = _open_bar(total, desc, unit)
    return _PlainBar(made) if made is not None else Progress()


def enabled(logging_config: dict[str, Any] | None) -> bool:
    """Whether a bar is wanted: the explicit config value, otherwise whether stderr is a terminal."""
    wanted = (logging_config or {}).get("progress", AUTO)
    if wanted is not None:
        return bool(wanted)
    try:
        return bool(sys.stderr.isatty())
    except Exception:  # noqa: BLE001
        # A stream without `isatty` (stderr captured in tests) is not a terminal.
        return False


def build(total: int, logging_config: dict[str, Any] | None = None) -> Progress:
    """A bar for `total` parts, or a no-op if there should be none.

    A missing `tqdm` is not a config error and not a reason to fail: the run
    environment is not ours to assemble, and demanding a dependency for decoration
    is wrong. `preflight` reports the absence; the run simply goes on without a bar.
    """
    if total <= 0 or not enabled(logging_config):
        return Progress()
    made = _open_bar(total, "parts", "parts")
    return _TqdmProgress(made) if made is not None else Progress()
