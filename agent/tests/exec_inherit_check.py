#!/usr/bin/env python3
"""Check that the execution fork **inherits** the language and metrics stack instead of loading them anew.

Run: ``python agent/tests/exec_inherit_check.py`` from the repository root.

Candidate code runs in a forked process, and the first thing it does is execute
the dialect preamble (``dsl_runtime.CODE_PREFIX_*``): ``import cadquery as cq``
and eleven ``from cadgen.* import ...``. If these packages are already in the
parent's ``sys.modules``, each ``import`` in the fork is a dict lookup. If not,
every candidate re-parses cadquery and drags OCP along, and the cost is visible
nowhere except in the execution ``wall_sec``, where it looks like "geometry is
slow".

This is what the ``ProxyPoolExecutor`` docstring promises: the grandchild
inherits the already imported OCP and the GT cache. The check asks the modules
themselves **in which process** they were imported, which cannot be fooled.

How it works. Stubs for ``OCP``, ``cadquery`` and ``cadgen`` are put on
``sys.path``; each appends its PID to a file when imported. The executed code
appends its own PID to another file. Two sets are then compared:

* a module is **inherited** if none of its imports happened in the process that
  executed the task (the sets do not intersect);
* a module is **loaded anew** if the import happened in the same process as the
  execution; then the number of such imports grows with the number of candidates.

The ``OCP`` stub is not only there so the check runs on a machine without CAD.
It is also a control: ``_preload_cad()`` imports OCP in the shim before the
fork, so OCP must come out inherited. If the control fails, the check itself is
broken, not the pipeline.

The metrics stack is checked differently, by asking the fork itself. numpy,
scipy and trimesh are real here (stubbing them would test the wrong code), so
the executed code prints what it found in ``sys.modules`` before its first
import. The module list comes from the guard (``execute.required_modules``)
rather than being duplicated here.

No CAD, models or network are needed. The GMS machinery (``gms`` + ``pykdtree``)
is checked only where the environment has it.

Background. The check used to be red, and that was about the pipeline:
``_preload_cad()`` raised only ``OCP.OSD``, so every grandchild imported
cadquery and cadgen itself, which cost most of a candidate's execution time. The
fix: ``_preload_cad()`` executes the active dialect's preamble in the parent, and
the fork keeps only the thread limiter and the ``_assert_preloaded`` guard.

The metrics stack had the same gap: measurement runs inside the fork
(`_evaluate`) and nobody warmed numpy, scipy, trimesh or our metrics module.
Some arrived by inheritance by accident (the worker imports `figure_run`, which
imports the metrics module), but the GMS machinery never did: `import_gms()` is
called lazily from `gms_handler`, i.e. first in the grandchild and again for
every candidate. The metrics-stack sections go red if `_preload_metrics()` is
removed: all four modules are missing in every executor.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_DIR))

from cad_agent import dsl_runtime  # noqa: E402

dsl_runtime.configure("wrapped")

# Imported before the stubs are laid out and pulls nothing heavy:
# `execute` knows about numpy and trimesh only by names in lists.
from cad_agent.capabilities import execute as execute_mod  # noqa: E402

FAILURES: list[str] = []

IMPORT_LOG_ENV = "CAD_AGENT_IMPORT_LOG"
EXEC_LOG_ENV = "CAD_AGENT_EXEC_LOG"

# Modules of the wrapped preamble: exactly what the fork executes at every step.
CADGEN_MODULES = (
    "selectors", "extrude", "shell", "hole", "revolve",
    "orto_cut", "sweep", "gear", "spring", "sweep_adv", "loft",
)

# Whether the GMS machinery is in the environment. `find_spec`, not an import:
# importing it here would put `gms` and `pykdtree` into the check's own
# sys.modules, and "inherited" would be indistinguishable from "dragged in by the
# check".
GMS_AVAILABLE = all(
    importlib.util.find_spec(name.split(".")[0]) is not None for name in execute_mod.GMS_MODULES
)

# What the fork must find ready. The list is asked from the guard rather than
# written here twice: if they drifted apart the check would be green while the
# run is red. The preamble is excluded: the recording stubs cover it, and its
# names would add nothing here.
WATCHED = tuple(
    name for name in execute_mod.required_modules(metrics=True, gms=GMS_AVAILABLE)
    if name in execute_mod.METRICS_MODULES + execute_mod.GMS_MODULES
)


def check(name: str, condition: bool, detail: str = "") -> None:
    # The detail is printed only on failure: an explanation like "the guard
    # guards nothing" next to the word OK reads as the opposite of what happened.
    # Numbers useful on success are printed on a separate line.
    print(f"  {'OK  ' if condition else 'FAIL'} {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


# --- stubs -------------------------------------------------------------------

_RECORDER = '''
import os
_log = os.environ.get("CAD_AGENT_IMPORT_LOG")
if _log:
    with open(_log, "a") as _stream:
        _stream.write(f"{__name__} {os.getpid()}\\n")
'''


def build_stubs(root: Path) -> None:
    """Lay out stubs, each of which records its own import."""
    root.mkdir(parents=True, exist_ok=True)

    (root / "OCP").mkdir(exist_ok=True)
    (root / "OCP" / "__init__.py").write_text(_RECORDER)
    (root / "OCP" / "OSD.py").write_text(_RECORDER + '''
class _Pool:
    def Init(self, n):
        pass


class OSD_ThreadPool:
    @staticmethod
    def DefaultPool_s():
        return _Pool()
''')

    # cadquery is stubbed just enough for `_evaluate` to reach the end:
    # `r.val()`, more than two faces, export to a file.
    (root / "cadquery.py").write_text(_RECORDER + '''
class _Shape:
    def Faces(self):
        return [0, 1, 2, 3]

    def export(self, path, **kwargs):
        with open(path, "w") as stream:
            stream.write("solid stub\\nendsolid stub\\n")


class Workplane:
    def val(self):
        return _Shape()
''')

    package = root / "cadgen"
    package.mkdir(exist_ok=True)
    (package / "__init__.py").write_text(_RECORDER)
    for name in CADGEN_MODULES:
        (package / f"{name}.py").write_text(
            _RECORDER + f"\n\ndef {name}(*args, **kwargs):\n    return None\n"
        )
    # Names that the preamble imports not by module name.
    (package / "selectors.py").write_text(
        _RECORDER + "\n\nclass PointOnEdgeSelector:\n    pass\n"
    )


def purge_stubs() -> None:
    """Purge the stubs from this process's ``sys.modules``.

    Required **before each** backend. The ``serial_fork`` and ``ephemeral_pool``
    constructors call ``_preload_cad()`` right here, in the check process, and a
    module left over from the previous backend is inherited by all processes at
    once. Its absence from the log then means "inherited from the check" rather
    than "inherited by the pipeline", and the control stops controlling.
    """
    for name in [
        m for m in sys.modules
        if m in ("cadquery", "cadgen", "OCP") or m.startswith(("cadgen.", "OCP."))
    ]:
        del sys.modules[name]


def task_code() -> str:
    """Dialect preamble + the executor's inheritance report + a minimal body.

    The metrics stack cannot be caught by stubs: numpy, scipy and trimesh are real
    here, and stubbing them would test the wrong code. So the fork itself is
    asked: the executed code prints what is in its ``sys.modules`` **before** it
    imports anything. The answer comes from the very process in question and
    cannot be faked.
    """
    return dsl_runtime.code_prefix("wrapped") + f'''
import os as _os, sys as _sys
_watched = {list(WATCHED)!r}
_present = [_name for _name in _watched if _name in _sys.modules]
with open(_os.environ["CAD_AGENT_EXEC_LOG"], "a") as _stream:
    _stream.write(f"{{_os.getpid()}} {{','.join(_present)}}\\n")
r = cq.Workplane()
'''


# --- log parsing -------------------------------------------------------------

def read_imports(path: Path) -> dict[str, set[int]]:
    """`module -> set of PIDs in which it was imported`."""
    by_module: dict[str, set[int]] = {}
    if not path.exists():
        return by_module
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        module, pid = line.rsplit(" ", 1)
        by_module.setdefault(module, set()).add(int(pid))
    return by_module


def read_execs(path: Path) -> dict[int, set[str]]:
    """`executor PID -> modules it found in sys.modules`."""
    by_pid: dict[int, set[str]] = {}
    if not path.exists():
        return by_pid
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        pid, _, present = line.partition(" ")
        by_pid[int(pid)] = {name for name in present.split(",") if name}
    return by_pid


def family(by_module: dict[str, set[int]], prefix: str) -> set[int]:
    """All PIDs in which at least one module of the family was imported."""
    pids: set[int] = set()
    for module, seen in by_module.items():
        if module == prefix or module.startswith(prefix + "."):
            pids |= seen
    return pids


# --- the run itself ----------------------------------------------------------

def run_backend(backend: str, pool_size: int, n_tasks: int, work: Path) -> tuple[dict[str, set[int]], set[int]]:
    """Run n_tasks tasks with the chosen backend, return (imports, executors).

    The stubs are imported **only** in children: the check process itself does not
    touch them, so everything in the log is the pipeline's work.
    """
    purge_stubs()
    import_log = work / f"{backend}_imports.txt"
    exec_log = work / f"{backend}_exec.txt"
    os.environ[IMPORT_LOG_ENV] = str(import_log)
    os.environ[EXEC_LOG_ENV] = str(exec_log)

    from cad_agent.capabilities.execute import EvalTask, build_executor

    tasks = [
        EvalTask(
            task_id=f"{backend}_{index}",
            code=task_code(),
            mesh_path=str(work / f"{backend}_{index}.stl"),
            measure=False,
        )
        for index in range(n_tasks)
    ]

    executor = build_executor(backend=backend, pool_size=pool_size, timeout=30.0)
    try:
        results = executor.evaluate(tasks)
    finally:
        executor.close()

    failed = [r for r in results if not r.success]
    if failed:
        print(f"    (tasks failed: {[(r.task_id, r.outcome, r.error) for r in failed][:2]})")

    return read_imports(import_log), read_execs(exec_log)


def assess(
    backend: str,
    imports: dict[str, set[int]],
    execs: dict[int, set[str]],
    n_tasks: int,
) -> None:
    executors = set(execs)
    print(f"\n--- {backend} ---")
    print(f"  executor processes: {len(executors)} (tasks {n_tasks})")

    # Control: OCP is raised in `_preload_cad()` before the fork and must be inherited.
    ocp = family(imports, "OCP")
    print(f"  OCP: imported in {len(ocp)} processes, of them executors {len(ocp & executors)}")
    check(
        f"[{backend}] control: OCP is not imported in the executor",
        bool(ocp) and not (ocp & executors),
        "no OCP import at all, so it was inherited not from the pipeline but from the check itself"
        if not ocp else f"{len(ocp & executors)} imports inside executors",
    )

    # How many TIMES OCP was imported, not just "not in the grandchild". The number
    # used to be printed without a bound, and a real defect slipped through:
    # `proxy_pool` forked shims unwarmed, each imported OCP itself, giving k
    # imports per worker against one for the neighbouring backends, i.e. many
    # simultaneous cold imports that made `warm_gt` stop waiting for confirmation.
    # The test stayed green because the grandchildren did not import.
    check(
        f"[{backend}] OCP is loaded once in the parent, not once per proxy",
        len(ocp) == 1,
        f"OCP imports in {len(ocp)} processes: warm-up runs after the fork, not before",
    )

    for name in ("cadquery", "cadgen"):
        seen = family(imports, name)
        inside = seen & executors
        print(f"  {name}: imported in {len(seen)} processes, of them executors {len(inside)}")
        check(
            f"[{backend}] {name} is inherited, not loaded again",
            bool(seen) and not inside,
            f"no imports at all" if not seen
            else f"{len(inside)} imports inside executors for {n_tasks} tasks",
        )

    # Metrics stack. Asked not of the stubs but of the executors themselves: each
    # prints what it found in `sys.modules` before its first import. A module that
    # is missing there will be raised by the fork itself, again for every
    # candidate, and the cost goes into `overhead_sec`, where it does not look
    # like work.
    missing_by_module: dict[str, int] = {}
    for present in execs.values():
        for name in WATCHED:
            if name not in present:
                missing_by_module[name] = missing_by_module.get(name, 0) + 1
    print(
        f"  metrics stack: {len(WATCHED)} modules, not inherited: "
        f"{sorted(missing_by_module) if missing_by_module else 'none'}"
    )
    check(
        f"[{backend}] the metrics stack is inherited by all executors",
        bool(execs) and not missing_by_module,
        "no executor reported, the check is broken" if not execs
        else ", ".join(
            f"{name} was missing in {count} of {len(execs)}"
            for name, count in sorted(missing_by_module.items())
        ),
    )


def main() -> None:
    print("Check that forks inherit the execution language and the metrics stack\n")

    # Honesty control: before the first line of the pipeline the metrics stack must
    # be absent from this process. Otherwise children inherit it from the check
    # rather than from the warm-up, and the "metrics stack inherited" section turns
    # green whether or not the pipeline warms it.
    print("--- control ---")
    leaked = [name for name in WATCHED if name in sys.modules]
    check(
        "[control] the check did not pull in the metrics stack itself",
        not leaked,
        f"already imported before the pipeline: {leaked}; the metrics section checks nothing",
    )
    if not GMS_AVAILABLE:
        print("  (no GMS machinery in this environment: its modules are excluded from the check)")

    with tempfile.TemporaryDirectory(prefix="exec_inherit_") as tmp:
        work = Path(tmp)
        stubs = work / "stubs"
        build_stubs(stubs)
        # Strictly before the dialect directory: the stubs must shadow the real
        # cadgen from vendor/, otherwise the check runs code that is absent locally.
        sys.path.insert(0, str(stubs))

        # Section 0. What the warm-up leaves behind.
        os.environ[IMPORT_LOG_ENV] = str(work / "preload_imports.txt")
        from cad_agent.capabilities import execute as execute_mod

        execute_mod._preload_cad()
        print("\n--- preload (_preload_cad) ---")
        for name in ("cadquery", "cadgen", *WATCHED):
            check(
                f"[preload] {name} is in sys.modules",
                name in sys.modules,
                "the preload does not touch it" if name not in sys.modules else "",
            )
        purge_stubs()

        # Section 1. The guard must not run idle: after the purge it must fail. This
        # tests exactly the class of error caught repeatedly before, an assertion
        # that is always true.
        print("\n--- guard (_assert_preloaded) ---")
        try:
            execute_mod._assert_preloaded("test")
        except AssertionError as exc:
            fired, text = True, str(exc)
        else:
            fired, text = False, ""
        check("[guard] fails when the preamble is not in sys.modules", fired,
              "did not fail: the guard guards nothing")
        check("[guard] names the missing modules",
              fired and "cadquery" in text and "cadgen." in text,
              text[:120])

        # The guard must ask by the task's composition, not "everything always". A
        # fork without measurement does not need the metrics stack, and demanding it
        # would fail on a legitimate case; a fork with measurement does need it.
        missing_metric = next(
            (name for name in execute_mod.METRICS_MODULES if name not in sys.modules), None
        )
        check("[guard] the metrics stack is warm, so there is nothing to require",
              missing_metric is None,
              f"{missing_metric} was not loaded by the warm-up, nothing more to check")
        if missing_metric is None:
            saved = sys.modules.pop("cad_agent.capabilities.metrics")
            try:
                # The preamble is also absent here (`purge_stubs` cleared it), so the
                # guard fails in both cases. What is checked is not the failure but
                # the content of the complaint: without measurement the metrics
                # stack must not appear in it.
                try:
                    execute_mod._assert_preloaded("test", metrics=False)
                except AssertionError as exc:
                    quiet_text = str(exc)
                else:
                    quiet_text = ""
                check("[guard] without a measurement the metrics stack is not required",
                      "cad_agent.capabilities.metrics" not in quiet_text,
                      "a metrics module was named missing for a task that does not measure")

                try:
                    execute_mod._assert_preloaded("test", metrics=True)
                except AssertionError as exc:
                    fired_m, text_m = True, str(exc)
                else:
                    fired_m, text_m = False, ""
                check("[guard] with a measurement fails on a missing metrics module", fired_m,
                      "did not fail: the metrics stack is not guarded at all")
                check("[guard] names the missing metrics module",
                      fired_m and "cad_agent.capabilities.metrics" in text_m,
                      text_m[:160])
            finally:
                sys.modules["cad_agent.capabilities.metrics"] = saved

        n_tasks = 6
        for backend, pool_size in (("proxy_pool", 2), ("serial_fork", 1), ("ephemeral_pool", 2)):
            imports, executors = run_backend(backend, pool_size, n_tasks, work)
            assess(backend, imports, executors, n_tasks)

        # The section goes last: importing the optimizer snapshot pulls its own
        # cadgen and would otherwise spoil the guard control above.
        print("\n--- optimizer warm-up (_preload_optimizer) ---")
        from cad_agent import dsl_runtime

        optimizer_name = f"{dsl_runtime.OPTIMIZER_PACKAGE}.optimizer_numerical"
        try:
            dsl_runtime.import_optimizer()
            available = True
        except dsl_runtime.OptimizerUnavailable:
            available = False
        sys.modules.pop(optimizer_name, None)
        try:
            execute_mod._preload_optimizer()
            raised = ""
        except Exception as exc:  # noqa: BLE001
            raised = f"{type(exc).__name__}: {exc}"
        check("[optimizer] warm-up does not fail", not raised, raised)
        if available:
            check("[optimizer] after warm-up the optimizer is in sys.modules, so the fork inherits it",
                  optimizer_name in sys.modules, "the warm-up did not import it")
        else:
            print("  (the optimizer cannot be loaded in this environment, only the absence of a crash was checked)")

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {', '.join(FAILURES)}")
        sys.exit(1)
    print("Execution forks inherit the execution language and the metrics stack.")


if __name__ == "__main__":
    main()
