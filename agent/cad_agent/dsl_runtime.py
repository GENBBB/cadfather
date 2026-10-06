"""Runtime dependencies from ``vendor/``: the DSL dialect and the det branch.

The only place in the project that touches ``sys.path``.

Paths lead into our ``vendor/``, their order is explicit, and a missing
dependency raises ``DetDepsUnavailable`` with a clear message rather than being
swallowed by a broad ``except``.

About the dialect. ``cadgen`` is the **language the prediction is executed in**,
not an auxiliary library: generated code runs with a prefix that imports its
operations from there. There are two dialects (``wrapped`` and ``chain``), they
are API-incompatible, and the dialect is chosen explicitly (not by the order of
``sys.path.insert`` lines). :func:`verify_dialect` checks that ``import cadgen``
really resolves into the chosen directory.

The check is not a formality: ``vendor/cad_optimizer/python`` contains its
**own** ``cadgen`` package, an older chain-family variant that matches none of
our snapshots. It must be on ``sys.path`` for the optimizer, and if it lands
ahead of the dialect directory the pipeline would silently speak another
language.

Order matters: :func:`configure` must run **before** ``utils`` is imported (it
takes ``CODE_PREFIX`` from here). Changing the dialect afterwards raises an
error instead of silently desynchronizing the prefix and the language.
"""

from __future__ import annotations

import os

import importlib.util
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# agent/cad_agent/dsl_runtime.py -> repository root is three levels up.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VENDOR_ROOT = REPO_ROOT / "vendor"
# Built on this machine: binaries are tied to the Python version and the machine,
# so they live outside the snapshot and are not synchronized between machines.
NATIVE_BUILD_ROOT = REPO_ROOT / "build_native"
# `cadfit._native` (the C++ part of det) is built by `agent/tools/build_cadfit_native.sh`
# into this directory, not into the snapshot; attached to the package in :func:`_import_cadfit`.
CADFIT_NATIVE_DIR = NATIVE_BUILD_ROOT / "cadfit"

DIALECTS = ("wrapped", "chain")
DEFAULT_DIALECT = "wrapped"

# The prefixes repeat CODE_PREFIX_WRAPPED / CODE_PREFIX_CHAIN from
# cicada/utils/pipeline.py verbatim: the model was trained on exactly this text.
CODE_PREFIX_WRAPPED = (
    "import cadquery as cq\n"
    "from cadgen.selectors import PointOnEdgeSelector\n"
    "from cadgen.extrude import extrude\n"
    "from cadgen.shell import shell\n"
    "from cadgen.hole import hole\n"
    "from cadgen.revolve import revolve\n"
    "from cadgen.orto_cut import orto_cut\n"
    "from cadgen.sweep import sweep\n"
    "from cadgen.gear import gear\n"
    "from cadgen.spring import spring\n"
    "from cadgen.sweep_adv import sweep_adv\n"
    "from cadgen.loft import loft\n"
    "r = None"
)

CODE_PREFIX_CHAIN = (
    "import cadquery as cq\n"
    "from cadgen.attach_array import _wp_attach_at\n"
    "from cadgen.sselectors import PointOnFaceSelector, PointOnEdgeSelector\n"
    "from cadgen.spherical_coords import SphericalAnglesDirection, Plane\n"
)

CODE_PREFIXES = {
    "wrapped": CODE_PREFIX_WRAPPED,
    "chain": CODE_PREFIX_CHAIN,
}


class DslRuntimeError(RuntimeError):
    """The dialect cannot be configured: missing directory, unknown name, late switch."""


class OptimizerUnavailable(RuntimeError):
    """The parameter optimizer did not come up: no snapshot or `_cad_grad` not built."""


class DetDepsUnavailable(RuntimeError):
    """The det-branch dependencies cannot be imported."""


class MetricsDepsUnavailable(RuntimeError):
    """The metrics machinery (GMS) cannot be imported."""


_ACTIVE: str | None = None


def dialect_root(dialect: str) -> Path:
    """Directory put on ``sys.path`` for ``import cadgen``."""
    return VENDOR_ROOT / "dsl" / dialect


def _managed_paths(dialect: str) -> list[Path]:
    """Paths in priority order: the dialect strictly first."""
    return [
        dialect_root(dialect),          # cadgen: the language the prediction runs in
        VENDOR_ROOT / "cadfit" / "src",  # cadfit: det
        # Optimizer snapshot: its modules and `cadgen_desugar`.
        VENDOR_ROOT / "cad_optimizer" / "python",
        # The built `_cad_grad`. We add this directory ourselves: the optimizer
        # modules add their own `build/` in the preamble, but each at its first
        # import, and a miss is hidden behind a printed warning and
        # `_cad_grad = None`. The snapshot ships no binaries: the module is built
        # for the environment by `agent/tools/build_cad_grad.sh`. The file name
        # carries the ABI tag, so builds for different interpreters do not clash.
        NATIVE_BUILD_ROOT / "cad_grad",
        VENDOR_ROOT / "image2cad",      # cadgen_emit
        VENDOR_ROOT / "cicada_metrics",  # gms: the GMS metric machinery
    ]


def _all_managed_paths() -> set[str]:
    paths: set[str] = set()
    for dial in DIALECTS:
        paths.update(str(p) for p in _managed_paths(dial))
    return paths


def configure(dialect: str = DEFAULT_DIALECT) -> str:
    """Choose the DSL dialect and attach ``vendor/`` to ``sys.path``.

    Idempotent for repeated calls with the same dialect. Switching the dialect
    once it is fixed is an error: ``utils.CODE_PREFIX`` is already computed, and a
    silent switch would desynchronize the prefix and the runtime.
    """
    global _ACTIVE

    if dialect not in DIALECTS:
        raise DslRuntimeError(
            f"unknown DSL dialect: {dialect!r}; available {', '.join(DIALECTS)}"
        )

    if _ACTIVE is not None:
        if _ACTIVE == dialect:
            return _ACTIVE
        raise DslRuntimeError(
            f"the dialect is already fixed as {_ACTIVE!r}, cannot switch to {dialect!r}: "
            "configure() must be called before importing utils/actions/multiagent "
            "(the code prefix is computed at import)"
        )

    # The dialect directory is mandatory: without it there is nothing to execute the prediction.
    root = dialect_root(dialect)
    if not root.is_dir():
        raise DslRuntimeError(
            f"no dialect directory {root}: the vendor/ snapshot is incomplete, see vendor/README.md"
        )

    # The other snapshots are optional. A run without det (e.g. the stepwise
    # baseline) must not fail at startup over something it does not need;
    # import_det_deps()/import_gms() report the shortage when it matters.
    managed = _managed_paths(dialect)
    for path in managed[1:]:
        if not path.is_dir():
            logger.warning("%s is missing — features that depend on it are unavailable", path)
    paths = [p for p in managed if p.is_dir()]

    # Remove our own paths entirely, then insert them in reverse priority order,
    # so the dialect directory ends up first in sys.path regardless of what was
    # there before.
    managed_all = _all_managed_paths()
    sys.path[:] = [p for p in sys.path if p not in managed_all]
    for path in reversed(paths):
        sys.path.insert(0, str(path))

    _ACTIVE = dialect
    logger.info("DSL dialect: %s (%s)", dialect, dialect_root(dialect))
    return _ACTIVE


def active_dialect() -> str:
    """The current dialect; configures the default one on first access."""
    if _ACTIVE is None:
        configure(DEFAULT_DIALECT)
    assert _ACTIVE is not None
    return _ACTIVE


def code_prefix(dialect: str | None = None) -> str:
    """Code prefix for a dialect (the active one by default)."""
    return CODE_PREFIXES[dialect or active_dialect()]


def preamble_modules(dialect: str | None = None) -> tuple[str, ...]:
    """Modules imported by the dialect preamble.

    Derived from the preamble text itself rather than kept as a second list,
    which would silently diverge when the preamble changes: the warm-up would
    warm one thing while execution imports another.

    Used by the warm-up (what to put into the parent's ``sys.modules``) and by the
    check in the fork (what must already be there).
    """
    names: list[str] = []
    for line in code_prefix(dialect).splitlines():
        line = line.strip()
        if line.startswith("import "):
            names.append(line.split()[1])
        elif line.startswith("from ") and " import " in line:
            names.append(line.split()[1])
    # dict.fromkeys, not set: the preamble order is kept, so an error message reads
    # in the order the code would hit it.
    return tuple(dict.fromkeys(names))


# Variables that bound the thread pools of native math libraries. They are read at
# **import**, so they must be set before numpy, scipy and trimesh arrive in the
# process.
NATIVE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


# Graphics stack of the renderer. The values are **fixed**, not "as many as
# threads": for llvmpipe `LP_NUM_THREADS=0` means "do not start a worker pool at
# all" while `1` means "start one worker thread", which is not what we need.
GRAPHICS_ENV_VARS = {
    # The Mesa software rasterizer starts a pool per core; with many cores that is
    # a lot of threads in every render worker.
    "LP_NUM_THREADS": "0",
    # Main one: VTK picks a GPU window (EGL) wherever it sees a card, which creates
    # a CUDA context per worker just because a Plotter exists there. Rendering
    # must be fully CPU-side, so the window is set explicitly.
    "VTK_DEFAULT_OPENGL_WINDOW": "vtkOSOpenGLRenderWindow",
    "VTK_SMP_BACKEND_IN_USE": "Sequential",
    "PYVISTA_OFF_SCREEN": "true",
}

# The run never needs a GPU: the generator and the agent live behind HTTP, and
# geometry and rendering run on the CPU. Unlike the variables above this one is
# set **over** the inherited value: an empty value hides the cards from the
# driver, and an inherited device list is the shell's environment, not a decision
# about our process. vLLM servers are unaffected: `run_system.sh` sets
# `CUDA_VISIBLE_DEVICES` for them separately on their own command line.
CUDA_ENV_VAR = "CUDA_VISIBLE_DEVICES"


def apply_graphics_limits(hide_cuda: bool = True) -> dict[str, str]:
    """Move rendering to the CPU and avoid creating a CUDA context.

    Called next to :func:`apply_thread_limits` for the same reason: pyvista and
    vtk read these variables when a window is created, and a window is created in
    every part worker.

    The two changes go together and are harmful alone. `VTK_DEFAULT_OPENGL_WINDOW`
    moves rendering from the GPU to llvmpipe, and llvmpipe without
    `LP_NUM_THREADS=0` starts a pool per core, so the avoided CUDA context would
    be paid for with hundreds of threads per worker.
    """
    effective: dict[str, str] = {}
    for name, value in GRAPHICS_ENV_VARS.items():
        os.environ.setdefault(name, value)
        effective[name] = os.environ[name]

    if hide_cuda:
        previous = os.environ.get(CUDA_ENV_VAR)
        os.environ[CUDA_ENV_VAR] = ""
        effective[CUDA_ENV_VAR] = ""
        if previous:
            logger.info("GPUs hidden from the run: %s=%r -> ''", CUDA_ENV_VAR, previous)

    logger.info("Render graphics stack: %s", effective)
    return effective


def apply_thread_limits(threads: int = 1) -> dict[str, str]:
    """Limit native thread pools before heavy modules are imported.

    The pipeline parallelizes with processes (a part is a process, with an
    execution pool inside, with forks inside). None of them needs internal
    multithreading, but BLAS and OpenMP by default start a pool **per core**,
    which is hundreds of extra threads per process on a many-core machine.

    Values that are already set are left alone: if they came from `launch.env`
    it is a deliberate decision and must not be silently overridden.
    """
    effective: dict[str, str] = {}
    for name in NATIVE_THREAD_ENV_VARS:
        os.environ.setdefault(name, str(threads))
        effective[name] = os.environ[name]
    return effective


def limit_ocp_threads(threads: int = 1) -> None:
    """Limit the OpenCASCADE thread pool.

    The third lever next to the two neighbours, and the only one that is NOT an
    environment variable: `OSD_ThreadPool` knows nothing about `OMP_NUM_THREADS`
    and has its own API. So `apply_thread_limits` does not cover it; call both.

    It lives here rather than with one consumer because there are two: the
    pipeline execution fork (`capabilities/execute.py`) and any other fork that
    builds solids. With the call in only one of them the other would silently
    build in a pool sized by the MACHINE's cores, visible only by counting
    threads.

    Order: before the first OCC operation. `OCP.OSD` is imported inside the
    function, not at module level: `dsl_runtime` is also called where OCP is not
    needed, and the import graph is an interface. An import failure is not
    swallowed: "OCC not found" on a path that is about to build a solid is a
    failure, not a reason to continue with a many-thread pool.
    """
    from OCP.OSD import OSD_ThreadPool  # noqa: PLC0415

    OSD_ThreadPool.DefaultPool_s().Init(threads)



def verify_dialect() -> Path:
    """Check that ``import cadgen`` resolves into the active dialect directory.

    Catches shadowing by the foreign ``cadgen`` package from
    ``vendor/cad_optimizer/python``: without this check a language swap shows
    nothing until the first code execution, where it looks like an ordinary
    geometry-building error.
    """
    dialect = active_dialect()
    expected = dialect_root(dialect).resolve()

    module = sys.modules.get("cadgen")
    if module is not None:
        origin = getattr(module, "__file__", None)
        source = "the already imported cadgen package"
    else:
        spec = importlib.util.find_spec("cadgen")
        if spec is None:
            raise DslRuntimeError(
                f"package cadgen not found on sys.path with dialect {dialect!r}; "
                f"expected in {expected}"
            )
        origin = spec.origin
        source = "the cadgen that would be imported"

    if origin is None:
        raise DslRuntimeError(f"{source} has no path on disk; the dialect cannot be determined")

    resolved = Path(origin).resolve()
    if expected not in resolved.parents:
        raise DslRuntimeError(
            f"{source} is in {resolved}, but the active dialect {dialect!r} expects {expected}. "
            "Most likely the dialect directory is shadowed by a foreign cadgen package "
            "(there is one in vendor/cad_optimizer/python)"
        )

    logger.info("Dialect check passed: cadgen from %s", resolved.parent)
    return resolved


def import_gms():
    """Import the GMS machinery from ``vendor/cicada_metrics``.

    Returns ``(TrimeshHandler, A_in_B_ball_matching_multiangle_v3)``.
    Requires ``pykdtree``.
    """
    active_dialect()  # make sure sys.path is configured

    try:
        from gms import (  # noqa: PLC0415
            A_in_B_ball_matching_multiangle_v3,
            TrimeshHandler,
        )
    except Exception as exc:
        raise MetricsDepsUnavailable(
            f"cannot import the GMS machinery from vendor/cicada_metrics "
            f"({type(exc).__name__}: {exc}). pykdtree is required; snapshot: "
            f"{VENDOR_ROOT / 'cicada_metrics' / 'gms.py'}"
        ) from exc

    return TrimeshHandler, A_in_B_ball_matching_multiangle_v3


@dataclass(frozen=True)
class DetDeps:
    """det-branch dependencies imported from ``vendor/``."""

    cadfit_single_pass: Callable[..., Any]
    best_primitive: Callable[..., Any]
    grid: Callable[..., Any]
    occupancy: Callable[..., Any]
    compute_residuals: Callable[..., Any]
    cut_op: Callable[..., Any]
    extrude_op: Callable[..., Any]
    revolve_op: Callable[..., Any]


OPTIMIZER_PACKAGE = "cad_optimizer_python"

# The name under which the snapshot expects itself in ABSOLUTE imports, and the
# submodules imported that way. `optimizer_numerical.py` and `optimizer_tool.py`
# lazily, inside an `if normalize:` branch, do
# `from python.cq_parser import _SyntheticNode, _make_slot`: in one place the
# snapshot refers to its package by the literal directory name, not relatively.
# Relative imports live under our name (:data:`OPTIMIZER_PACKAGE`), this one does
# not, and it failed with `ModuleNotFoundError: No module named 'python'` on
# EVERY call with normalization.
#
# Why an alias rather than a second `sys.path` entry: the `python/` directory on
# `sys.path` would yield TOP-level modules, and the snapshot's relative imports
# would break as described in `_optimizer_package`. Why per submodule rather than
# one `sys.modules["python"] = package`: the package alone is not enough.
# Importing `python.cq_parser` would find `__path__` and load cq_parser a SECOND
# time as a separate object, and a `ParamSlot` from the second copy is not the
# same class as the one in `parse_result` prepared by the first. A submodule
# alias guarantees a single module.
OPTIMIZER_ALIAS = "python"
OPTIMIZER_ALIASED_SUBMODULES = ("cq_parser",)


def _optimizer_package() -> Any:
    """Register the `vendor/cad_optimizer/python` snapshot as a package.

    Snapshot modules are written as parts of a package: `optimizer_numerical.py`
    does `from .cq_parser import ...`. A direct `import optimizer_numerical` with
    `python/` on `sys.path` yields a **top-level** module without a parent
    package, and the relative import fails with `attempted relative import with
    no known parent package`.

    The `python/` directory is a package (it has `__init__.py`), but its name is
    literally `python`, and hanging the snapshot on such a name is wrong: it is
    generic and occupies the whole process. So the package is registered under its
    own name (:data:`OPTIMIZER_PACKAGE`) and the modules are taken as its
    submodules; relative imports inside the snapshot then work as the author
    intended.

    One canonical name was not enough: in two places the snapshot refers to itself
    absolutely, by directory name. An alias is added on top of the registration
    (:data:`OPTIMIZER_ALIAS` and :data:`OPTIMIZER_ALIASED_SUBMODULES`; see there
    why the package alone is not enough).

    The snapshot itself must not be edited: the next refresh would silently
    discard the edit.
    """
    import importlib  # noqa: PLC0415

    active_dialect()  # make sure vendor/ is on sys.path

    if OPTIMIZER_PACKAGE not in sys.modules:
        package_dir = VENDOR_ROOT / "cad_optimizer" / "python"
        init_path = package_dir / "__init__.py"
        if not init_path.is_file():
            raise OptimizerUnavailable(
                f"the optimizer snapshot is incomplete: no {init_path}"
            )
        spec = importlib.util.spec_from_file_location(
            OPTIMIZER_PACKAGE, init_path, submodule_search_locations=[str(package_dir)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[OPTIMIZER_PACKAGE] = module
        spec.loader.exec_module(module)

    package = sys.modules[OPTIMIZER_PACKAGE]

    # Alias under the literal directory name; see `OPTIMIZER_ALIAS`.
    # `setdefault`, not assignment: if the name `python` already belongs to
    # something in this process we may not take it silently. The submodules are
    # still registered as ours: import consults `sys.modules` before the parent's
    # `__path__`, so the snapshot gets exactly the objects already loaded under
    # the canonical name.
    sys.modules.setdefault(OPTIMIZER_ALIAS, package)
    for name in OPTIMIZER_ALIASED_SUBMODULES:
        alias = f"{OPTIMIZER_ALIAS}.{name}"
        if alias not in sys.modules:
            sys.modules[alias] = importlib.import_module(f"{OPTIMIZER_PACKAGE}.{name}")

    return package


def import_optimizer() -> Any:
    """Import `optimizer_numerical` from the `vendor/` snapshot.

    The snapshot is registered as a package by :func:`_optimizer_package`; without
    that, relative imports inside the snapshot fail.
    """
    import importlib  # noqa: PLC0415

    _optimizer_package()

    try:
        return importlib.import_module(f"{OPTIMIZER_PACKAGE}.optimizer_numerical")
    except Exception as exc:
        raise OptimizerUnavailable(
            f"cannot import the parameter optimizer from {VENDOR_ROOT / 'cad_optimizer'} "
            f"({type(exc).__name__}: {exc})"
        ) from exc


def import_desugar() -> Any:
    """Import `cadgen_desugar` from the `vendor/cad_optimizer` snapshot.

    Same technique as :func:`import_optimizer`, for the same reason: the module
    lives inside the snapshot package. A separate function because the consumer
    differs: the wrapped-to-chain translation is needed **before** the optimizer
    and makes sense even if the optimizer itself did not come up.

    About the dialect. The snapshot is inconsistent upstream: the neighbouring
    `vendor/cad_optimizer/python/cadgen` is the chain family, while
    `cadgen_desugar` imports **wrapped** modules (`cadgen.extrude`,
    `cadgen.hole`, `cadgen.orto_cut`, `cadgen.revolve`, `cadgen.selectors`) that
    that chain package lacks (see `vendor/cad_optimizer/PROVENANCE.md`). That
    works for us: with wrapped active the dialect directory is first on
    `sys.path` and the translation runs **in our process**. The snapshot's own
    `desugar_cadgen()`, hardwired to a foreign interpreter path, is never called.

    Returns the module, not a function: the caller sees both `_desugar_impl` and
    `is_cadgen_functional`, and a test sees what to patch.
    """
    import importlib  # noqa: PLC0415

    _optimizer_package()

    try:
        return importlib.import_module(f"{OPTIMIZER_PACKAGE}.cadgen_desugar")
    except Exception as exc:
        raise OptimizerUnavailable(
            f"cannot import the wrapped->chain translation from {VENDOR_ROOT / 'cad_optimizer'} "
            f"({type(exc).__name__}: {exc})"
        ) from exc


# The replacements below (`_PyKDTree`, `_use_pykdtree`, `_use_voxel_lattice`) are
# not called by the det runtime: `vendor/cadfit` does the same with direct calls.
# They remain for the original-snapshot test bench and `voxel_lattice_check.py`.
DET_KDTREE_LEAFSIZE = 16


class _PyKDTree:
    """``pykdtree`` KD-tree in place of ``scipy.spatial.cKDTree`` inside the det snapshot.

    The snapshot reads exactly two things from the tree: ``query(points, 1)``
    returning (distance, index), and ``.data`` (the cloud's extent in
    ``sweep_candidates``); the wrapper provides only those. The search is exact,
    as in scipy, and a query is about twice as fast; ``cKDTree.query`` from
    ``ring_d_batch`` was the hottest spot of det.
    """

    def __init__(self, data: Any, *args: Any, **kwargs: Any) -> None:
        import numpy as np  # noqa: PLC0415
        from pykdtree.kdtree import KDTree  # noqa: PLC0415

        self._np = np
        self.data = np.ascontiguousarray(data, dtype=np.float64)
        self._tree = KDTree(self.data, leafsize=DET_KDTREE_LEAFSIZE)

    def query(self, x: Any, k: int = 1, *args: Any, **kwargs: Any) -> Any:
        return self._tree.query(self._np.ascontiguousarray(x, dtype=self._np.float64), k=k)


def _use_pykdtree(cadfit_pass: Any) -> None:
    """Replace the snapshot's tree. The foreign file is not edited: the module looks
    up the name ``cKDTree`` in its globals at call time, so the replacement covers
    all three places (`sweep_candidates`, end continuation, oriented pass).

    Falling back to scipy is loud: det output is the same, but wall time is a run
    condition, and a silently changed tree would read as "det got slower"."""
    try:
        import pykdtree.kdtree  # noqa: F401, PLC0415
    except Exception as exc:
        logger.warning(
            "pykdtree cannot be imported (%s: %s) — det stays on scipy cKDTree, "
            "det wall time is higher than usual", type(exc).__name__, exc,
        )
        return
    cadfit_pass.cKDTree = _PyKDTree


def _use_voxel_lattice() -> None:
    """Voxelize trimesh ``subdivide`` with a per-face lattice (`capabilities/voxel_lattice.py`).

    The snapshot calls the mesh method (``gt_mesh.voxelized(2.0)`` builds the
    ``_gvox`` grid in ``cadfit_single_pass``; ``voxelized(pitch)`` repairs GT in
    ``_repair_gt``) rather than a module name, so the class method is replaced.
    It takes effect in the process where det runs: its forks (call and skeleton
    warm-up). The result is bit-identical and much faster. Other methods
    (``method='ray'`` etc.) and foreign arguments take the original path. A repeat
    call does nothing."""
    import trimesh  # noqa: PLC0415

    if getattr(trimesh.Trimesh.voxelized, "_voxel_lattice", False):
        return
    from cad_agent.capabilities.voxel_lattice import voxelize_lattice  # noqa: PLC0415

    original = trimesh.Trimesh.voxelized

    def voxelized(self: Any, pitch: Any, method: str = "subdivide", **kwargs: Any) -> Any:
        if method == "subdivide" and set(kwargs) <= {"max_iter", "edge_factor"}:
            return voxelize_lattice(self, pitch, **kwargs)
        return original(self, pitch, method=method, **kwargs)

    voxelized._voxel_lattice = True  # type: ignore[attr-defined]
    trimesh.Trimesh.voxelized = voxelized


def _import_cadfit() -> Any:
    """Import the ``cadfit`` package and attach the built ``_native`` to it.

    The module lives outside the snapshot (:data:`CADFIT_NATIVE_DIR`), and
    ``section_analyzer`` looks for it as ``cadfit._native`` at its import, so the
    directory is added to ``cadfit.__path__`` before the first import of
    ``cadfit.candidates``. Without a build det works without C++, and
    ``section_analyzer`` warns about it.
    """
    active_dialect()  # make sure sys.path is configured
    import cadfit  # noqa: PLC0415

    native_dir = str(CADFIT_NATIVE_DIR)
    if CADFIT_NATIVE_DIR.is_dir() and native_dir not in cadfit.__path__:
        cadfit.__path__.append(native_dir)
    return cadfit


def cadfit_native_status() -> dict[str, Any]:
    """Whether ``cadfit._native`` is loaded and from where (``cadfit.native_status()``)."""
    return _import_cadfit().native_status()


def import_det_deps() -> DetDeps:
    """Import the det-branch dependencies; a clear error on failure.

    det is the ``vendor/cadfit`` snapshot (see its PROVENANCE.md): it calls
    ``pykdtree`` and the lattice voxelization itself, so process-wide replacements
    (:func:`_use_pykdtree`, :func:`_use_voxel_lattice`) are not needed.

    The emitters ``_extrude_op``/``_cut_op``/``_revolve_op`` live in ``cadgen_emit``
    and produce **wrapped** calls (``r=extrude(...)``, ``r=hole(...)``,
    ``r=revolve(...)``). The chain headers in ``cadfit_pass.py`` belong to
    ``emit_cadquery_vlm``, a path our pipeline does not use.
    """
    try:
        _import_cadfit()
        from cadfit.candidates.cadfit_pass import (  # noqa: PLC0415
            _best_primitive,
            _grid,
            _occupancy,
            cadfit_single_pass,
        )
        from cadfit.candidates.residual import compute_residuals  # noqa: PLC0415
        from cadgen_emit import _cut_op, _extrude_op, _revolve_op  # noqa: PLC0415
    except Exception as exc:
        raise DetDepsUnavailable(
            "cannot import the det branch dependencies from vendor/ "
            f"({type(exc).__name__}: {exc}). Expected cadfit in "
            f"{VENDOR_ROOT / 'cadfit' / 'src'} and cadgen_emit in "
            f"{VENDOR_ROOT / 'image2cad'}"
        ) from exc

    return DetDeps(
        cadfit_single_pass=cadfit_single_pass,
        best_primitive=_best_primitive,
        grid=_grid,
        occupancy=_occupancy,
        compute_residuals=compute_residuals,
        cut_op=_cut_op,
        extrude_op=_extrude_op,
        revolve_op=_revolve_op,
    )
