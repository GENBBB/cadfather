#!/usr/bin/env bash
# Build the LOCAL checks environment from `check_env.txt`.
#
#   agent/env/build_check_env.sh              build (or complete) and verify
#   agent/env/build_check_env.sh --check      only verify an existing environment
#   agent/env/build_check_env.sh --recreate   remove and rebuild from scratch
#
# Where and with what, overridable by variables:
#   CHECK_ENV_DIR=<path>       where to put it (default: `.venv-checks` in the root)
#   CHECK_ENV_PYTHON=<python>  what to create it with (default: system python3.12)
#
# Why a venv rather than conda: the environment belongs to the project, so it must
# be removed together with the working copy and rebuilt with one command. It lives
# inside the tree and is excluded from git and from syncing to a server.
#
# Why 3.12: there is no wheel of `cadquery-ocp==7.7.2` for 3.13, and that is the
# version used in the run environment; installing another would check a different
# OCC locally than the one that runs on the node. 3.12 also matches the node.
#
# Pitfall of the system 3.12: it has NO `ensurepip` (Ubuntu ships `python3-venv` as a
# separate package), so `-m venv` fails while creating pip. The environment is then
# created with `--without-pip` and pip is installed from outside, with the system pip
# and the `--python` flag. This is tested by trying, not by checking for a package:
# the normal path is tried first, and the workaround only if it fails.
#
# The OUTCOME is verified, not legality (same reasoning as in `build_env.sh`): after
# installation ALL versions from the list are compared, not a sample, and every
# module is imported. "The install succeeded" and "the environment is what was
# asked for" are different claims, and they diverge silently.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
REQ="$HERE/check_env.txt"
ENV_DIR="${CHECK_ENV_DIR:-$REPO_ROOT/.venv-checks}"

CHECK_ONLY=0
RECREATE=0
for arg in "$@"; do
    case "$arg" in
        --check)    CHECK_ONLY=1 ;;
        --recreate) RECREATE=1 ;;
        -h|--help)  awk 'NR==1 {next} /^#/ {print; next} {exit}' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown flag: $arg" >&2; exit 2 ;;
    esac
done

# --- what to create with ------------------------------------------------------
# Candidates are tried by RUNNING them: the name `python3.12` on PATH may be
# anything, and exactly the 3.12 line is needed (see the header).
pick_base_python() {
    local candidate
    for candidate in "${CHECK_ENV_PYTHON:-}" python3.12 /usr/bin/python3.12 python3; do
        [[ -n "$candidate" ]] || continue
        command -v "$candidate" >/dev/null 2>&1 || [[ -x "$candidate" ]] || continue
        if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
            BASE_PYTHON="$candidate"
            return 0
        fi
    done
    return 1
}

PYTHON="$ENV_DIR/bin/python"

if (( ! CHECK_ONLY )); then
    (( RECREATE )) && { echo "Removing $ENV_DIR"; rm -rf "$ENV_DIR"; }

    if [[ ! -x "$PYTHON" ]]; then
        pick_base_python || {
            echo "No Python 3.12 interpreter, and the check environment is built only for it." >&2
            echo "Install (Ubuntu):    sudo apt install python3.12 python3.12-venv" >&2
            echo "Or specify your own: CHECK_ENV_PYTHON=<python3.12> $0" >&2
            exit 1
        }
        echo "Base: $BASE_PYTHON ($("$BASE_PYTHON" -V 2>&1))"
        echo "Creating $ENV_DIR"
        if ! "$BASE_PYTHON" -m venv "$ENV_DIR" 2>/dev/null; then
            # No ensurepip: create an empty environment and install pip from outside.
            rm -rf "$ENV_DIR"
            "$BASE_PYTHON" -m venv --without-pip "$ENV_DIR"
            "$BASE_PYTHON" -m pip --python "$PYTHON" install --quiet --upgrade pip setuptools wheel || {
                echo "Could not put pip into $ENV_DIR (neither ensurepip nor a system pip)." >&2
                exit 1
            }
        fi
    fi

    echo "Installing the pins from $(basename "$REQ")"
    "$PYTHON" -m pip install --quiet --disable-pip-version-check -r "$REQ"
    # Snapshot of EVERYTHING that resulted: the list above holds direct pins, and
    # several times more arrived with them. It is the rollback point and the
    # comparison baseline for the next rebuild.
    "$PYTHON" -m pip freeze > "$ENV_DIR/pip-freeze.txt"
fi

[[ -x "$PYTHON" ]] || { echo "No environment: $ENV_DIR (build it: $0)" >&2; exit 1; }

# --- verification -------------------------------------------------------------
"$PYTHON" - "$REQ" <<'PY'
"""Compare what was built with the list and exercise every module."""
import importlib
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

req_path = Path(sys.argv[1])
wanted: dict[str, str] = {}
for line in req_path.read_text(encoding="utf-8").splitlines():
    line = line.split("#", 1)[0].strip()
    if not line:
        continue
    name, _, pin = line.partition("==")
    wanted[name.strip()] = pin.strip()

failed: list[str] = []
print(f"\nInterpreter: {sys.executable}\n             {sys.version.split()[0]}")
print(f"Pins in the list: {len(wanted)}")

mismatched = []
for name, pin in sorted(wanted.items()):
    try:
        actual = version(name)
    except PackageNotFoundError:
        mismatched.append(f"{name}: not installed")
        continue
    if pin and actual != pin:
        mismatched.append(f"{name}: {actual}, pinned {pin}")
if mismatched:
    failed.append("installed set differs from the list")
    print("\nINSTALLED SET DIFFERS:")
    for line in mismatched:
        print(f"  {line}")
else:
    print("The installed set matches the list in full.")

# Import is a separate claim: a package is installed by distribution name but
# breaks under its module name (`cadquery-ocp` -> `OCP`), and the OCC wheel fails
# not at install time but at first import, for lack of a system library.
MODULES = {
    "yaml": "configs",
    "numpy": "everywhere",
    "scipy": "metrics",
    "OCP": "OCC kernel",
    "cadquery": "DSL execution",
    "trimesh": "meshes and metrics",
    "pykdtree": "GMS",
    "rtree": "point selection",
    "pyvista": "view rendering",
    "vtk": "view rendering",
    "PIL": "images in the prompt",
    "openai": "vLLM client",
    "tqdm": "per-part progress bar",
    "point_cloud_utils": "point proposal",
}
print()
for module, why in MODULES.items():
    try:
        importlib.import_module(module)
    except Exception as exc:
        failed.append(f"import {module}")
        print(f"  BAD  {module:20s} {why}: {type(exc).__name__}: {exc}")
    else:
        print(f"  OK   {module:20s} {why}")

# Boolean engine: without it IoU cannot be computed at all, and the package list
# does not show that, because the engine is chosen by an import at call time.
try:
    import trimesh

    available = getattr(trimesh.boolean, "engines_available", set())
    if "manifold" in available:
        print(f"  OK   {'boolean engine':20s} manifold (IoU is computed)")
    else:
        failed.append("trimesh boolean engine")
        print(f"  BAD  {'boolean engine':20s} no manifold: {sorted(available)}")
except Exception as exc:
    failed.append("trimesh boolean engine")
    print(f"  BAD  {'boolean engine':20s} {type(exc).__name__}: {exc}")

if failed:
    print(f"\nNOT BUILT: {len(failed)} — {', '.join(failed)}")
    raise SystemExit(1)
print("\nThe check environment is built and matches the list.")
PY

cat <<TXT

How to run:
  ./agent/run_tests.sh                     the checks (the environment is picked up automatically)

Deliberately absent: vLLM, torch, CUDA and \`_cad_grad\`; see the header of
$(basename "$REQ"). Without \`_cad_grad\` the optimizer is silently off; to build it for
this environment: ./agent/tools/build_cad_grad.sh $ENV_DIR/bin/python
TXT
