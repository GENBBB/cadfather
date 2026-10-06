#!/usr/bin/env bash
# Build `cadfit._native` (the C++ part of det: `cluster_normals` + `section`) for a given interpreter.
#
# Why. det comes from the `vendor/cadfit` snapshot (see its PROVENANCE.md), and its
# C++ module is its own, separate from the optimizer's `_cad_grad` and containing no
# optimizer code. Without the module det does not crash: `section_analyzer` warns and
# falls back to the Python branches, so the section detectors are empty and the det
# output differs. State: `cadfit.native_status()`.
#
# What the script does: builds the module from `vendor/cadfit/native` with its own
# `CMakeLists.txt` and puts it into `build_native/cadfit/`, from where `dsl_runtime`
# attaches it to the package (`cadfit.__path__`). The recipe in
# `vendor/cadfit/native/build.sh` is not used: it places the module inside the
# package tree, i.e. into the snapshot.
#
# Usage:
#
#     ./agent/tools/build_cadfit_native.sh                # with the interpreter from PATH
#     ./agent/tools/build_cadfit_native.sh /path/to/python # with an explicit one
#
# Requires: cmake >= 3.15, a C++17 compiler, pybind11 in the interpreter's environment.
# The script does not touch the `vendor/cadfit/` snapshot, it only reads it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_DIR="$REPO_ROOT/vendor/cadfit/native"
OUTPUT_DIR="$REPO_ROOT/build_native/cadfit"
BUILD_DIR="$REPO_ROOT/build_native/.cmake-cadfit"

PYTHON="${1:-python}"
command -v "$PYTHON" >/dev/null || { echo "[x] no such interpreter: $PYTHON"; exit 1; }
PYTHON="$(command -v "$PYTHON")"

TAG="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
echo "[*] Interpreter: $PYTHON (module suffix $TAG)"

"$PYTHON" -c 'import pybind11' 2>/dev/null || {
    echo "[x] pybind11 is missing in this environment: $PYTHON -m pip install pybind11"; exit 1; }
PYBIND11_DIR="$("$PYTHON" -m pybind11 --cmakedir)"
command -v cmake >/dev/null || { echo "[x] cmake not found (>= 3.15 required)"; exit 1; }

# The build directory is wiped: it caches the interpreter path.
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR" "$OUTPUT_DIR"

cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -Dpybind11_DIR="$PYBIND11_DIR" \
    -DPYBIND11_FINDPYTHON=ON \
    -DPython_EXECUTABLE="$PYTHON" \
    -DPYTHON_EXECUTABLE="$PYTHON" \
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$OUTPUT_DIR"

cmake --build "$BUILD_DIR" --parallel "$(nproc 2>/dev/null || echo 4)"

MODULE="$OUTPUT_DIR/_native$TAG"
[[ -f "$MODULE" ]] || { echo "[x] Build finished but the module is missing: expected $MODULE"; exit 1; }

echo "[*] Checking the import via dsl_runtime..."
PYTHONPATH="$REPO_ROOT/agent${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - <<'PY'
from cad_agent import dsl_runtime
status = dsl_runtime.cadfit_native_status()
print(f"[+] {status}")
assert status["available"] and status["section"], "cadfit._native was not picked up"
PY

echo "[+] Done: $MODULE"
