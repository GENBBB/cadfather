#!/usr/bin/env bash
# Build `_cad_grad` for the current interpreter.
#
# Why. The parameter optimizer (`res.optimize`) relies on the C++ module
# `_cad_grad`, whose binary is tied to the Python version. The snapshot ships no
# prebuilt binaries, so the module is built for each environment. Without it the
# snapshot modules fail to import (`ImportError`).
#
# What it does: builds the module from the snapshot sources with our recipe
# (`agent/tools/cad_grad/CMakeLists.txt`) and puts the result in
# `build_native/cad_grad/`, from where `dsl_runtime` picks it up. The file name
# carries the ABI tag (`_cad_grad.cpython-312-x86_64-linux-gnu.so`), so builds
# for different interpreters coexist in one directory.
#
# Usage:
#
#     ./agent/tools/build_cad_grad.sh                # interpreter from PATH
#     ./agent/tools/build_cad_grad.sh /path/to/python # an explicit one
#
# Environment variables:
#
#     CAD_GRAD_SYMBOLS=1   keep symbols and debug info in the module, for a
#                          profiler (`py-spy --native`), which otherwise sees
#                          `0x...` addresses instead of functions. The machine
#                          code is the same: `Release` stays (`-O3`, pybind11
#                          LTO); `-g` is added and the `strip` that
#                          pybind11_add_module attaches to `Release` is
#                          disabled. Not `RelWithDebInfo`: it uses `-O2` and
#                          pybind11 drops LTO there.
#     CAD_GRAD_OUT=<dir>   where to put the module instead of
#                          build_native/cad_grad (for comparing two builds, not
#                          for the runtime).
#     CAD_GRAD_PATCHES=0   build the snapshot as is, without
#                          agent/tools/cad_grad/patches (by default the patches
#                          are applied) - a reference for equivalence checks.
#
# Requires: cmake >= 3.15, a C++17 compiler, pybind11 in the same environment as
# the target interpreter (`pip install pybind11`).
#
# The `vendor/cad_optimizer/` snapshot is only read, never modified.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RECIPE_DIR="$REPO_ROOT/agent/tools/cad_grad"
# Absolute: CMake would resolve a relative path against the build directory.
OUTPUT_DIR="$(realpath -m "${CAD_GRAD_OUT:-$REPO_ROOT/build_native/cad_grad}")"
BUILD_DIR="$(dirname "$OUTPUT_DIR")/.cmake"

PYTHON="${1:-python}"
command -v "$PYTHON" >/dev/null || { echo "[x] no such interpreter: $PYTHON"; exit 1; }
PYTHON="$(command -v "$PYTHON")"

TAG="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
VERSION="$("$PYTHON" -c 'import platform; print(platform.python_version())')"
echo "[*] Interpreter: $PYTHON (Python $VERSION, module suffix $TAG)"

# --- checks before the build, to avoid digging through the CMake log --------
if ! "$PYTHON" -c 'import pybind11' 2>/dev/null; then
    cat <<EOF
[x] pybind11 is missing in this environment, and without it the module target is not even created.
    Install it with the same interpreter and retry:

        $PYTHON -m pip install pybind11

EOF
    exit 1
fi
PYBIND11_DIR="$("$PYTHON" -m pybind11 --cmakedir)"
echo "[*] pybind11: $("$PYTHON" -c 'import pybind11; print(pybind11.__version__)') ($PYBIND11_DIR)"

command -v cmake >/dev/null || { echo "[x] cmake not found (>= 3.15 required)"; exit 1; }
echo "[*] cmake: $(cmake --version | head -1)"

# --- build -------------------------------------------------------------------
# The build directory is wiped: it caches the interpreter path, and a rerun with
# a different Python would build the module for the previous one.
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR" "$OUTPUT_DIR"

SYMBOL_ARGS=()
if [[ "${CAD_GRAD_SYMBOLS:-0}" == 1 ]]; then
    SYMBOL_ARGS=(-DCMAKE_CXX_FLAGS=-g "-DCMAKE_STRIP=$(type -P true)")  # a file, not the shell builtin
    echo "[*] Symbols: -g, strip disabled (machine code same as a regular build)"
fi

cmake -S "$RECIPE_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    "${SYMBOL_ARGS[@]}" \
    -DCAD_GRAD_PATCHES="$([[ "${CAD_GRAD_PATCHES:-1}" == 0 ]] && echo OFF || echo ON)" \
    -Dpybind11_DIR="$PYBIND11_DIR" \
    -DPYBIND11_FINDPYTHON=ON \
    -DPython_EXECUTABLE="$PYTHON" \
    -DPYTHON_EXECUTABLE="$PYTHON" \
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$OUTPUT_DIR"

cmake --build "$BUILD_DIR" --parallel "$(nproc 2>/dev/null || echo 4)"

MODULE="$OUTPUT_DIR/_cad_grad$TAG"
[[ -f "$MODULE" ]] || {
    echo "[x] Build finished, but there is no module: expected $MODULE"
    echo "    directory contents: $(ls "$OUTPUT_DIR" 2>/dev/null || echo 'empty')"
    exit 1
}

# --- check with the same interpreter ----------------------------------------
echo "[*] Checking the import..."
"$PYTHON" - "$OUTPUT_DIR" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import _cad_grad
names = [name for name in dir(_cad_grad) if not name.startswith("__")]
print(f"[+] _cad_grad imports: {_cad_grad.__file__}")
print(f"[+] functions available: {len(names)} (for example: {', '.join(sorted(names)[:5])})")
PY

# Machine-code fingerprint: builds with and without symbols, patched and
# reference, are compared by it rather than by file size.
if command -v objcopy >/dev/null; then
    TEXT_HASH="$(objcopy -O binary --only-section=.text "$MODULE" /dev/stdout | sha256sum | cut -c1-16)"
    echo "[+] .text sha256: $TEXT_HASH"
fi

echo
echo "[+] Done: $MODULE"
echo "    The runtime picks it up by itself: build_native/cad_grad is on sys.path"
echo "    via dsl_runtime. Full check: agent/tools/preflight.py --config <config>"
