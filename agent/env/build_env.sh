#!/usr/bin/env bash
# Rebuild a node environment from a recorded package list (`run_env.txt` / `gen_env.txt`).
#
#   agent/env/build_env.sh run            # run and assistant environment (py 3.12.13)
#   agent/env/build_env.sh gen            # generator environment (py 3.10.18, vLLM 0.10)
#   agent/env/build_env.sh run --check    # only compare what is already built against the list
#
# Where and with what, overridable via variables:
#   ENV_ROOT=<dir>      where to put the environments
#   CONDA=<path>        which conda creates the base
#
# CLUSTER PITFALLS THAT SHAPE THIS SCRIPT.
#
# 1. `conda env list` (and anything else that iterates over registered
#    environments) HANGS: `~/.conda/environments.txt` is shared by everyone and
#    lists dozens of paths into a dead directory. So the environment is created
#    strictly with `-p <path>`, and the checks never call conda: they call the
#    resulting interpreter.
# 2. The OUTCOME is checked, not legitimacy: after installation ALL versions from
#    the list are compared, not a sample. "The install went through" and "the
#    environment is the same" are different claims, and they diverge silently.
# 3. The loader finds the SYSTEM `libstdc++`, not the environment's. On the way to
#    the `sqlite3` CLI, vLLM pulls a `.so` from the nvidia pip wheels, whose RUNPATH
#    lists only nvidia directories: `$PREFIX/lib` is not in the search at all, and
#    the loader falls through to the system path. Then the ALREADY loaded library
#    wins: the environment's `libicui18n.so.78` requires CXXABI_1.3.15, the system
#    `libstdc++` provides 1.3.13, and the server dies on import ("vLLM did not come
#    up" without a single line about the model). The environment itself has CXXABI
#    1.3.17, so the problem is the search order, not the contents: fixed by
#    `LD_LIBRARY_PATH=$PREFIX/lib` baked into the `bin/vllm` wrapper (see
#    `wrap_vllm` below). A wrapper, not `activate.d`: `run_system.sh` calls
#    `$PREFIX/bin/vllm` directly, the environment is not activated, and `activate.d`
#    never runs.
# 6. BUILD DIRECTORIES MUST NOT LIVE ON NFS. The node's `$HOME` is entirely NFS
#    (the same volume as our directories), and the default JIT caches live in
#    `~/.cache`. A flashinfer kernel `ninja` build hung there for eight minutes in
#    `D`/`rpc_wait_bit_killable` with no open files, after reading `.ninja_deps` and
#    stalling on a path operation; the parent `EngineCore` sat in `pipe_read` the
#    whole time, so from outside it looks like "vLLM hangs at startup" with nothing
#    in the log. Plain `ls`/`touch` in the same directory stay instant, and `.nfs*`
#    files sit next to it: traces of files deleted while still open on a client that
#    never came back (two of them hold GPUs). So all build caches are redirected to
#    a local overlay (`/tmp`), where there is no NFS RPC. The price: the first
#    start after a container restart pays for the build again.
# 5. NO `/usr/local/cuda` AND NO nvcc IN ANY ENVIRONMENT. JIT kernel builds look
#    for `$CUDA_HOME/bin/nvcc`, and with an empty CUDA_HOME they fall through to the
#    hard-coded `/usr/local/cuda`, which the node does not have. The failure comes
#    at WARNING level: vLLM prints an `nvcc: not found` line per target (35 for the
#    assistant, the Qwen3-Next GDN kernels), falls back to the slow path and starts
#    as if nothing happened, so the mechanism is silently off and always has been.
#    There is one toolkit on the node, in the base conda (12.8.93 vs torch cu128 in
#    `run`), but its layout is not what JIT expects: headers are in
#    `targets/x86_64-linux/include`, libraries in `targets/x86_64-linux/lib`, and
#    `include/cccl` and `lib64` do not exist. So `link_cuda_home` builds a tree of
#    links of the right shape in the environment, and CUDA_HOME points at it: links,
#    not a copy, because the foreign directory is read-only to us and gets updated
#    without us.
# 4. `_cad_grad` is tied to the interpreter's ABI. It is NOT installed by pip and is
#    not in these lists: after building the run environment, run
#    `agent/tools/build_cad_grad.sh <new python>` separately, otherwise the optimizer
#    stays silently disabled (foreign code swallows the import miss).
set -euo pipefail

ENV_ROOT="${ENV_ROOT:?set ENV_ROOT to the directory for the environments}"
CONDA="${CONDA:-$(command -v conda)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROLE="${1:-}"; shift || true
CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

case "$ROLE" in
  run) PYVER=3.12.13; REQ="$HERE/run_env.txt"; PREFIX="$ENV_ROOT/cad_run";  LATE=() ;;
  gen) PYVER=3.10.18; REQ="$HERE/gen_env.txt"; PREFIX="$ENV_ROOT/cad_gen";
       # built from source and must match torch, so they are installed last
       LATE=(flash-attn xformers) ;;
  *) echo "usage: $0 {run|gen} [--check]" >&2; exit 2 ;;
esac
PY="$PREFIX/bin/python"

echo "role:        $ROLE"
echo "env:         $PREFIX"
echo "built from:  $REQ"
echo "python:      $PYVER"

# --- CUDA_HOME: where to get nvcc ------------------------------------------
# A link tree of the shape JIT expects (pitfall 5 in the header). Idempotent:
# `ln -sfn` repoints a link without nesting new ones. Built BEFORE installation,
# because flash-attn and other source builds look for nvcc at the pip stage, not
# at runtime.
link_cuda_home() {
    local prefix="$1" home="$1/cuda-home"
    local base; base="$(cd "$(dirname "$CONDA")/.." && pwd)"
    local target="$base/targets/x86_64-linux"
    if [[ ! -x "$base/bin/nvcc" || ! -d "$target/include" ]]; then
        echo "[!] $base has no nvcc or headers - CUDA_HOME not assembled,"
        echo "    JIT kernel builds will stay disabled (pitfall 5 in the header)"
        return 0
    fi
    mkdir -p "$home"
    ln -sfn "$base/bin"       "$home/bin"
    ln -sfn "$base/nvvm"      "$home/nvvm"
    ln -sfn "$base/targets"   "$home/targets"
    ln -sfn "$target/include" "$home/include"
    ln -sfn "$target/lib"     "$home/lib64"
    local nvcc_ver torch_ver
    nvcc_ver="$("$home/bin/nvcc" --version | sed -n 's/.*release \([0-9][0-9.]*\),.*/\1/p')"
    echo "[*] CUDA_HOME: $home -> $base (nvcc $nvcc_ver)"
    # The major version must match the one torch was built for: a mismatch is
    # silent and surfaces later as a line about a kernel that failed to build
    # in the middle of startup.
    torch_ver="$("$PY" -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true)"
    if [[ -n "$torch_ver" && "${nvcc_ver%%.*}" != "${torch_ver%%.*}" ]]; then
        echo "[!] nvcc $nvcc_ver vs torch cu$torch_ver: major versions differ,"
        echo "    kernels may fail to build; then unset CUDA_HOME entirely instead of editing it"
    fi
}

if (( ! CHECK_ONLY )); then
    if [[ -x "$PY" ]]; then
        echo "[*] the environment already exists; creation skipped"
    else
        echo "[*] creating the base (conda create -p, without consulting the environment list)"
        "$CONDA" create -y -p "$PREFIX" "python=$PYVER" pip
    fi
fi

# Both roles and both modes: in `--check` the environment already exists, but the
# link tree may not have been built (it appeared later than the environments).
# A full `if` rather than `&&`: under `set -e` a false condition at the end of a
# branch would kill the whole script with the branch's status.
if [[ -d "$PREFIX" ]]; then
    link_cuda_home "$PREFIX"
    if [[ -x "$PREFIX/cuda-home/bin/nvcc" ]]; then
        export CUDA_HOME="$PREFIX/cuda-home"
    fi
fi

if (( ! CHECK_ONLY )); then
    # The list goes in one call: it is self-consistent, and pip either installs it
    # whole or fails BEFORE installing with a clear conflict.
    MAIN="$(mktemp)"; trap 'rm -f "$MAIN"' EXIT
    if (( ${#LATE[@]} )); then
        grep -vE "^($(IFS='|'; echo "${LATE[*]}"))==" "$REQ" > "$MAIN"
    else
        cp "$REQ" "$MAIN"
    fi
    echo "[*] pip install: $(grep -cv '^#\|^$' "$MAIN") packages"
    "$PY" -m pip install --disable-pip-version-check --no-input -r "$MAIN"

    for late in "${LATE[@]}"; do
        spec="$(grep -E "^$late==" "$REQ" || true)"
        [[ -n "$spec" ]] || continue
        echo "[*] separately (built against torch): $spec"
        "$PY" -m pip install --disable-pip-version-check --no-input --no-build-isolation "$spec" \
            || {
                echo "[!] $spec failed to install; investigate separately, the rest of the environment is intact"
                echo "    for flash-attn the cause is usually the same: the build did not get nvcc (CUDA_HOME above)."
                echo "    The recipe for a prebuilt wheel is in the header of agent/env/gen_env.txt"
            }
    done
fi

# --- vLLM wrapper: library search order -------------------------------------
# Idempotent: the real script moves to `vllm-real` once, after that only the
# wrapper is rewritten. `pip install vllm` puts its own `bin/vllm` back and wipes
# the wrapper, so the step repeats on every build, and the check below runs the
# CLI for real.
wrap_vllm() {
    local prefix="$1"
    [[ -e "$prefix/bin/vllm" ]] || return 0
    if [[ ! -e "$prefix/bin/vllm-real" ]] || head -1 "$prefix/bin/vllm" | grep -q python; then
        mv -f "$prefix/bin/vllm" "$prefix/bin/vllm-real"
    fi
    # The line is written only when the tree is built: a CUDA_HOME pointing nowhere
    # is worse than an empty one, since torch and flashinfer will trust it and not
    # search further (pitfall 5 in the header).
    local cuda_line=""
    if [[ -x "$prefix/cuda-home/bin/nvcc" ]]; then
        cuda_line="export CUDA_HOME=\"$prefix/cuda-home\""
    fi
    # Build caches go to local disk, not NFS (pitfall 6). Each environment has its
    # own directory: the roles have different torch and CUDA, and a shared cache
    # would cost cross misses for nothing.
    local cache="/tmp/cad_agent_cache_${USER:-jovyan}/$(basename "$prefix")"
    cat > "$prefix/bin/vllm" <<WRAP
#!/usr/bin/env bash
# WRAPPER, not vLLM itself (installed by agent/env/build_env.sh, pitfall 3 in its header).
# The environment's lib comes before system ones: otherwise the CLI pulls the system
# libstdc++ through the RUNPATH of the nvidia pip wheels and dies importing sqlite3.
# CUDA_HOME lets JIT kernel builds find nvcc: there is no /usr/local/cuda on the
# node (pitfall 5). Build caches go to local disk: on NFS ninja hangs in D (pitfall 6).
export LD_LIBRARY_PATH="$prefix/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
$cuda_line
# Build caches on local disk: on NFS ninja hangs in D (pitfall 6).
mkdir -p "$cache" 2>/dev/null || true
export FLASHINFER_WORKSPACE_BASE="$cache"
export VLLM_CACHE_ROOT="$cache/vllm"
export TRITON_CACHE_DIR="$cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$cache/inductor"
exec "$prefix/bin/vllm-real" "\$@"
WRAP
    chmod +x "$prefix/bin/vllm"
    echo "[*] wrapper $prefix/bin/vllm updated (LD_LIBRARY_PATH=$prefix/lib${cuda_line:+, CUDA_HOME=$prefix/cuda-home}, build caches in $cache)"
}
wrap_vllm "$PREFIX"

# --- outcome check ----------------------------------------------------------
echo "=== comparison with the list ==="
"$PY" - "$REQ" <<'PY_CHECK'
import importlib.metadata as md, sys, re
want = {}
for line in open(sys.argv[1]):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    name, _, ver = line.partition("==")
    want[re.sub(r"[-_.]+", "-", name).lower()] = (name, ver)
have = {}
for d in md.distributions():
    n = (d.metadata["Name"] or "").strip()
    if n:
        have[re.sub(r"[-_.]+", "-", n).lower()] = d.version
missing, moved = [], []
for key, (name, ver) in sorted(want.items()):
    got = have.get(key)
    if got is None:
        missing.append(f"{name}: MISSING (expected {ver})")
    elif got != ver:
        moved.append(
            f"{name}: listed {ver}, installed {got}")
print(f"listed {len(want)}, installed {len(have)}")
for title, rows in (("MISSING", missing), ("DIFFERENT VERSION", moved)):
    if rows:
        print(f"--- {title} ({len(rows)}):")
        for r in rows:
            print("   ", r)
print("python:", sys.version.split()[0])
raise SystemExit(1 if (missing or moved) else 0)
PY_CHECK
rc=$?

# The vLLM CLI is RUN FOR REAL, not merely suggested in a hint. The check above
# answers "are the versions the same", and this run answers "does it start at all":
# pitfall 3 lives exactly between the two, with every version matching.
echo
echo "=== vLLM CLI launch ==="
if vllm_out="$("$PREFIX/bin/vllm" --version 2>&1)"; then
    echo "[+] $PREFIX/bin/vllm --version -> $(echo "$vllm_out" | tail -1)"
else
    echo "[!] $PREFIX/bin/vllm --version FAILED: a server will not start from this environment:"
    echo "$vllm_out" | tail -3
    rc=1
fi

echo
echo "=== next steps ==="
if [[ "$ROLE" == run ]]; then
cat <<EOF
  1. ./agent/tools/build_cad_grad.sh $PY      # otherwise the optimizer is silently disabled
  2. ./agent/run_tests.sh --python $PY        # the full set of checks in the environment
  3. ./agent/tools/preflight.py --config <config>
EOF
fi
exit $rc
