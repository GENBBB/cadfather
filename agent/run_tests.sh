#!/usr/bin/env bash
# Run the local checks with one command.
#
#   ./run_tests.sh                                   all checks, in the checks environment
#   ./run_tests.sh smoke cost_check                  only these
#   ./run_tests.sh --python /path/to/env/bin/python  with a given interpreter
#   ./run_tests.sh --config <yaml>                   in the run environment of a config
#   ./run_tests.sh --jobs 4                          four at a time
#   ./run_tests.sh --jobs 1                          strictly one after another
#   ./run_tests.sh --list                            list the checks
#
# Why a script. The checks are discovered from disk (`tests/*.py`), so a new
# file cannot be forgotten, and a failure does not drown in the output of the
# previous ones: a "name - outcome - time" table is printed at the end and the
# full output of each check goes to its own file.
#
# Some checks run ONE AT A TIME on purpose. `launch_check` (servers start in
# parallel, not one by one), `worker_death_check` (a part fits the wall-clock
# ceiling) and `smoke` (warm-up and det timeouts) assert things about
# duration, so a neighbour loading the machine would fail them on correct
# code. They are listed in `SOLO` and always run alone; the rest say nothing
# about time and run in a batch.
#
# Batch width is set by `--jobs`. By default it is derived from FREE MEMORY,
# not from the core count: a check peaks at about 400 MB (VTK and trimesh per
# part process, plus forks), and memory runs out long before cores do.
# `--jobs 1` restores the strict order.
#
# Where the interpreter comes from, in order:
#   --python <path>    exactly this one;
#   --config <config>  `launch.run_env` of the config, i.e. the environment the
#                      run uses;
#   $CAD_AGENT_PYTHON  otherwise;
#   .venv-checks       the local checks environment, if built;
#   python3 on PATH    last.
#
# The local environment (`.venv-checks` at the repo root, built by
# `agent/env/build_check_env.sh`) is picked up automatically, and this is
# never silent: the "Interpreter:" line prints it with its version. Explicit
# `--python` and `$CAD_AGENT_PYTHON` take precedence over a found environment.
#
# On a server the checks must run in the run environment; the simplest way is
# `--python $ENV/bin/python`. `--config` also works there but needs an
# interpreter with PyYAML to read the config, and the system python on a server
# may lack it. `run_system.sh` has the same constraint and the same knob,
# `CAD_AGENT_PYTHON`. The chosen interpreter is exported as `CAD_AGENT_PYTHON`
# to child processes: `launch_check` parses configs with it.

set -uo pipefail

AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$AGENT_DIR")"
TESTS_DIR="$AGENT_DIR/tests"

PYTHON=""
CONFIG_PATH=""
# Local checks environment (`agent/env/build_check_env.sh`). This is only a path;
# whether it is built is decided below by checking the file.
CHECK_ENV_PYTHON="${CHECK_ENV_DIR:-$REPO_ROOT/.venv-checks}/bin/python"
PARSER_PYTHON="${CAD_AGENT_PYTHON:-python3}"
TIMEOUT_SEC=1200
FAIL_FAST=0
LOG_DIR=""
JOBS=""
WANTED=()

# Checks whose assertions are about time. They run alone for any `--jobs`.
SOLO=(launch_check worker_death_check smoke)

# How many to run at once when not specified. The divisor is the measured peak of
# a check (~400 MB) plus headroom for its forks; the cap of 4 keeps the machine
# usable while the checks run.
auto_jobs() {
    local cores avail_mb by_mem
    cores="$(nproc 2>/dev/null || echo 1)"
    avail_mb="$(awk '/^MemAvailable:/ { print int($2 / 1024) }' /proc/meminfo 2>/dev/null)"
    [[ -n "$avail_mb" ]] || { echo 1; return; }
    by_mem=$(( avail_mb / 500 ))
    (( by_mem < 1 )) && by_mem=1
    (( cores > 4 )) && cores=4
    (( by_mem < cores )) && cores=$by_mem
    echo "$cores"
}

while (( $# )); do
    case "$1" in
        --python)     PYTHON="$2"; shift 2 ;;
        --config)     CONFIG_PATH="$2"; shift 2 ;;
        --timeout)    TIMEOUT_SEC="$2"; shift 2 ;;
        --jobs)       JOBS="$2"; shift 2 ;;
        --log-dir)    LOG_DIR="$2"; shift 2 ;;
        --fail-fast)  FAIL_FAST=1; shift ;;
        --list)       ls -1 "$TESTS_DIR"/*.py | xargs -n1 basename; exit 0 ;;
        # Help is the whole header, up to the first non-comment line (same trick as
        # in run_system.sh: a fixed line range silently goes stale).
        -h|--help)    awk 'NR==1 {next} /^#/ {print; next} {exit}' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*)           echo "Unknown flag: $1" >&2; exit 2 ;;
        *)            WANTED+=("$1"); shift ;;
    esac
done

# The interpreter from the config is the same `launch.run_env` the run uses.
# It is read with the same config parser (`launch_plan.py`) as in `run_system.sh`:
# a second way to read the same section would silently drift from the first.
# What parses the config: not just any interpreter. PyYAML in a local conda 3.13
# fails on `collections.Hashable`, and a server's `/usr/bin/python3` may lack
# PyYAML. So candidates are tried by running them, not by existence, for the
# same reason `launch_check` does not take the first python it finds.
pick_parser_python() {
    local candidate
    for candidate in "${CAD_AGENT_PYTHON:-}" "$PYTHON" "$CHECK_ENV_PYTHON" python3 /usr/bin/python3; do
        [[ -n "$candidate" ]] || continue
        if "$candidate" -c 'import yaml; yaml.safe_load("a: 1")' >/dev/null 2>&1; then
            PARSER_PYTHON="$candidate"
            return 0
        fi
    done
    return 1
}

if [[ -z "$PYTHON" && -n "$CONFIG_PATH" ]]; then
    [[ -f "$CONFIG_PATH" ]] || { echo "No such config: $CONFIG_PATH" >&2; exit 1; }
    pick_parser_python || {
        echo "No interpreter with a working PyYAML found to read $CONFIG_PATH." >&2
        echo "Say explicitly which interpreter runs the checks:  --python <env>/bin/python" >&2
        echo "or what reads the config:             CAD_AGENT_PYTHON=<python with PyYAML>" >&2
        exit 1
    }
    # Via `eval`, as in `run_system.sh`, rather than cutting the string out:
    # values pass through `shlex.quote`, which does NOT quote a plain path.
    # Parsing by quotes silently produced an empty RUN_ENV, the script fell back
    # to `python3` from PATH and failed every check with `ModuleNotFoundError`,
    # i.e. it lied about the code by testing the wrong interpreter.
    # The plan goes into a variable instead of straight into `eval`: otherwise
    # the outcome of `eval` is seen, not of config parsing, and a failed
    # `launch_plan` would pass as success with empty values.
    RUN_ENV=""
    PLAN="$("$PARSER_PYTHON" "$AGENT_DIR/tools/launch_plan.py" --config "$CONFIG_PATH")" \
        || { echo "Cannot parse the config: $CONFIG_PATH (see the message above)" >&2; exit 1; }
    eval "$PLAN"
    if [[ -n "$RUN_ENV" ]]; then
        [[ -x "$RUN_ENV/bin/python" ]] || { echo "No interpreter: $RUN_ENV/bin/python" >&2; exit 1; }
        PYTHON="$RUN_ENV/bin/python"
    else
        # A silent fallback to the active environment is exactly the case where
        # the checks run not where the launcher thinks. So say it out loud.
        echo "$CONFIG_PATH has no launch.run_env; using the active environment"
    fi
fi
# The local checks environment comes after an explicitly named one but before
# `python3` from PATH: locally PATH has conda 3.13, in which half of the checks
# fail at import. There is no silent fallback: the interpreter taken is printed
# below with its version.
if [[ -z "$PYTHON" && -z "${CAD_AGENT_PYTHON:-}" && -x "$CHECK_ENV_PYTHON" ]]; then
    PYTHON="$CHECK_ENV_PYTHON"
fi
PYTHON="${PYTHON:-${CAD_AGENT_PYTHON:-python3}}"
command -v "$PYTHON" >/dev/null 2>&1 || [[ -x "$PYTHON" ]] || {
    echo "Interpreter not found: $PYTHON" >&2; exit 1; }

# The list of checks comes from disk, not from the script text. An added file
# joins the run by itself, so "forgot to list it" is no longer possible.
ALL=()
while IFS= read -r path; do ALL+=("$path"); done < <(ls -1 "$TESTS_DIR"/*.py 2>/dev/null | sort)
(( ${#ALL[@]} )) || { echo "No checks in $TESTS_DIR." >&2; exit 1; }

# What was requested: a bare name, with or without `.py`, or a full path.
SELECTED=()
if (( ${#WANTED[@]} )); then
    for want in "${WANTED[@]}"; do
        found=""
        for path in "${ALL[@]}"; do
            base="$(basename "$path")"
            if [[ "$base" == "$want" || "$base" == "$want.py" || "$path" == "$want" \
                  || "$base" == "${want}_check.py" ]]; then
                found="$path"; break
            fi
        done
        [[ -n "$found" ]] || { echo "No such check: $want (list: --list)" >&2; exit 2; }
        SELECTED+=("$found")
    done
else
    SELECTED=("${ALL[@]}")
fi

if [[ -z "$LOG_DIR" ]]; then
    LOG_DIR="$REPO_ROOT/work_dirs/tests/$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$LOG_DIR"

# `CAD_AGENT_PYTHON` is the interpreter that PARSES configs (`launch_check` takes
# it from there), not the one that runs the checks. Usually they are the same,
# but not always: in a local conda 3.13 PyYAML 6.0.1 fails on
# `collections.Hashable`, and making it the parser would fail `launch_check` on
# something it does not test. So the chosen interpreter is passed on only if it
# can really read YAML.
if "$PYTHON" -c 'import yaml; yaml.safe_load("a: 1")' >/dev/null 2>&1; then
    export CAD_AGENT_PYTHON="$PYTHON"
else
    echo "PyYAML in $PYTHON does not work - the checks will parse configs themselves"
fi

[[ -n "$JOBS" ]] || JOBS="$(auto_jobs)"
[[ "$JOBS" =~ ^[0-9]+$ ]] && (( JOBS >= 1 )) || {
    echo "--jobs needs an integer >= 1, not '$JOBS'" >&2; exit 2; }

# What runs in the batch and what runs alone. Decided BEFORE the run so the
# "loners" are visible in the header: otherwise it is unclear why the run does
# not speed up fourfold.
BATCH=(); LONE=()
for path in "${SELECTED[@]}"; do
    name="$(basename "$path" .py)"
    solo=0
    for reserved in "${SOLO[@]}"; do [[ "$name" == "$reserved" ]] && solo=1; done
    if (( solo || JOBS == 1 )); then LONE+=("$path"); else BATCH+=("$path"); fi
done

echo "Interpreter:    $PYTHON ($("$PYTHON" -V 2>&1))"
echo "Checks:        ${#SELECTED[@]} of ${#ALL[@]}"
echo "Full output:   $LOG_DIR"
echo "Timeout:       $TIMEOUT_SEC s per check"
if (( JOBS > 1 )); then
    echo "Parallel:      $JOBS (alone: ${#LONE[@]} - they measure time)"
else
    echo "Parallel:      1 (sequential)"
fi
echo

# Each check's outcome goes to a file, not a variable: the body of `run_one` runs
# in a background subshell and variables do not come back out of it. Same reason
# as for the forks in the harness itself.
META_DIR="$LOG_DIR/.meta"
mkdir -p "$META_DIR"

run_one() {
    local path="$1" name log started code elapsed why
    name="$(basename "$path" .py)"
    log="$LOG_DIR/$name.log"
    started=$(date +%s)
    # `cd` to the repo root: some checks look up configs and manifests relative
    # to it, and from another directory they would fail on something other than
    # what they test.
    ( cd "$REPO_ROOT" && timeout "$TIMEOUT_SEC" "$PYTHON" "$path" ) >"$log" 2>&1
    code=$?
    elapsed=$(( $(date +%s) - started ))

    if (( code == 0 )); then
        why=""
    else
        # The last meaningful line: for our checks it is either "FAILED ..." or a
        # traceback. It also goes into the final table.
        why="$(grep -E 'FAILED|Error|error:' "$log" | tail -n 1 | cut -c1-100)"
        [[ -n "$why" ]] || why="$(tail -n 1 "$log" | cut -c1-100)"
        (( code == 124 )) && why="timeout of $TIMEOUT_SEC s exceeded"
    fi
    printf '%s\t%s\t%s\n' "$code" "$elapsed" "$why" > "$META_DIR/$name"

    # The line is printed WHOLE with one `printf`: in a batch the neighbours write
    # to the same terminal, and output built from several calls would interleave.
    if (( code == 0 )); then
        printf '%-24s ok       %4d s\n' "$name" "$elapsed"
    else
        printf '%-24s FAIL     %4d s  (code %d)\n    %s\n    output: %s\n' \
            "$name" "$elapsed" "$code" "$why" "$log"
    fi
}

any_failed() {
    local name code
    for name in "$@"; do
        [[ -f "$META_DIR/$name" ]] || continue
        IFS=$'\t' read -r code _ _ < "$META_DIR/$name"
        (( code != 0 )) && return 0
    done
    return 1
}

# The batch first, then the loners: loners measure time, and running them while
# the batch holds the machine would measure the batch.
running=0
for path in "${BATCH[@]}"; do
    while (( running >= JOBS )); do wait -n; running=$(( running - 1 )); done
    run_one "$path" &
    running=$(( running + 1 ))
done
while (( running > 0 )); do wait -n; running=$(( running - 1 )); done

BATCH_NAMES=()
for path in "${BATCH[@]}"; do BATCH_NAMES+=("$(basename "$path" .py)"); done
if (( FAIL_FAST )) && any_failed "${BATCH_NAMES[@]}"; then
    LONE=()
fi

for path in "${LONE[@]}"; do
    run_one "$path"
    (( FAIL_FAST )) && any_failed "$(basename "$path" .py)" && break
done

# The table is built in SELECTED order, not finish order: a run read by eye
# must have the same order every time.
NAMES=(); RESULTS=(); TIMES=(); TAILS=()
FAILED=0
for path in "${SELECTED[@]}"; do
    name="$(basename "$path" .py)"
    [[ -f "$META_DIR/$name" ]] || continue   # not reached: fail-fast stopped the run
    IFS=$'\t' read -r code elapsed why < "$META_DIR/$name"
    NAMES+=("$name"); TIMES+=("$elapsed")
    if (( code == 0 )); then
        RESULTS+=("ok"); TAILS+=("")
    else
        FAILED=$(( FAILED + 1 )); RESULTS+=("fail"); TAILS+=("$why")
    fi
done

echo
echo "==================== summary ===================="
for (( i = 0; i < ${#NAMES[@]}; i++ )); do
    if [[ "${RESULTS[$i]}" == "ok" ]]; then
        printf '  [+] %-24s %4d s\n' "${NAMES[$i]}" "${TIMES[$i]}"
    else
        printf '  [!] %-24s %4d s  %s\n' "${NAMES[$i]}" "${TIMES[$i]}" "${TAILS[$i]}"
    fi
done
echo
if (( FAILED )); then
    echo "FAILED $FAILED of ${#NAMES[@]}. Full output: $LOG_DIR"
    exit 1
fi
echo "All ${#NAMES[@]} checks are green."
