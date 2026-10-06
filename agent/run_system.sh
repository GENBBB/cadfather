#!/usr/bin/env bash
# Start the vLLM servers and run an experiment.
#
#   ./run_system.sh                           default config (configs/dialogue_lean.yaml)
#   ./run_system.sh CFG                       a specific config
#   ./run_system.sh CFG --no-servers          servers are already up
#   ./run_system.sh CFG --keep-servers        start the missing ones and leave them running
#   ./run_system.sh CFG --servers-only        start servers and exit, no run
#   ./run_system.sh CFG --servers-only --only generation      one role
#   ./run_system.sh CFG --servers-only --server-timeout 1800  wait longer
#   ./run_system.sh CFG --split control       same config, another sample
#   ./run_system.sh CFG --name my_run         own directory name
#
# The run directory is named `launch.run_name` from the config (`--name` overrides it).
# A repeated launch with the same name does not overwrite the previous directory but takes
# the first free name with a suffix: `name`, `name_2`, `name_3`. The name is claimed by an
# atomic `mkdir`, so two launches started at the same time cannot land in one directory.
#
# `--split` overrides `experiment.subsample.split` from the config: evo, control or test.
# It exists so that one config need not be copied for different samples: a copy that must
# match in everything but one word silently drifts apart. The effective value is recorded
# in the run directory's `config.json`.
#
# `--keep-servers` is the mode for a series of runs: only servers that do not respond yet
# are started, and none is stopped at the end. The next run on the same config starts at
# once, without minutes of weight loading. Unlike `--no-servers`, which starts nothing and
# fails if there are no servers, this one brings the picture to the required state.
#
# Only servers that **survived until ready** are kept. A dead one or one that did not come
# up in time is stopped as in the normal mode: it cannot be reused anyway (the `/v1/models`
# probe will not answer) but it holds the cards. The promise of the mode is "do not stop
# what works", not "do not stop anything".
#
# `--servers-only` is `--keep-servers` without a run: start the servers, wait for
# readiness, record them in the registry and exit. It implies `--keep-servers`: stopping
# what was started on exit would mean there was no point in starting it. It conflicts with
# `--no-servers` just like `--keep-servers` does. It does not combine with `--split`: the
# sample is read by the run, which does not exist here, and a silently accepted flag would
# make the user think the conditions were set when nothing was set.
#
# `--only <role>` starts one role from the config and leaves the other alone. Only together
# with `--servers-only`: for a run the set of servers is defined by the config, and "a run
# without the assistant" is another config, not a command-line flag. Roles start at the same
# time and readiness is awaited in one loop, so the first one that misses `server_timeout_sec`
# takes an already started neighbor with it: `server_failed` exits the script and cleanup
# stops everything spawned, because `SERVERS_READY` is still zero. Separately, each role
# survives until ready on its own, is recorded in the servers registry, and the next launch
# with `--keep-servers` reuses it.
#
# `--server-timeout <sec>` overrides `launch.server_timeout_sec` for one launch. The first
# start of a role can take much longer than later ones (kernel JIT builds happen with a cold
# cache once), while the config's timeout covers the usual case. Editing the config does not
# help: for paired measurements the configs must differ by exactly the declared number of
# lines, and "started today with a different timeout" is not among them.
#
# Logs of such a start-up go not to a run directory but to `runs_root/servers/`: a run
# directory without a run is a lie in `work_dirs`, and the `RUN` field of the servers
# registry entry stays empty for the same reason. The front end of this mode is
# `tools/servers.sh up <config>`; the start-up itself lives here and only here, because the
# servers (in the normal mode) are stopped by the same process that spawned them.
#
# A reused server is checked via `/v1/models`: vLLM returns `root` there (the path to the
# weights), and it must match `model.*` from the config. Checking by name alone is not
# possible: `--served-model-name` coincides across checkpoints by construction, which is
# why this script reads the config.
#
# In this mode servers are launched through `setsid`, as a separate session. Otherwise a
# signal to the process group would stop exactly what is promised to be kept: on an ssh
# drop SIGHUP goes to the whole group, and the servers would leave with the script.
# Ctrl-C is not involved: bash sets SIGINT to SIG_IGN for asynchronous children of a
# non-interactive shell, so they do not see it.
#
# The price of a separate session is a lost terminal, and plain `ps` shows only processes
# of **its own** terminal: the kept servers are not visible there at all although they run.
# Therefore every server that survived until ready leaves an entry in the directory
# `CAD_AGENT_SERVERS_DIR` (default `/tmp/cad_agent_servers_$USER`), and they can be
# inspected and stopped with `tools/servers.sh`, which also prints a ready `ps -p ...` line.
#
# Everything that used to be hardcoded here (model paths, ports, vLLM flags, the runs
# directory) is taken from the **same config** that the run itself reads later (sections
# `model`, `server`, `launch`). Otherwise the servers could start on one model while the run
# addressed another: the `--served-model-name` values coincide, and a substitution went
# unnoticed.
#
# Environments. There are up to three and they are independent:
#   launch.servers.<role>.vllm_env - this server's own environment;
#   launch.vllm_env                - default environment of the servers;
#   launch.run_env                 - where the run itself comes from.
# An empty value means "whatever is active in the current shell".
# The generator and the assistant may be built for different vLLM versions; then each has
# its own environment and variables (launch.servers.<role>.env on top of launch.env).
# A server's variables are seen only by it: a common export would make one build's setting
# visible to another, which is exactly what must be avoided.
# The config is parsed by the interpreter from --python, $CAD_AGENT_PYTHON or PATH.
#
# Servers start IN PARALLEL: they use different cards and weight loading takes minutes, so
# waiting for them one by one would add those minutes up. All are started first, then comes
# a common wait loop; each server's readiness deadline is counted from its own start.

set -euo pipefail

AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$AGENT_DIR")"

CONFIG_PATH=""
SPLIT=""
RUN_NAME_OVERRIDE=""
PARSER_PYTHON="${CAD_AGENT_PYTHON:-python3}"
START_SERVERS=1
KEEP_SERVERS=0
# Start the servers and exit without starting a run. A separate flag rather than "a run
# with an empty set": an empty run would still create a directory, read the sample and
# require the run environment, i.e. fail for reasons unrelated to starting the servers.
SERVERS_ONLY=0
# Which role to start if not all. Empty means all described in the config; otherwise
# exactly the one named.
ONLY_ROLE=""
# Readiness timeout for one launch; empty means the one from the config.
SERVER_TIMEOUT_OVERRIDE=""
# Whether the started servers survived until ready. Separate from `KEEP_SERVERS`, because
# these are different things: the mode asks "whether to keep", this flag says "whether
# there is anything to keep". One number for two lifetimes has already cost us once.
SERVERS_READY=0

while (( $# )); do
    case "$1" in
        --python)       PARSER_PYTHON="$2"; shift 2 ;;
        --no-servers)   START_SERVERS=0; shift ;;
        --keep-servers) KEEP_SERVERS=1; shift ;;
        --servers-only) SERVERS_ONLY=1; shift ;;
        --only)         ONLY_ROLE="$2"; shift 2 ;;
        --server-timeout) SERVER_TIMEOUT_OVERRIDE="$2"; shift 2 ;;
        --split)        SPLIT="$2"; shift 2 ;;
        --name)         RUN_NAME_OVERRIDE="$2"; shift 2 ;;
        # The help is the whole header, up to the first non-comment line. It used to be a
        # line-number range, which silently fell behind the text: the header grew while
        # `-h` kept showing its former piece.
        -h|--help)      awk 'NR==1 {next} /^#/ {print; next} {exit}' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*)            echo "Unknown flag: $1" >&2; exit 2 ;;
        *)             CONFIG_PATH="$1"; shift ;;
    esac
done

if (( KEEP_SERVERS )) && (( ! START_SERVERS )); then
    echo "--keep-servers and --no-servers contradict each other: the first brings the servers" >&2
    echo "to the required state, the second forbids starting anything." >&2
    exit 2
fi

# Checked separately from the previous one and before `--servers-only` turns on
# `--keep-servers`: otherwise for the pair `--servers-only --no-servers` the script would
# complain about a flag the user did not write, and they would go and fix the wrong thing.
if (( SERVERS_ONLY )) && (( ! START_SERVERS )); then
    echo "--servers-only and --no-servers contradict each other: the first only starts" >&2
    echo "servers, the second forbids starting anything." >&2
    exit 2
fi

if (( SERVERS_ONLY )) && [[ -n "$SPLIT" ]]; then
    echo "--split makes no sense here: with --servers-only there is no run, and the split" >&2
    echo "is read by the run itself. Accepting the flag silently would leave the user" >&2
    echo "believing the conditions are set." >&2
    exit 2
fi

if [[ -n "$SERVER_TIMEOUT_OVERRIDE" ]]; then
    # Integer and positive only: zero or letters would give a timeout at which the very
    # first poll declares the server not up, i.e. a failure that looks like a model
    # problem.
    if [[ ! "$SERVER_TIMEOUT_OVERRIDE" =~ ^[1-9][0-9]*$ ]]; then
        echo "--server-timeout takes an integer number of seconds greater than zero (got: $SERVER_TIMEOUT_OVERRIDE)" >&2
        exit 2
    fi
fi

if [[ -n "$ONLY_ROLE" ]]; then
    case "$ONLY_ROLE" in
        generation|assistant) ;;
        *) echo "--only takes a server role: generation or assistant (got: $ONLY_ROLE)" >&2
           exit 2 ;;
    esac
    # Requiring `--servers-only` is not a formality: with a run the flag would mean
    # "start half of what is needed and go compute", i.e. a failure minutes later and not
    # where it was introduced.
    if (( ! SERVERS_ONLY )); then
        echo "--only works only together with --servers-only: the server set of a run" >&2
        echo "comes from the config, not the command line. To start one role separately:" >&2
        echo "  $0 <config> --servers-only --only $ONLY_ROLE" >&2
        exit 2
    fi
fi

# The "servers only" mode is `--keep-servers` without a run: what was started must
# survive the script's exit, otherwise there was no point in starting it.
if (( SERVERS_ONLY )); then
    KEEP_SERVERS=1
fi

# The value is checked here and not only in `run_experiment.py`: there it would surface
# only after the models are up, i.e. minutes later. The same reason the config is parsed
# first below.
if [[ -n "$SPLIT" ]]; then
    case "$SPLIT" in
        evo|control|test) ;;
        *) echo "Unknown split: $SPLIT (expected evo, control or test)" >&2; exit 2 ;;
    esac
fi

CONFIG_PATH="${CONFIG_PATH:-$AGENT_DIR/configs/dialogue_lean.yaml}"
[[ -f "$CONFIG_PATH" ]] || { echo "No such config: $CONFIG_PATH" >&2; exit 1; }
CONFIG_PATH="$(cd "$(dirname "$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")"

# The config is parsed once, before everything else: if something is wrong with it, it
# must be known before the models are started and not five minutes later.
PLAN_ARGS=(--config "$CONFIG_PATH")
# The name is validated by the same config parse as everything else: a bad name must
# surface before the models are started, and one check per system is enough.
[[ -n "$RUN_NAME_OVERRIDE" ]] && PLAN_ARGS+=(--run-name "$RUN_NAME_OVERRIDE")
eval "$("$PARSER_PYTHON" "$AGENT_DIR/tools/launch_plan.py" "${PLAN_ARGS[@]}")"

# After the config parse, not instead of it: the run directory's `config.json` and the
# failure messages must carry the number actually waited by.
if [[ -n "$SERVER_TIMEOUT_OVERRIDE" ]]; then
    echo "Server readiness deadline: ${SERVER_TIMEOUT_OVERRIDE} s (config: ${SERVER_TIMEOUT_SECONDS})"
    SERVER_TIMEOUT_SECONDS="$SERVER_TIMEOUT_OVERRIDE"
fi

[[ "$RUNS_ROOT" = /* ]] || RUNS_ROOT="$REPO_ROOT/$RUNS_ROOT"
# A start-up without a run still writes logs, but they do not become a run directory: an
# empty `name_3` next to real runs reads as a run that lost its results. A separate
# `servers/` branch tells the truth and does not take a name in the run series. Claiming a
# name inside it is the same atomic `mkdir` below: two start-ups begun side by side must
# not write logs over each other.
if (( SERVERS_ONLY )); then
    RUNS_ROOT="$RUNS_ROOT/servers"
fi
mkdir -p "$RUNS_ROOT"

# Run directory: `RUN_NAME`, and on a repeated launch the first free name with a suffix
# (`_2`, `_3`, ...). Taken suffixes are skipped, so a run never writes into someone else's
# directory even if the series of names has gaps.
#
# `mkdir` without `-p` matters: it does **not** stay silent when the directory exists and
# claims the name atomically. A separate "does it exist" check would leave a window in
# which two launches started side by side get one directory and write over each other.
OUTPUT_DIR=""
for (( RUN_SUFFIX = 1; RUN_SUFFIX <= 999; RUN_SUFFIX++ )); do
    CANDIDATE="${RUNS_ROOT}/${RUN_NAME}"
    (( RUN_SUFFIX > 1 )) && CANDIDATE="${CANDIDATE}_${RUN_SUFFIX}"
    if MKDIR_ERROR=$(mkdir "$CANDIDATE" 2>&1); then
        OUTPUT_DIR="$CANDIDATE"
        break
    fi
    # The name is taken, so try the next one. Any other trouble (no permission, no runs
    # directory) is an error, not a reason to keep iterating over names.
    if [[ ! -d "$CANDIDATE" ]]; then
        echo "Cannot create the run directory: $MKDIR_ERROR" >&2
        exit 1
    fi
done
if [[ -z "$OUTPUT_DIR" ]]; then
    echo "All names from $RUN_NAME to ${RUN_NAME}_999 in $RUNS_ROOT are taken." >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR/logs"

echo "Config:          $CONFIG_PATH"
if (( SERVERS_ONLY )); then
    echo "Server logs:     $OUTPUT_DIR"
else
    echo "Run directory:   $OUTPUT_DIR"
fi

# Registry of "what is up now". A reference to a kept server must outlive the script
# itself: the PID is printed once, while servers get asked about a day later from another
# shell where the printed line is gone, and plain `ps` does not show them (own session,
# no terminal).
SERVERS_DIR="${CAD_AGENT_SERVERS_DIR:-/tmp/cad_agent_servers_${USER}}"

export TMPDIR="/tmp/vllm_tmp_${USER}"
export VLLM_RPC_BASE_PATH="/tmp/vllm_rpc_${USER}"
mkdir -p "$TMPDIR" "$VLLM_RPC_BASE_PATH"

# Environment variables from launch.env (e.g. HF_HOME): both the servers and the run see
# them. Keep them in the config rather than in the script text, so an edit does not stay
# on one machine.
for pair in "${LAUNCH_ENV[@]}"; do
    export "${pair?}"
    echo "Environment:     ${pair%%=*}"
done

PIDS=()

# The signal goes to the server's process GROUP if it leads its own group. This is how
# `--keep-servers` works: `setsid` execs in place, the entry's PID is the session leader,
# and `VLLM::EngineCore` and other children live in the same group. A signal to a single PID
# reached only the front end, and one hanging at start did not take the engine at once: an
# orphaned `EngineCore` held the card for minutes more. Without `setsid` the server sits in
# the script's own group, and a group signal would stop the script too; then the signal goes
# to a single PID.
stop_server_pid() {
    local pid="$1"
    if [[ "$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')" == "$pid" ]]; then
        kill -TERM -- "-$pid" 2>/dev/null || true
    else
        kill "$pid" 2>/dev/null || true
    fi
}

cleanup() {
    if (( ! ${#PIDS[@]} )); then
        # A run where everything was reused: the script started no servers of its own, so there
        # is nothing to stop, but the servers are up and that must be reported. Otherwise the
        # mode is silent exactly where it works best.
        if (( KEEP_SERVERS )); then
            echo "No own servers were started: everything was reused and keeps running."
            echo "  what is up:  ${AGENT_DIR}/tools/servers.sh"
        fi
        return 0
    fi
    if (( KEEP_SERVERS && SERVERS_READY )); then
        # Exactly what the mode exists for. PIDs are printed because nobody is left to stop
        # them: the script exits, the servers stay. Plain `ps` will not show them: they are
        # in their own session without a terminal, and such a `ps` selects processes by the
        # caller's terminal. So a ready line with `-p` is printed, not just a list of PIDs.
        local csv; csv="$(IFS=,; echo "${PIDS[*]}")"
        echo "Servers stay up (--keep-servers): PID ${PIDS[*]}"
        echo "  inspect:     ps -o pid,etime,%cpu,rss,args -p ${csv}"
        echo "  what is up:  ${AGENT_DIR}/tools/servers.sh"
        echo "  stop:        ${AGENT_DIR}/tools/servers.sh stop --all   (or kill -- -PID: the group with the engine)"
        return 0
    fi
    echo "Stopping vLLM servers..."
    for pid in "${PIDS[@]}"; do stop_server_pid "$pid"; done
}
trap cleanup EXIT

# Path to an executable in the environment: empty means take it from PATH.
in_env() {
    local env_path="$1" binary="$2"
    if [[ -n "$env_path" ]]; then
        [[ -x "$env_path/bin/$binary" ]] || { echo "No $binary in the environment $env_path" >&2; exit 1; }
        echo "$env_path/bin/$binary"
    else
        command -v "$binary" || { echo "No $binary in PATH" >&2; exit 1; }
    fi
}

# Started but not yet ready servers. Filled by `start_server`, drained by
# `wait_for_servers`: waiting is separated from starting precisely for parallelism.
PENDING_ROLES=()
PENDING_URLS=()
PENDING_LOGS=()
PENDING_PIDS=()
PENDING_DEADLINES=()
PENDING_MODELS=()
PENDING_DEVICES=()

# An already running server: does it respond and **is it the right one**. The check is the
# `root` from `/v1/models`, the weights path that vLLM puts there. Comparing by `id` is
# pointless: that is `--served-model-name`, which is the same across checkpoints by
# construction, exactly the silent substitution for which this script reads the config.
reuse_server() {
    local role="$1" model="$2" url="$3"
    local probe="${url%/}/models"

    curl -fs "$probe" >/dev/null 2>&1 || return 1

    local served
    served="$(curl -fs "$probe" 2>/dev/null | "$PARSER_PYTHON" -c '
import json, sys
try:
    payload = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for item in payload.get("data") or []:
    print(item.get("root") or "", end="")
    break
' 2>/dev/null || true)"

    if [[ -z "$served" ]]; then
        # Something responds, but what is unknown. Silently reusing is not possible: the
        # promise "the same checkpoint" is not backed by anything here.
        echo "ERROR: something answers at ${url}, but ${probe} does not report the weights path." >&2
        echo "Check the server by hand, or stop it and run without --keep-servers." >&2
        exit 1
    fi
    if [[ "$served" != "$model" ]]; then
        echo "ERROR: the wrong server is up at ${url}." >&2
        echo "  config expects: $model" >&2
        echo "  answering:      $served" >&2
        exit 1
    fi

    echo "Server ${role} is already up at ${url}; reusing it."
    echo "  weights:   $served"
    return 0
}

server_failed() {
    local index="$1" reason="$2"
    echo "ERROR: server ${PENDING_ROLES[index]} ${reason}. Last 80 lines of ${PENDING_LOGS[index]}:" >&2
    tail -n 80 "${PENDING_LOGS[index]}" >&2 || true
    exit 1
}

wait_for_servers() {
    (( ${#PENDING_ROLES[@]} )) || return 0
    echo "Waiting for servers: ${PENDING_ROLES[*]}"

    local -a ready=()
    local i
    local remaining=${#PENDING_ROLES[@]}
    for (( i = 0; i < ${#PENDING_ROLES[@]}; i++ )); do ready[i]=0; done

    # The poll step GROWS rather than being fixed. A fixed 5 s took its 5 s even from a
    # server that already responds: the first pass inevitably lands right after launch,
    # when the port is not listening yet, and readiness was noticed only on the second. For
    # a real vLLM this cost up to 5 s at start-up, and for the checks run 5 s on EVERY
    # start-up. Backing off from 0.2 s to the same 5 s: a fast server is caught within
    # fractions of a second, a slow one pays a handful of extra curls in the first seconds.
    # `launch_check` also knows the step cap (POLL_INTERVAL_SEC) and computes the wall
    # budget from it, so the number must be changed in both places.
    local poll_sec=0.2

    while (( remaining > 0 )); do
        for (( i = 0; i < ${#PENDING_ROLES[@]}; i++ )); do
            if (( ready[i] )); then
                continue
            fi
            if curl -fs "${PENDING_URLS[i]}" >/dev/null; then
                echo "Server ${PENDING_ROLES[i]} is ready."
                ready[i]=1
                remaining=$(( remaining - 1 ))
                continue
            fi
            # Waiting for a dead process is pointless: without this check the script sat out
            # the whole timeout although the answer was in the log from the first second.
            # With parallel start-up this matters more: polling covers all at once, so a
            # failed neighbor is seen immediately and not after the previous one comes up.
            if ! kill -0 "${PENDING_PIDS[i]}" 2>/dev/null; then
                server_failed "$i" "died before coming up"
            fi
            # Each has its own deadline, counted from its own launch. A common countdown
            # would give the second server the less time the longer the first took to come
            # up, while they come up at the same time.
            if (( $(date +%s) >= PENDING_DEADLINES[i] )); then
                server_failed "$i" "did not come up within ${SERVER_TIMEOUT_SECONDS} s"
            fi
        done
        if (( remaining > 0 )); then
            sleep "$poll_sec"
            poll_sec="$(awk -v p="$poll_sec" 'BEGIN { p *= 2; print (p > 5 ? 5 : p) }')"
        fi
    done
}

start_server() {
    local role="$1" model="$2" url="$3" env_path="$4" devices="$5" vars_name="$6"; shift 6
    local log_path="$OUTPUT_DIR/logs/${role}_vllm.log"
    local vllm; vllm="$(in_env "$env_path" vllm)"
    # This server's variables go by reference to an array, not a copy: the array name is
    # built from the role, and passing its contents positionally would mix them with the
    # vllm flags.
    local -n server_vars="$vars_name"

    echo "Launching ${role}: $(basename "$model")"
    echo "  env:       ${env_path:-<active>}"
    echo "  devices:   ${devices:-<all>}"
    if (( ${#server_vars[@]} )); then
        echo "  variables: ${server_vars[*]%%=*}"
    fi

    # `env` instead of export: the assignments live only in this process. Plus the
    # environment's bin goes ahead of PATH: vLLM starts neighbors (ray and so on), and they
    # must come from the same environment as vLLM itself.
    # In `--keep-servers` mode the server goes into its own session: without that it stays
    # in the script's process group, and a group signal (SIGHUP on an ssh drop, SIGTERM from
    # someone else's cleanup) would stop it together with the script, i.e. exactly what the
    # mode promises to keep.
    # `$!` still remains the server's PID: in a non-interactive shell a background job is not
    # a group leader, and setsid execs in place without a fork.
    local -a detach=()
    (( KEEP_SERVERS )) && detach=(setsid)

    CUDA_VISIBLE_DEVICES="$devices" "${detach[@]}" env "${server_vars[@]}" \
        PATH="${env_path:+$env_path/bin:}$PATH" \
        "$vllm" serve "$model" "$@" \
        > "$log_path" 2>&1 &
    PIDS+=($!)

    PENDING_ROLES+=("$role")
    PENDING_URLS+=("${url%/}/models")
    PENDING_LOGS+=("$log_path")
    PENDING_PIDS+=("${PIDS[-1]}")
    PENDING_DEADLINES+=($(( $(date +%s) + SERVER_TIMEOUT_SECONDS )))
    PENDING_MODELS+=("$model")
    PENDING_DEVICES+=("$devices")
}

# The entry for a server that stays alive is the only trace by which it will be found
# after the script exits. It is written only in `--keep-servers` and only after readiness:
# one that did not survive until then is stopped, and an entry for it would be a lie, the
# kind that makes the cards look occupied by someone else's process.
record_kept_servers() {
    (( ${#PENDING_ROLES[@]} )) || return 0
    mkdir -p "$SERVERS_DIR"
    # The `RUN` field is the run the servers belong to, and in `--servers-only` there is
    # none: there `OUTPUT_DIR` holds only logs. Writing it there would make
    # `tools/servers.sh` print "run: ..." for a directory in which there was no run: one
    # field for two different events.
    local run_dir="$OUTPUT_DIR"
    (( SERVERS_ONLY )) && run_dir=""
    local i url key
    for (( i = 0; i < ${#PENDING_ROLES[@]}; i++ )); do
        url="${PENDING_URLS[i]%/models}"
        # The file name is by address, not by role: a role repeats from config to config,
        # while an occupied port is exactly one process. The entry is replaced as a whole: if
        # another server is at this address now, the former entry is a lie.
        key="$(printf '%s' "$url" | tr -c 'A-Za-z0-9._-' '_')"
        cat > "$SERVERS_DIR/${key}.env" <<EOF
ROLE=${PENDING_ROLES[i]}
PID=${PENDING_PIDS[i]}
URL=${url}
MODEL=${PENDING_MODELS[i]}
DEVICES=${PENDING_DEVICES[i]}
LOG=${PENDING_LOGS[i]}
CONFIG=${CONFIG_PATH}
RUN=${run_dir}
EOF
    done
    echo "Registry of running servers: $SERVERS_DIR"
}

# `--only`: a role is either selected or absent from this launch. A separate function so
# that the condition stands once per role and reads next to the start-up itself.
role_selected() { [[ -z "$ONLY_ROLE" || "$ONLY_ROLE" == "$1" ]]; }

if (( START_SERVERS )); then
    # In the normal mode `reuse_server` is not called at all: the script starts its own
    # servers and stops them itself, while silently attaching to someone else's process is
    # exactly the substitution this script exists to prevent. An occupied port is a vLLM
    # error there, but `wait_for_servers` is not obliged to notice it: it reads readiness
    # from the port, and a foreign server answers on it.
    # `--only` for a role absent from the config is a typo, not "nothing to do": a silent
    # success would send the user to wait for a server nobody started.
    if [[ "$ONLY_ROLE" == generation && -z "$GENERATION_ENABLED" ]] \
    || [[ "$ONLY_ROLE" == assistant  && -z "$ASSISTANT_ENABLED"  ]]; then
        echo "--only $ONLY_ROLE: this role is absent from the config or disabled." >&2
        echo "  config: $CONFIG_PATH" >&2
        exit 2
    fi

    if ! role_selected generation; then
        echo "Server generation skipped (--only $ONLY_ROLE)."
    elif [[ -n "$GENERATION_ENABLED" ]]; then
        if ! { (( KEEP_SERVERS )) && reuse_server generation "$GENERATION_MODEL" "$GENERATION_URL"; }; then
            start_server generation "$GENERATION_MODEL" "$GENERATION_URL" \
                "$GENERATION_ENV" "$GENERATION_DEVICES" GENERATION_ENV_VARS "${GENERATION_ARGV[@]}"
        fi
    else
        echo "Server generation is not in the config; skipping."
    fi

    if ! role_selected assistant; then
        echo "Server assistant skipped (--only $ONLY_ROLE)."
    elif [[ -n "$ASSISTANT_ENABLED" ]]; then
        if ! { (( KEEP_SERVERS )) && reuse_server assistant "$ASSISTANT_MODEL" "$ASSISTANT_URL"; }; then
            start_server assistant "$ASSISTANT_MODEL" "$ASSISTANT_URL" \
                "$ASSISTANT_ENV" "$ASSISTANT_DEVICES" ASSISTANT_ENV_VARS "${ASSISTANT_ARGV[@]}"
        fi
    else
        echo "Server assistant is not started (absent from the config or enabled: false)."
    fi

    wait_for_servers
    # From here on a failed run does not stop the servers: they are working, and the next
    # attempt must find them up, which is what the mode is for.
    SERVERS_READY=1
    # An expanded `if`, not `&&`: under `set -e` a false condition at the end of the
    # branch would bring down the whole script with the branch's status.
    if (( KEEP_SERVERS )); then
        record_kept_servers
    fi
    # What follows is the run, and in this mode there is none. Exit via `exit 0` rather
    # than an abort: the cleanup on exit (`cleanup`) must run and print what is left
    # standing and how to stop it.
    if (( SERVERS_ONLY )); then
        echo "Servers are up; the run is not started (--servers-only)."
        echo "  what is up:  ${AGENT_DIR}/tools/servers.sh"
        echo "  logs:        ${OUTPUT_DIR}/logs"
        exit 0
    fi
else
    echo "Servers are not started (--no-servers): the run uses servers that are already up."
fi

RUN_PYTHON="$(in_env "$RUN_ENV" python)"
echo "Run python:      $RUN_PYTHON"

# An array, not a string: an empty argument in a string would collapse, and under
# `set -u` an empty array in an expansion is an error. Here the array is never empty.
RUN_ARGS=(--config "$CONFIG_PATH" --output-dir "$OUTPUT_DIR")
if [[ -n "$SPLIT" ]]; then
    RUN_ARGS+=(--split "$SPLIT")
fi

"$RUN_PYTHON" "$AGENT_DIR/run_experiment.py" "${RUN_ARGS[@]}"
