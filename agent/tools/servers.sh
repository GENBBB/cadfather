#!/usr/bin/env bash
# vLLM servers: bring up, inspect, shut down.
#
#   tools/servers.sh                  table: which, where, alive or not, how much they hold
#   tools/servers.sh up <config>      bring up this config's servers and leave them running
#   tools/servers.sh up <config> --only generation   bring up one role, leave the other alone
#   tools/servers.sh ps               the same as raw `ps` lines, unprocessed
#   tools/servers.sh stop --all       shut down everything in the directory
#   tools/servers.sh stop generation  shut down by role, PID or port
#
# `up` exists because servers are sometimes needed by something other than a run:
# an external driver may need live endpoints BEFORE it starts its own run.
#
# The config is mandatory and has no default on purpose: "bring up the servers"
# without saying which is a bet on which checkpoint will occupy the GPUs for
# hours. Such a mistake is not visible right away: `--served-model-name` is the
# same across checkpoints by construction, so a run would silently go to the
# wrong weights.
#
# Flags after the config go to `run_system.sh` as is; that is how `--only <role>`
# works: roles are brought up separately so that one that misses
# `server_timeout_sec` does not take an already running neighbour down with it.
#
# This command does NOT bring anything up itself: it hands the work to
# `run_system.sh --servers-only`. Not out of laziness: in the normal mode that
# same script also shuts its servers down (by the PIDs of its children and an
# exit trap), and a copy of the bring-up logic here would be a second source of
# truth about environments, vLLM flags and the record directory.
#
# Why it is needed at all. Servers in `--keep-servers` mode go into a separate
# session (`setsid`): otherwise SIGHUP on an ssh drop would kill exactly what the
# mode promises to keep. The price is a lost terminal, and `ps` without flags
# selects processes **by the caller's terminal**: running servers are not visible
# in it at all, and the mode looks as if it started nothing. They can be found by
# PID, but it is printed once and goes away with the shell, so every server that
# reaches readiness leaves a record in the directory.
#
# The directory is `CAD_AGENT_SERVERS_DIR`, by default `/tmp/cad_agent_servers_$USER`:
# it lives on the same machine as the servers and is wiped together with it.
# A record is a pointer, not the truth: liveness is checked with `kill -0`,
# readiness by probing `/v1/models`, and the record of a dead server is removed
# on the spot.
#
# Servers brought up bypassing this script are unknown to the directory, so the
# list is supplemented by a direct `ps` scan for the launch string (`vllm serve`).
# vLLM's children (`VLLM::EngineCore` and the like) are deliberately excluded: the
# process to stop and count is the one that holds the port.

set -uo pipefail

SERVERS_DIR="${CAD_AGENT_SERVERS_DIR:-/tmp/cad_agent_servers_${USER}}"
PROBE_TIMEOUT=3
STOP_TIMEOUT=20

AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SYSTEM="$AGENT_DIR/run_system.sh"

usage() { awk 'NR==1 {next} /^#/ {print; next} {exit}' "${BASH_SOURCE[0]}"; }

# Record fields. Read line by line rather than with `source`: the contents of a
# file in the directory are data, and there is no reason to execute them as code.
read_record() {
    local file="$1" key value
    ROLE=""; PID=""; URL=""; MODEL=""; DEVICES=""; LOG=""; CONFIG=""; RUN=""
    while IFS='=' read -r key value; do
        case "$key" in
            ROLE)    ROLE="$value" ;;
            PID)     PID="$value" ;;
            URL)     URL="$value" ;;
            MODEL)   MODEL="$value" ;;
            DEVICES) DEVICES="$value" ;;
            LOG)     LOG="$value" ;;
            CONFIG)  CONFIG="$value" ;;
            RUN)     RUN="$value" ;;
        esac
    done < "$file"
}

alive() { [[ -n "${1:-}" ]] && kill -0 "$1" 2>/dev/null; }

# Signal the server's process GROUP if it leads its own group. This is how
# `--keep-servers` works: `setsid` execs in place, so the record's PID is the
# session leader, and `VLLM::EngineCore` and other children live in the same
# group. A signal to a single PID reached only the frontend, and a frontend
# stuck at startup did not take the engine down at once, so an orphaned
# `EngineCore` held the GPU for minutes. Without `setsid` the server sits in the
# script's own group, and a group signal would kill the script, so the signal
# goes to the single PID as before.
leads_group() {
    [[ -n "${1:-}" ]] || return 1
    [[ "$(ps -o pgid= -p "$1" 2>/dev/null | tr -d ' ')" == "$1" ]]
}

# Signal target: `-PID` (the whole group) for a group leader, otherwise the PID
# itself. Computed BEFORE signalling: once the leader is dead, `ps` can no longer
# be asked about its group.
signal_target() { if leads_group "$1"; then echo "-$1"; else echo "$1"; fi; }

# Whether the target is alive: the process or anyone in its group.
target_alive() { kill -0 -- "$1" 2>/dev/null; }

answers() { curl -fs --max-time "$PROBE_TIMEOUT" "${1%/}/models" >/dev/null 2>&1; }

records() {
    [[ -d "$SERVERS_DIR" ]] || return 0
    shopt -s nullglob
    local file
    for file in "$SERVERS_DIR"/*.env; do echo "$file"; done
    shopt -u nullglob
}

# The `ps` line for one PID: what plain `ps` without flags does not show.
ps_line() {
    ps -o pid=,etime=,%cpu=,rss=,args= -p "$1" 2>/dev/null | head -n 1
}

# Servers not in the directory: started by hand, or the record was lost along
# with `/tmp`. Found by launch string; for a real vLLM it is `.../vllm serve ...`.
strays() {
    local known="$1"
    ps -u "$USER" -o pid=,args= 2>/dev/null | while read -r pid args; do
        [[ "$args" == *"vllm serve"* ]] || continue
        [[ ",$known," == *",$pid,"* ]] && continue
        echo "$pid ${args}"
    done
}

list_servers() {
    local found=0 pids=() line
    local file
    while read -r file; do
        [[ -n "$file" ]] || continue
        read_record "$file"
        if ! alive "$PID"; then
            # The record of a dead server is a lie about busy GPUs, so it is removed
            # right here: the directory must answer about now, not about the past.
            echo "${ROLE:-?} PID ${PID:-?} - dead, entry removed (${URL:-?})"
            rm -f "$file"
            continue
        fi
        found=$(( found + 1 ))
        pids+=("$PID")
        local state="ready"
        answers "$URL" || state="alive but not answering"
        echo "${ROLE:-?}  PID ${PID}  ${state}"
        echo "  address: ${URL}"
        echo "  weights: ${MODEL}"
        echo "  devices: ${DEVICES:-<all>}"
        line="$(ps_line "$PID")"
        echo "  in ps:   ${line:0:160}"
        [[ -n "$LOG" ]]    && echo "  log:     ${LOG}"
        [[ -n "$RUN" ]]    && echo "  run:     ${RUN}"
        [[ -n "$CONFIG" ]] && echo "  config:  ${CONFIG}"
    done < <(records)

    local known=""
    (( ${#pids[@]} )) && known="$(IFS=,; echo "${pids[*]}")"
    local stray_lines; stray_lines="$(strays "$known")"
    if [[ -n "$stray_lines" ]]; then
        echo
        echo "Not in the registry (started bypassing $0, or the entry was lost):"
        while read -r line; do
            [[ -n "$line" ]] && echo "  ${line:0:160}"
        done <<< "$stray_lines"
    fi

    if (( found == 0 )); then
        if [[ -z "$stray_lines" ]]; then
            echo "The registry is empty: no servers left by --keep-servers ($SERVERS_DIR)."
            echo "Start:       $0 up <config>"
        fi
        return 0
    fi
    echo
    echo "All at once: ps -o pid,etime,%cpu,rss,args -p ${known}"
    echo "Stop:        $0 stop --all"
}

# Bring up the config's servers and leave them standing. The only thing of its
# own here is argument checking: `run_system.sh --servers-only` does the bring-up.
up_servers() {
    if (( ! $# )); then
        echo "Nothing to start: give a config." >&2
        echo "  for example: $0 up agent/configs/dialogue_lean.yaml" >&2
        echo "  configs:  ls $AGENT_DIR/configs/*.yaml" >&2
        return 2
    fi
    # The config must come first. Without this check `servers.sh up --keep-servers`
    # would go to `run_system.sh` with no config, which would take its own default,
    # and the user would get the wrong weights brought up with no error message.
    if [[ "$1" == -* ]]; then
        echo "The first argument must be a config, not a flag: $1" >&2
        echo "  for example: $0 up agent/configs/dialogue_lean.yaml $1" >&2
        return 2
    fi
    local config="$1"; shift
    [[ -x "$RUN_SYSTEM" || -f "$RUN_SYSTEM" ]] || {
        echo "$RUN_SYSTEM not found - the start logic lives there" >&2; return 2; }
    # `exec`: this process has nothing left to do, and the bring-up's exit code and
    # output must reach the caller as its own.
    exec bash "$RUN_SYSTEM" "$config" --servers-only "$@"
}

# Raw `ps`: the same, without our processing, for checking by eye.
ps_servers() {
    local pids=() file
    while read -r file; do
        [[ -n "$file" ]] || continue
        read_record "$file"
        alive "$PID" && pids+=("$PID")
    done < <(records)
    if (( ! ${#pids[@]} )); then
        echo "No live entries in the registry ($SERVERS_DIR)."
        return 0
    fi
    ps -o pid,etime,%cpu,rss,args -p "$(IFS=,; echo "${pids[*]}")"
}

# Stop target: `--all`, PID, role or port. Role and port because nobody
# remembers a PID by heart, while the role is in the record and in the config.
matches() {
    local target="$1"
    [[ "$target" == "--all" ]] && return 0
    [[ "$target" == "$PID" ]] && return 0
    [[ "$target" == "$ROLE" ]] && return 0
    [[ "$URL" == *":$target"* ]] && return 0
    return 1
}

stop_servers() {
    (( $# )) || { echo "What to stop: --all, a PID, a role or a port." >&2; return 2; }
    local hit=0 file target killed=()
    while read -r file; do
        [[ -n "$file" ]] || continue
        read_record "$file"
        for target in "$@"; do
            matches "$target" || continue
            hit=$(( hit + 1 ))
            if alive "$PID"; then
                target="$(signal_target "$PID")"
                if [[ "$target" == -* ]]; then
                    echo "Stopping ${ROLE} PID ${PID} with its process group (${URL})"
                else
                    echo "Stopping ${ROLE} PID ${PID} (${URL})"
                fi
                kill -TERM -- "$target" 2>/dev/null || true
                killed+=("$target:$file")
            else
                echo "${ROLE} PID ${PID} is already dead - removing the entry."
                rm -f "$file"
            fi
            break
        done
    done < <(records)

    (( hit )) || { echo "No matching servers in the registry ($SERVERS_DIR)." >&2; return 1; }

    # A record is removed only after the process dies (for a group leader, after
    # the WHOLE group dies): removing it earlier would lose the PID of a server (or
    # its engine) that refused to die while still holding GPUs.
    (( ${#killed[@]} )) || return 0
    # The poll step grows for the same reason as in `wait_for_servers`: a process
    # that dies within tens of milliseconds must not cost a whole second.
    local deadline=$(( SECONDS + STOP_TIMEOUT )) entry pid path left poll_sec=0.05
    while (( SECONDS < deadline )); do
        left=0
        for entry in "${killed[@]}"; do
            pid="${entry%%:*}"; path="${entry#*:}"
            if target_alive "$pid"; then left=1; else rm -f "$path"; fi
        done
        (( left )) || break
        sleep "$poll_sec"
        poll_sec="$(awk -v p="$poll_sec" 'BEGIN { p *= 2; print (p > 1 ? 1 : p) }')"
    done
    for entry in "${killed[@]}"; do
        pid="${entry%%:*}"; path="${entry#*:}"
        if target_alive "$pid"; then
            echo "${pid#-} (target ${pid}) did not die within ${STOP_TIMEOUT} s - entry kept, force with: kill -9 -- ${pid}" >&2
        else
            rm -f "$path"
        fi
    done
}

case "${1:-list}" in
    list|"")    list_servers ;;
    up)         shift; up_servers "$@" ;;
    ps)         ps_servers ;;
    stop)       shift; stop_servers "$@" ;;
    -h|--help)  usage ;;
    *)          echo "Unknown command: $1" >&2; usage >&2; exit 2 ;;
esac
