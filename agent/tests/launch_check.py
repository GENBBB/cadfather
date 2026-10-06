#!/usr/bin/env python3
"""Check of launch: two vLLM servers from different environments.

Run: ``python agent/tests/launch_check.py`` from the repository root.

Why it looks like this. `run_system.sh` is the one part of the system that cannot be
checked by import: Python parses the config but the shell starts the servers, and all the
interesting errors live at the seam: quoting, empty arrays, variables leaked to the wrong
process. So the real script is run here with fake `vllm` and `python`: every stub records
its own environment and arguments, and the checks read what was recorded.

What is checked:

1. the generator and the assistant take `vllm` **each from its own** environment;
2. a server's variables are seen only by it, while common ones (`launch.env`) by both;
3. in a server's PATH its own environment comes first (vLLM launches neighbors);
4. JSON inside flags survives passing through the shell;
5. the run is started by the interpreter from `launch.run_env`;
6. a typo in the keys of the `launch` section is an error, not a silent default;
7. the servers start in parallel, not one after another;
8. `--keep-servers` brings the picture to the desired state: starts what is missing,
   reuses what is already up, refuses a foreign checkpoint, and does not kill working
   servers either on exit or on a terminal hangup; what is left is visible in the
   `tools/servers.sh` registry and is killed by it, the whole process group, together with
   the engine, not by a single frontend;
9. the run directory is named by `launch.run_name`, a rerun takes a suffix and leaves the
   previous directory alone, `--name` overrides the config, a bad name is rejected before
   the servers are started;
10. `tools/servers.sh up` starts the servers WITHOUT a run: the run does not start at all,
    no run directory appears, the servers outlive the script and are visible in the
    registry, and a second `up` reuses them.
11. `--only <role>` starts one role: the other is not launched at all and its port is
    silent, and a second call brings it up beside the first without touching the first.
    Without `--servers-only` and with an unknown role the flag is rejected before startup.
12. `--server-timeout` overrides the readiness deadline from the config: with it a server
    that met the config deadline is declared not up, and the refusal names exactly the
    overriding number.

There are no real models or GPUs here: this checks the launch wrapper, not vLLM.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_SYSTEM = REPO_ROOT / "agent" / "run_system.sh"
SERVERS_TOOL = REPO_ROOT / "agent" / "tools" / "servers.sh"

# The config is parsed by an interpreter in which PyYAML works. A local conda Python 3.13
# with an old PyYAML fails on `collections.Hashable`, and patching it for the test is
# pointless, so the system one is used.
#
# `CAD_AGENT_PYTHON` is the same knob that `run_system.sh` itself reads. It is not for
# show: on a server `/usr/bin/python3` exists but has no PyYAML, and without the override
# the check would fail there on the choice of interpreter rather than on what it checks.
PARSER_PYTHON = (
    os.environ.get("CAD_AGENT_PYTHON")
    or ("/usr/bin/python3" if Path("/usr/bin/python3").exists() else sys.executable)
)

# The readiness poll step in `run_system.sh`. It is needed here only so that the time
# threshold is derived from it instead of being picked as a number: the wall measurement
# is quantized by exactly this step.
# The CEILING of the poll step in `wait_for_servers`, not the step itself: it backs off
# from 0.2 s to this number. The budget below is computed from the ceiling, i.e. with margin.
POLL_INTERVAL_SEC = 5

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if condition else 'FAIL'} {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


FAKE_VLLM = """#!/usr/bin/env bash
# Fake vLLM: records how and with what it was started, and answers 200.
# START is the moment the process starts, before any delay: from it the
# parallelism check sees when the server actually started, not when it became ready.
{{
  echo "SELF $0"
  echo "PID $$"
  echo "START $(date +%s.%N)"
  echo "ARGV $*"
  env | sort
}} > "{log}"

# Engine: like `VLLM::EngineCore`, it lives in the front end's process group and
# does not exit right after the front end dies, but a minute later. A signal to the
# front end PID alone would leave it hanging; only a signal to the whole group stops it.
bash -c 'while kill -0 "$1" 2>/dev/null; do sleep 0.2; done; sleep 60' engine $$ &
echo "ENGINE $!" >> "{log}"

# A model loads for minutes; the stub imitates this with a delay before it
# takes the port. Without the delay sequential and parallel start-up are
# indistinguishable: both are "ready" instantly.
trap 'exit 143' TERM
sleep "${{FAKE_VLLM_DELAY:-0}}" & wait $!

port=""
prev=""
for arg in "$@"; do
  [[ "$prev" == "--port" ]] && port="$arg"
  prev="$arg"
done

exec {python} - "$port" "$2" <<'PY'
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

# `root` is the weights path, exactly as a real vLLM reports it. It is how
# `--keep-servers` tells whether the right server is already up: `id` is
# `--served-model-name`, which is the same for different checkpoints by construction.
# Literal curly braces are not allowed here: the stub body goes through
# `str.format`, so dictionaries are built with `dict(...)`.
PAYLOAD = json.dumps(dict(
    object="list",
    data=[dict(id="fake", object="model", root=sys.argv[2])],
)).encode()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(PAYLOAD)

    def log_message(self, *args):
        pass


HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
PY
"""

FAKE_PYTHON = """#!/usr/bin/env bash
# Fake run interpreter: the run itself is not needed here.
{{
  echo "SELF $0"
  echo "PID $$"
  echo "ARGV $*"
  env | sort
}} > "{log}"
# A run can be long, and the Ctrl-C check sends the signal exactly in this phase:
# the servers are ready and the run is still going.
sleep "${{FAKE_RUN_DELAY:-0}}"
exit 0
"""


# A foreign server on the port: answers 200 but says nothing about weights. The real case
# is a service forgotten on this port or a vLLM of another version.
STRANGER_SERVER = """
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"object": "list", "data": []}')

    def log_message(self, *args):
        pass


HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
"""


def make_env(root: Path, name: str, binary: str, body: str, log: Path) -> Path:
    env_dir = root / "envs" / name
    (env_dir / "bin").mkdir(parents=True, exist_ok=True)
    path = env_dir / "bin" / binary
    path.write_text(body.format(log=log, python=PARSER_PYTHON), encoding="utf-8")
    path.chmod(0o755)
    return env_dir


def read_record(path: Path) -> dict[str, str]:
    """What the stub recorded: lines `NAME value` and `NAME=value`."""
    record: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(("SELF ", "PID ", "START ", "ARGV ", "ENGINE ")):
            key, _, value = line.partition(" ")
            record[key] = value
        elif "=" in line:
            key, _, value = line.partition("=")
            record[key] = value
    return record


def write_config(root: Path, gen_env: Path, asst_env: Path, run_env: Path,
                 gen_port: int, asst_port: int, extra_launch: str = "",
                 name: str = "launch_config.yaml", model: str = "generation") -> Path:
    model_dir = root / "models" / model
    model_dir.mkdir(parents=True, exist_ok=True)

    config = f"""dsl: wrapped

model:
  generation_model_path: {model_dir}
  assistant_model_path: Qwen/Qwen3-VL-32B-Instruct

server:
  generation_base_url: http://127.0.0.1:{gen_port}/v1
  assistant_base_url: http://127.0.0.1:{asst_port}/v1
  generation_served_model_name: generation
  assistant_served_model_name: assistant

launch:
  vllm_env: {gen_env}
  run_env: {run_env}
  runs_root: {root / "runs"}
  server_timeout_sec: 30
  env:
    COMMON_MARKER: for-both
{extra_launch}  servers:
    generation:
      enabled: true
      cuda_visible_devices: "0,1"
      env:
        GENERATION_MARKER: only-generation
      args:
        trust-remote-code: true
        mm-processor-kwargs: '{{"min_pixels": 1, "max_pixels": 2}}'
    assistant:
      enabled: true
      vllm_env: {asst_env}
      cuda_visible_devices: "2,3"
      env:
        ASSISTANT_MARKER: only-assistant
        OMP_NUM_THREADS: "7"

experiment:
  details:
    - test: {root}
  n_workers: 1
  scaffold:
    kind: baseline
"""
    path = root / name
    path.write_text(config, encoding="utf-8")
    return path


def models_probe(port: int, timeout: float = 1.0) -> dict | None:
    """What `/v1/models` answers on this port, or None if it does not answer."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=timeout) as response:
            return json.load(response)
    except Exception:
        return None


def wait_until(predicate, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.2)
    return bool(predicate())


def alive(pid: int) -> bool:
    """Whether the process is alive. `--keep-servers` servers are not our children, only `kill -0`."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def kept_pids(stdout: str) -> list[int]:
    """The PIDs the script named on exit: "... remain up: PID ..."."""
    marker = "stay up (--keep-servers): PID "
    for line in stdout.splitlines():
        if marker in line:
            return [int(word) for word in line.split(marker, 1)[1].split()]
    return []


def kill_pids(pids) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def main() -> int:
    if shutil.which("curl") is None:
        print("no curl - skipping the launch check")
        return 0

    root = Path(tempfile.mkdtemp(prefix="launch_check_"))
    logs = root / "records"
    logs.mkdir()

    gen_env = make_env(root, "gen", "vllm", FAKE_VLLM, logs / "generation.txt")
    asst_env = make_env(root, "asst", "vllm", FAKE_VLLM, logs / "assistant.txt")
    run_env = make_env(root, "run", "python", FAKE_PYTHON, logs / "run.txt")

    gen_port, asst_port = free_port(), free_port()
    config_path = write_config(root, gen_env, asst_env, run_env, gen_port, asst_port)

    print("1. The script starts both servers and reaches the run")
    completed = subprocess.run(
        # `--split` is here not for the run itself but to check that the flag reaches
        # `run_experiment.py` intact: the stub records its argv.
        ["bash", str(RUN_SYSTEM), str(config_path), "--python", PARSER_PYTHON,
         "--split", "control"],
        capture_output=True, text=True, timeout=180,
        env={**os.environ, "PATH": os.environ.get("PATH", "")},
    )
    check("run_system.sh finished", completed.returncode == 0,
          (completed.stdout + completed.stderr)[-800:])
    if completed.returncode != 0:
        return 1
    check("both servers declared ready",
          completed.stdout.count("is ready.") == 2, completed.stdout[-400:])

    generation = read_record(logs / "generation.txt")
    assistant = read_record(logs / "assistant.txt")
    run = read_record(logs / "run.txt")

    print("2. Each server is started from its own environment")
    check("the generator took vllm from its own environment",
          generation["SELF"] == str(gen_env / "bin" / "vllm"), generation["SELF"])
    check("the assistant took vllm from its own environment",
          assistant["SELF"] == str(asst_env / "bin" / "vllm"), assistant["SELF"])
    check("server environments are really different", generation["SELF"] != assistant["SELF"])
    check("the run is started by the interpreter from run_env",
          run["SELF"] == str(run_env / "bin" / "python"), run["SELF"])

    print("3. A server's variables are seen only by it")
    check("the generator variable is set for the generator",
          generation.get("GENERATION_MARKER") == "only-generation", str(generation.get("GENERATION_MARKER")))
    check("the generator variable did NOT leak to the assistant",
          "GENERATION_MARKER" not in assistant, str(assistant.get("GENERATION_MARKER")))
    check("the assistant variable is set for the assistant",
          assistant.get("ASSISTANT_MARKER") == "only-assistant", str(assistant.get("ASSISTANT_MARKER")))
    check("the assistant variable did NOT leak to the generator",
          "ASSISTANT_MARKER" not in generation, str(generation.get("GENERATION_MARKER")))
    check("common variables are visible to both and to the run",
          generation.get("COMMON_MARKER") == assistant.get("COMMON_MARKER") == run.get("COMMON_MARKER") == "for-both",
          f"{generation.get('COMMON_MARKER')}/{assistant.get('COMMON_MARKER')}/{run.get('COMMON_MARKER')}")
    check("a server variable did not leak into the run",
          "GENERATION_MARKER" not in run and "ASSISTANT_MARKER" not in run, str(run.get("GENERATION_MARKER")))
    check("a server without its own value gets the thread default",
          generation.get("OMP_NUM_THREADS") == "1" and generation.get("MKL_NUM_THREADS") == "1",
          f"{generation.get('OMP_NUM_THREADS')}/{generation.get('MKL_NUM_THREADS')}")
    check("the server's own value overrides the default",
          assistant.get("OMP_NUM_THREADS") == "7" and assistant.get("MKL_NUM_THREADS") == "1",
          f"{assistant.get('OMP_NUM_THREADS')}/{assistant.get('MKL_NUM_THREADS')}")
    check("the servers' thread default did not reach the run",
          run.get("OMP_NUM_THREADS") != "1" or os.environ.get("OMP_NUM_THREADS") == "1",
          str(run.get("OMP_NUM_THREADS")))

    print("4. PATH and maps")
    check("the environment bin is first in PATH for the generator",
          generation.get("PATH", "").startswith(f"{gen_env}/bin:"), generation.get("PATH", "")[:120])
    check("the environment bin is first in PATH for the assistant",
          assistant.get("PATH", "").startswith(f"{asst_env}/bin:"), assistant.get("PATH", "")[:120])
    check("maps are assigned per the config",
          (generation.get("CUDA_VISIBLE_DEVICES"), assistant.get("CUDA_VISIBLE_DEVICES")) == ("0,1", "2,3"),
          f"{generation.get('CUDA_VISIBLE_DEVICES')} / {assistant.get('CUDA_VISIBLE_DEVICES')}")

    print("5. Flags arrived intact")
    argv = generation["ARGV"]
    check("port from server.generation_base_url", f"--port {gen_port}" in argv, argv)
    check("a valueless flag is passed as a flag", "--trust-remote-code" in argv, argv)
    check("JSON in a flag survived the shell", '{"min_pixels": 1, "max_pixels": 2}' in argv, argv)
    check("the run got the same config", str(config_path) in run["ARGV"], run["ARGV"])
    check("the split reached the run", "--split control" in run["ARGV"], run["ARGV"])

    # A bad selection must be rejected BEFORE the servers start: there it would cost minutes
    # of weight loading, and a typo should be learned at once.
    bogus = subprocess.run(
        ["bash", str(RUN_SYSTEM), str(config_path), "--python", PARSER_PYTHON,
         "--split", "no-such"],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "PATH": os.environ.get("PATH", "")},
    )
    check("a bad split is rejected", bogus.returncode == 2, str(bogus.returncode))
    check("it says what exactly is bad", "no-such" in bogus.stderr, bogus.stderr[-200:])
    check("no servers were started meanwhile", "is ready." not in bogus.stdout, bogus.stdout[-300:])

    print("6. A typo in the launch section is an error, not silence")
    typo_config = write_config(root, gen_env, asst_env, run_env, free_port(), free_port(),
                               extra_launch="  vllm_envs: /mistyped/key\n")
    plan = subprocess.run(
        [PARSER_PYTHON, str(REPO_ROOT / "agent" / "tools" / "launch_plan.py"), "--config", str(typo_config)],
        capture_output=True, text=True,
    )
    check("an unknown launch key is rejected", plan.returncode != 0, plan.stdout[-200:])
    check("it says which key is extra", "vllm_envs" in plan.stderr, plan.stderr[-200:])

    # `env` with a string instead of a mapping is the likeliest confusion between
    # "environment variables" and "path to an environment".
    swapped = write_config(root, gen_env, asst_env, run_env, free_port(), free_port())
    swapped_text = swapped.read_text(encoding="utf-8").replace(
        "      env:\n        GENERATION_MARKER: only-generation\n",
        f"      env: {gen_env}\n",
    )
    swapped.write_text(swapped_text, encoding="utf-8")
    plan = subprocess.run(
        [PARSER_PYTHON, str(REPO_ROOT / "agent" / "tools" / "launch_plan.py"), "--config", str(swapped)],
        capture_output=True, text=True,
    )
    check("a path instead of variables is rejected", plan.returncode != 0, plan.stdout[-200:])
    check("the correct key name is suggested", "vllm_env" in plan.stderr, plan.stderr[-200:])

    print("7. Servers start in parallel, not one after another")
    # The stubs occupy the port not at once but after DELAY seconds. If the script waits for
    # servers one by one, the second does not even start until the first is ready, and the gap
    # between their starts is at least DELAY. With a parallel start the gap is fractions of a
    # second.
    delay = 5
    parallel_config = write_config(root, gen_env, asst_env, run_env, free_port(), free_port())
    for record in (logs / "generation.txt", logs / "assistant.txt"):
        record.unlink(missing_ok=True)

    started = time.monotonic()
    completed = subprocess.run(
        ["bash", str(RUN_SYSTEM), str(parallel_config), "--python", PARSER_PYTHON],
        capture_output=True, text=True, timeout=180,
        env={**os.environ, "PATH": os.environ.get("PATH", ""), "FAKE_VLLM_DELAY": str(delay)},
    )
    elapsed = time.monotonic() - started
    check("run_system.sh finished with slow servers", completed.returncode == 0,
          (completed.stdout + completed.stderr)[-800:])

    if completed.returncode == 0:
        gen_start = float(read_record(logs / "generation.txt")["START"])
        asst_start = float(read_record(logs / "assistant.txt")["START"])
        gap = abs(asst_start - gen_start)
        check("the assistant started without waiting for the generator",
              gap < delay - 1, f"start gap {gap:.1f} s with delay {delay} s")
        # The wall measurement is quantized by the poll step of `run_system.sh`: readiness is
        # noticed no earlier than the next poll. Hence the bound: in parallel it is delay + poll,
        # sequentially 2*delay + two polls. The step there grows, so with the ceiling the bound
        # is only roomier: "one delay, not two" is checked, and the margin is enough for it.
        budget = delay + 2 * POLL_INTERVAL_SEC
        check("startup cost one delay, not two",
              elapsed < budget, f"{elapsed:.1f} s with threshold {budget} s (delay {delay} s)")

    print("8. --keep-servers: completes the picture and does not stop what works")
    # Next, servers that the script deliberately does not kill are started, and this check
    # kills them. So the whole section lives in try/finally: a failure in the middle must not
    # leave processes on the machine.
    kept: list[int] = []
    stranger: subprocess.Popen | None = None
    try:
        # Antonym flags. Checked before the config is parsed, so any config will do.
        clash = subprocess.run(
            ["bash", str(RUN_SYSTEM), str(config_path), "--keep-servers", "--no-servers"],
            capture_output=True, text=True, timeout=60,
        )
        check("--keep-servers together with --no-servers is rejected", clash.returncode == 2,
              f"code {clash.returncode}: {(clash.stdout + clash.stderr)[-300:]}")
        check("the flag conflict is explained",
              "--keep-servers" in clash.stderr and "--no-servers" in clash.stderr,
              clash.stderr[-300:])

        keep_gen, keep_asst = free_port(), free_port()
        keep_config = write_config(root, gen_env, asst_env, run_env, keep_gen, keep_asst,
                                   name="keep_config.yaml")
        keep_argv = ["bash", str(RUN_SYSTEM), str(keep_config), "--python", PARSER_PYTHON]
        # The registry of running servers is moved to a temporary root: the real one lives in the
        # user's `/tmp`, and the check has no right either to show its entries or to remove them.
        servers_dir = root / "servers"
        clean_env = {**os.environ, "PATH": os.environ.get("PATH", ""),
                     "CAD_AGENT_SERVERS_DIR": str(servers_dir)}
        tool_argv = ["bash", str(SERVERS_TOOL)]
        for record in (logs / "generation.txt", logs / "assistant.txt"):
            record.unlink(missing_ok=True)

        first = subprocess.run(keep_argv + ["--keep-servers"], capture_output=True, text=True,
                               timeout=180, env=clean_env)
        check("the run with --keep-servers finished", first.returncode == 0,
              (first.stdout + first.stderr)[-800:])
        kept = kept_pids(first.stdout)
        check("the script announced that servers stay",
              "stay up" in first.stdout, first.stdout[-400:])
        check("PIDs are named for a manual stop", len(kept) == 2, first.stdout[-400:])
        # `setsid` is checked too: it execs in place, so the printed `$!` must be the PID of the
        # server itself, not of a vanished intermediary. If setsid forked, these PIDs would
        # already be dead and `wait_for_servers` would declare the server dead inside the script.
        check("servers are alive after the script exits",
              bool(kept) and all(alive(pid) for pid in kept), str(kept))
        check("both ports respond after the script exits",
              models_probe(keep_gen) is not None and models_probe(keep_asst) is not None,
              f"{models_probe(keep_gen)} / {models_probe(keep_asst)}")

        # A plain `ps` will not show the servers left behind: they are in their own session
        # without a terminal, and such a `ps` selects processes by the caller's terminal. So the
        # script must name a command that will show them and leave a record that outlives the
        # shell itself, otherwise the mode looks as if it started nothing.
        check("a ps command showing the kept servers is printed",
              any("ps -o pid" in line and all(str(pid) in line for pid in kept)
                  for line in first.stdout.splitlines()), first.stdout[-600:])
        check("the running-servers registry has two entries",
              len(sorted(servers_dir.glob("*.env"))) == 2,
              str(sorted(f.name for f in servers_dir.glob("*.env")) if servers_dir.exists()
                  else f"no directory {servers_dir}"))

        listing = subprocess.run(tool_argv, capture_output=True, text=True, timeout=60,
                                 env=clean_env)
        check("the registry shows both servers ready",
              listing.stdout.count("ready") == 2, listing.stdout[-600:])
        check("the catalog named PIDs, roles and addresses",
              all(str(pid) in listing.stdout for pid in kept)
              and "generation" in listing.stdout and "assistant" in listing.stdout
              and str(keep_gen) in listing.stdout and str(keep_asst) in listing.stdout,
              listing.stdout[-600:])
        catalog_ps = subprocess.run(tool_argv + ["ps"], capture_output=True, text=True,
                                    timeout=60, env=clean_env)
        check("the registry shows servers as ps lines",
              all(str(pid) in catalog_ps.stdout for pid in kept), catalog_ps.stdout[-600:])

        # A second run on the same config: what the mode exists for.
        before = {role: read_record(logs / f"{role}.txt") for role in ("generation", "assistant")}
        second = subprocess.run(keep_argv + ["--keep-servers"], capture_output=True, text=True,
                                timeout=180, env=clean_env)
        check("the repeated run finished", second.returncode == 0,
              (second.stdout + second.stderr)[-800:])
        check("both servers reused", second.stdout.count("reusing it") == 2,
              second.stdout[-500:])
        check("nothing was started again", "Launching" not in second.stdout,
              second.stdout[-500:])
        # A run that reused everything did not start its own servers, and used to say nothing
        # about them at all, though the servers stand.
        check("on reuse it says where to look at what is up",
              "tools/servers.sh" in second.stdout, second.stdout[-500:])
        after = {role: read_record(logs / f"{role}.txt") for role in before}
        check("server processes are the same, not new",
              all(before[role]["PID"] == after[role]["PID"]
                  and before[role]["START"] == after[role]["START"] for role in before),
              f"was {[before[r]['PID'] for r in before]}, now {[after[r]['PID'] for r in after]}")

        # The same `--served-model-name` on the port but another checkpoint: exactly the silent
        # substitution because of which matching is by `root`, not by `id`.
        wrong_config = write_config(root, gen_env, asst_env, run_env, keep_gen, keep_asst,
                                    name="wrong_config.yaml", model="other")
        wrong = subprocess.run(["bash", str(RUN_SYSTEM), str(wrong_config), "--python",
                                PARSER_PYTHON, "--keep-servers"],
                               capture_output=True, text=True, timeout=180, env=clean_env)
        check("a foreign checkpoint on the port is rejected", wrong.returncode != 0,
              (wrong.stdout + wrong.stderr)[-400:])
        check("it says the wrong server is up", "the wrong server" in wrong.stderr,
              wrong.stderr[-400:])
        check("both paths are named - expected and actual",
              str(root / "models" / "other") in wrong.stderr
              and str(root / "models" / "generation") in wrong.stderr, wrong.stderr[-400:])

        # It answers but does not report `root`: the promise "the same checkpoint" is backed by
        # nothing, so it cannot be reused.
        mute_port = free_port()
        stranger = subprocess.Popen([PARSER_PYTHON, "-c", STRANGER_SERVER, str(mute_port)],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        check("the foreign server came up",
              wait_until(lambda: models_probe(mute_port) is not None, 20))
        mute_config = write_config(root, gen_env, asst_env, run_env, mute_port, free_port(),
                                   name="mute_config.yaml")
        mute = subprocess.run(["bash", str(RUN_SYSTEM), str(mute_config), "--python",
                               PARSER_PYTHON, "--keep-servers"],
                              capture_output=True, text=True, timeout=180, env=clean_env)
        check("a server without a weights path is rejected", mute.returncode != 0,
              (mute.stdout + mute.stderr)[-400:])
        check("it says why it cannot be reused",
              "does not report the weights path" in mute.stderr, mute.stderr[-400:])

        # Normal mode on busy ports: the script does not attach to a foreign server but starts
        # its own, and that one dies on the busy port.
        #
        # The return code is deliberately not checked here. Readiness is confirmed by a reply
        # from the port, and a foreign server answers there, so the death of our own may go
        # unnoticed. This is not a property of `--keep-servers` but an old gap in
        # `wait_for_servers`; what is checked here is exactly what is promised: there is no
        # reuse without the flag.
        before_plain = read_record(logs / "generation.txt")
        plain = subprocess.run(keep_argv, capture_output=True, text=True, timeout=180,
                               env=clean_env)
        check("without the flag a foreign server is not reused",
              "reusing it" not in plain.stdout, plain.stdout[-400:])
        check("without the flag the script starts its own server",
              read_record(logs / "generation.txt")["PID"] != before_plain["PID"],
              f'{before_plain["PID"]} -> {read_record(logs / "generation.txt")["PID"]}')

        # Terminal hangup: SIGHUP goes to the whole process group of the script, and the servers
        # in their own session must survive it. That is what `setsid` in the mode is for. The
        # signal is sent in the run phase: the servers are already ready, and their safety is
        # what is promised.
        #
        # SIGHUP is checked, not SIGINT: bash sets SIGINT to SIG_IGN for the async children of a
        # non-interactive shell, so Ctrl-C does not threaten the servers even without `setsid`;
        # on it the check would be green with `setsid` removed (measured). SIGHUP and SIGTERM
        # reach the group, and the difference shows: without `setsid` the server dies, with it
        # it lives.
        ctrl_gen, ctrl_asst = free_port(), free_port()
        ctrl_config = write_config(root, gen_env, asst_env, run_env, ctrl_gen, ctrl_asst,
                                   name="ctrl_config.yaml")
        run_record = logs / "run.txt"
        run_record.unlink(missing_ok=True)
        interrupted = subprocess.Popen(
            ["bash", str(RUN_SYSTEM), str(ctrl_config), "--python", PARSER_PYTHON,
             "--keep-servers"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            start_new_session=True, env={**clean_env, "FAKE_RUN_DELAY": "10"},
        )
        reached_run = wait_until(run_record.exists, 120)
        check("the script reached the run phase", reached_run)
        os.killpg(os.getpgid(interrupted.pid), signal.SIGHUP)
        ctrl_stdout, _ = interrupted.communicate(timeout=120)
        kept += kept_pids(ctrl_stdout)
        check("the run was interrupted, not completed", interrupted.returncode != 0,
              str(interrupted.returncode))
        check("servers are announced as kept after an interrupt",
              "stay up" in ctrl_stdout, ctrl_stdout[-500:])
        check("servers survived SIGHUP to the process group",
              models_probe(ctrl_gen) is not None and models_probe(ctrl_asst) is not None,
              ctrl_stdout[-500:])

        # A server that did not live to readiness is killed in this mode too: there is no
        # reason to leave it, it cannot be reused (the probe will not answer), and it holds GPUs.
        slow_gen, slow_asst = free_port(), free_port()
        slow_config = write_config(root, gen_env, asst_env, run_env, slow_gen, slow_asst,
                                   name="slow_config.yaml")
        slow_config.write_text(
            slow_config.read_text(encoding="utf-8").replace("server_timeout_sec: 30",
                                                            "server_timeout_sec: 3"),
            encoding="utf-8")
        for record in (logs / "generation.txt", logs / "assistant.txt"):
            record.unlink(missing_ok=True)
        slow = subprocess.run(
            ["bash", str(RUN_SYSTEM), str(slow_config), "--python", PARSER_PYTHON,
             "--keep-servers"],
            capture_output=True, text=True, timeout=180,
            env={**clean_env, "FAKE_VLLM_DELAY": "12"},
        )
        check("a server that did not come up is an error", slow.returncode != 0,
              (slow.stdout + slow.stderr)[-400:])
        check("servers that did not come up are not announced as kept",
              "stay up" not in slow.stdout, slow.stdout[-400:])
        check("it says the servers are being stopped", "Stopping vLLM servers" in slow.stdout,
              slow.stdout[-400:])
        slow_pids = [int(read_record(logs / f"{role}.txt")["PID"])
                     for role in ("generation", "assistant")
                     if (logs / f"{role}.txt").exists()]
        # They go on the cleanup list before the check, not after: if the readiness guard is
        # broken, nobody else will kill them, and a failed check must not leave processes on the
        # machine.
        kept += slow_pids
        check("servers that never became ready are really stopped",
              bool(slow_pids) and wait_until(lambda: not any(alive(pid) for pid in slow_pids), 20),
              str(slow_pids))
        # A frontend hanging at startup is exactly the case where the engine survived a signal
        # to one PID: the whole group is killed, and the engine with it.
        slow_engines = [int(read_record(logs / f"{role}.txt")["ENGINE"])
                        for role in ("generation", "assistant")
                        if (logs / f"{role}.txt").exists()]
        check("their engines are stopped with them",
              bool(slow_engines)
              and wait_until(lambda: not any(alive(pid) for pid in slow_engines), 20),
              str(slow_engines))
        # A record of a killed server is a lie about busy GPUs. The registry holds only those
        # that lived to readiness: those are the ones promised not to be killed.
        check("servers that did not come up are not in the registry",
              not any(str(slow_gen) in f.name or str(slow_asst) in f.name
                      for f in servers_dir.glob("*.env")),
              str(sorted(f.name for f in servers_dir.glob("*.env"))))

        # Series cleanup: the registry must not only show but also kill, which is why it
        # outlives the shell in which the PIDs were printed.
        standing = [pid for pid in kept if alive(pid)]
        stopped = subprocess.run(tool_argv + ["stop", "--all"], capture_output=True, text=True,
                                 timeout=90, env=clean_env)
        check("the catalog stopped everything that was left up",
              bool(standing) and stopped.returncode == 0
              and wait_until(lambda: not any(alive(pid) for pid in standing), 30),
              f"{standing}: {(stopped.stdout + stopped.stderr)[-400:]}")
        check("entries of stopped servers are removed",
              not sorted(servers_dir.glob("*.env")),
              str(sorted(f.name for f in servers_dir.glob("*.env"))))
    finally:
        kill_pids(kept)
        if stranger is not None:
            stranger.kill()
            stranger.wait(timeout=10)

    print("9. Run directory name and the suffix of a repeated launch")
    runs_root = root / "runs"
    named_config = write_config(root, gen_env, asst_env, run_env, free_port(), free_port(),
                                name="named_config.yaml", extra_launch="  run_name: my_run\n")
    named_argv = ["bash", str(RUN_SYSTEM), str(named_config), "--python", PARSER_PYTHON]
    clean = {**os.environ, "PATH": os.environ.get("PATH", "")}

    first_named = subprocess.run(named_argv, capture_output=True, text=True, timeout=180, env=clean)
    check("the run with a name from the config finished", first_named.returncode == 0,
          (first_named.stdout + first_named.stderr)[-800:])
    check("the directory is named from the config", (runs_root / "my_run").is_dir(),
          str(sorted(p.name for p in runs_root.iterdir())))
    check("the directory name is printed", str(runs_root / "my_run") in first_named.stdout,
          first_named.stdout[-300:])

    # The previous directory must stay untouched: a run takes hours, and silently writing
    # over its results is the worst thing a repeat of the same command can end with.
    marker = runs_root / "my_run" / "keep-me.txt"
    marker.write_text("previous run", encoding="utf-8")

    second_named = subprocess.run(named_argv, capture_output=True, text=True, timeout=180, env=clean)
    check("a repeated run with the same name finished", second_named.returncode == 0,
          (second_named.stdout + second_named.stderr)[-800:])
    check("the repeated launch took a suffix", (runs_root / "my_run_2").is_dir(),
          str(sorted(p.name for p in runs_root.iterdir())))
    check("the previous directory is untouched",
          marker.exists() and marker.read_text(encoding="utf-8") == "previous run",
          str(sorted(p.name for p in (runs_root / "my_run").iterdir())))
    check("the repeated run wrote to a new directory",
          str(runs_root / "my_run_2") in second_named.stdout, second_named.stdout[-300:])

    named_cli = subprocess.run(named_argv + ["--name", "from-flag"], capture_output=True,
                               text=True, timeout=180, env=clean)
    check("the run with a name from a flag finished", named_cli.returncode == 0,
          (named_cli.stdout + named_cli.stderr)[-800:])
    check("the --name flag overrode the config", (runs_root / "from-flag").is_dir(),
          str(sorted(p.name for p in runs_root.iterdir())))

    # A path separator in the name would take the run into a foreign directory. Rejected
    # before the servers start, by the same config parsing as the other errors.
    bad_name = subprocess.run(named_argv + ["--name", "../outside"], capture_output=True,
                              text=True, timeout=60, env=clean)
    check("a bad name is rejected", bad_name.returncode != 0, str(bad_name.returncode))
    check("the invalid name is named", "outside" in bad_name.stderr, bad_name.stderr[-300:])
    check("no servers started with an invalid name", "is ready." not in bad_name.stdout,
          bad_name.stdout[-300:])
    check("no directory created outside runs_root", not (root / "outside").exists())

    print("10. servers.sh up: servers without a run")
    # Servers that the script deliberately does not kill are started again here, and this
    # check kills them. Hence try/finally, as in section 8.
    up_kept: list[int] = []
    try:
        up_gen, up_asst = free_port(), free_port()
        up_config = write_config(root, gen_env, asst_env, run_env, up_gen, up_asst,
                                 name="up_config.yaml", extra_launch="  run_name: up_run\n")
        up_servers_dir = root / "servers_up"
        up_env = {**os.environ, "PATH": os.environ.get("PATH", ""),
                  "CAD_AGENT_SERVERS_DIR": str(up_servers_dir)}
        # A config without the `--python` flag: `servers.sh up` passes the argument tail on to
        # `run_system.sh`, and this is checked too: without the pass-through the config parsing
        # would go to the wrong interpreter and fail on PyYAML.
        up_argv = ["bash", str(SERVERS_TOOL), "up", str(up_config), "--python", PARSER_PYTHON]

        no_config = subprocess.run(["bash", str(SERVERS_TOOL), "up"],
                                   capture_output=True, text=True, timeout=60, env=up_env)
        check("up without a config is rejected", no_config.returncode == 2, str(no_config.returncode))
        # A default config here would be a bet on which checkpoint takes the GPUs; this must be
        # said outright rather than starting whatever comes.
        check("it says a config is needed", "give a config" in no_config.stderr,
              no_config.stderr[-300:])
        flag_first = subprocess.run(["bash", str(SERVERS_TOOL), "up", "--keep-servers"],
                                    capture_output=True, text=True, timeout=60, env=up_env)
        check("a flag instead of a config is rejected", flag_first.returncode == 2,
              f"code {flag_first.returncode}: {(flag_first.stdout + flag_first.stderr)[-300:]}")

        run_record = logs / "run.txt"
        run_record.unlink(missing_ok=True)
        for record in (logs / "generation.txt", logs / "assistant.txt"):
            record.unlink(missing_ok=True)

        up = subprocess.run(up_argv, capture_output=True, text=True, timeout=180, env=up_env)
        check("servers.sh up finished", up.returncode == 0,
              (up.stdout + up.stderr)[-800:])
        up_kept = kept_pids(up.stdout)
        check("both servers are up", up.stdout.count("is ready.") == 2, up.stdout[-500:])
        # The main promise of the mode: there is no run. The run stub writes its file first
        # thing, so its absence means exactly "was not launched".
        check("the run was not started", not run_record.exists(), up.stdout[-500:])
        check("it says there will be no run", "the run is not started" in up.stdout,
              up.stdout[-500:])
        check("servers are alive after the script exits",
              len(up_kept) == 2 and all(alive(pid) for pid in up_kept), str(up_kept))
        check("both ports respond",
              models_probe(up_gen) is not None and models_probe(up_asst) is not None,
              f"{models_probe(up_gen)} / {models_probe(up_asst)}")

        # There must be no run directory: there was no run. An empty directory named like a run
        # next to real ones reads as a run that lost its results.
        check("no run directory created", not (runs_root / "up_run").exists(),
              str(sorted(p.name for p in runs_root.iterdir())))
        server_logs = sorted((runs_root / "servers").glob("up_run*/logs/*_vllm.log"))
        check("server logs went to runs_root/servers", len(server_logs) == 2,
              str([str(p) for p in server_logs]))

        up_records = sorted(up_servers_dir.glob("*.env"))
        check("the server registry has two entries", len(up_records) == 2,
              str([f.name for f in up_records]))
        # The RUN field is the run that the servers belong to. There is none here, and putting
        # the logs directory there would print "run: ..." about a directory in which there was
        # no run.
        run_fields = [line for f in up_records
                      for line in f.read_text(encoding="utf-8").splitlines()
                      if line.startswith("RUN=")]
        check("the RUN field in the records is empty", run_fields == ["RUN=", "RUN="], str(run_fields))
        listing = subprocess.run(["bash", str(SERVERS_TOOL)], capture_output=True, text=True,
                                 timeout=60, env=up_env)
        check("the registry shows both servers ready",
              listing.stdout.count("ready") == 2, listing.stdout[-600:])
        check("the catalog is silent about a nonexistent run",
              "  run:" not in listing.stdout, listing.stdout[-600:])

        # A repeat `up` is the same as a repeat run with --keep-servers: what is up is reused
        # rather than started a second time on a busy port.
        before_up = {role: read_record(logs / f"{role}.txt") for role in ("generation", "assistant")}
        again = subprocess.run(up_argv, capture_output=True, text=True, timeout=180, env=up_env)
        check("the repeated up finished", again.returncode == 0,
              (again.stdout + again.stderr)[-800:])
        check("both servers reused", again.stdout.count("reusing it") == 2,
              again.stdout[-500:])
        after_up = {role: read_record(logs / f"{role}.txt") for role in before_up}
        check("server processes are the same",
              all(before_up[role]["PID"] == after_up[role]["PID"] for role in before_up),
              f"was {[before_up[r]['PID'] for r in before_up]}, "
              f"now {[after_up[r]['PID'] for r in after_up]}")
        check("the repeated up did not start a run either", not run_record.exists(),
              again.stdout[-500:])

        standing = [pid for pid in up_kept if alive(pid)]
        up_engines = [int(read_record(logs / f"{role}.txt")["ENGINE"])
                      for role in ("generation", "assistant")]
        engines_were_alive = all(alive(pid) for pid in up_engines)
        stopped = subprocess.run(["bash", str(SERVERS_TOOL), "stop", "--all"],
                                 capture_output=True, text=True, timeout=90, env=up_env)
        check("what up started is stopped by the same catalog",
              bool(standing) and stopped.returncode == 0
              and wait_until(lambda: not any(alive(pid) for pid in standing), 30),
              f"{standing}: {(stopped.stdout + stopped.stderr)[-400:]}")
        # The engine lives a minute after the frontend: it can die within that time only from a
        # signal to the group, not from the death of the parent.
        check("stop kills the engine too, not just the frontend",
              engines_were_alive
              and wait_until(lambda: not any(alive(pid) for pid in up_engines), 30),
              f"{up_engines}: {(stopped.stdout + stopped.stderr)[-400:]}")
    finally:
        kill_pids(up_kept)

    print("11. --only: roles start separately")
    # Why the section exists: the roles start simultaneously, and readiness is awaited by a
    # single loop, so one that misses its deadline takes an already started neighbor with it
    # (`server_failed` exits the script, `SERVERS_READY` is still zero, cleanup kills
    # everything). Separately each role has its own deadline and its own fate.
    only_kept: list[int] = []
    try:
        only_gen, only_asst = free_port(), free_port()
        only_config = write_config(root, gen_env, asst_env, run_env, only_gen, only_asst,
                                   name="only_config.yaml",
                                   extra_launch="  run_name: only_run\n")
        only_servers_dir = root / "servers_only"
        only_env = {**os.environ, "PATH": os.environ.get("PATH", ""),
                    "CAD_AGENT_SERVERS_DIR": str(only_servers_dir)}

        # Refusals come before startup: a silently accepted flag would mean the person went off
        # to wait for a server that nobody started.
        with_run = subprocess.run(
            ["bash", str(RUN_SYSTEM), str(only_config), "--python", PARSER_PYTHON,
             "--only", "generation"],
            capture_output=True, text=True, timeout=60, env=only_env)
        check("--only without --servers-only is rejected", with_run.returncode == 2,
              f"code {with_run.returncode}: {(with_run.stdout + with_run.stderr)[-300:]}")
        check("it says what is missing", "--servers-only" in with_run.stderr,
              with_run.stderr[-300:])
        bad_role = subprocess.run(
            ["bash", str(RUN_SYSTEM), str(only_config), "--python", PARSER_PYTHON,
             "--servers-only", "--only", "both"],
            capture_output=True, text=True, timeout=60, env=only_env)
        check("an unknown role is rejected", bad_role.returncode == 2,
              f"code {bad_role.returncode}: {(bad_role.stdout + bad_role.stderr)[-300:]}")
        check("the valid roles are named", "assistant" in bad_role.stderr,
              bad_role.stderr[-300:])

        for record in (logs / "generation.txt", logs / "assistant.txt", logs / "run.txt"):
            record.unlink(missing_ok=True)

        only_argv = ["bash", str(SERVERS_TOOL), "up", str(only_config),
                     "--python", PARSER_PYTHON, "--only"]
        first = subprocess.run(only_argv + ["generation"], capture_output=True, text=True,
                               timeout=180, env=only_env)
        check("starting one role finished", first.returncode == 0,
              (first.stdout + first.stderr)[-800:])
        only_kept += kept_pids(first.stdout)
        check("exactly one server declared ready", first.stdout.count("is ready.") == 1,
              first.stdout[-500:])
        # The main point: the second role was not launched at all. The stub writes its file first
        # thing, so its absence means exactly "was not called".
        check("the assistant was not started", not (logs / "assistant.txt").exists(),
              first.stdout[-500:])
        check("it says the role was skipped", "assistant skipped" in first.stdout,
              first.stdout[-500:])
        check("the assistant port is silent", models_probe(only_asst) is None)
        check("the server registry has one entry",
              len(sorted(only_servers_dir.glob("*.env"))) == 1,
              str([f.name for f in sorted(only_servers_dir.glob("*.env"))]))

        started = read_record(logs / "generation.txt")
        second = subprocess.run(only_argv + ["assistant"], capture_output=True, text=True,
                                timeout=180, env=only_env)
        check("starting the second role finished", second.returncode == 0,
              (second.stdout + second.stderr)[-800:])
        only_kept += kept_pids(second.stdout)
        check("this time the generator is skipped", "generation skipped" in second.stdout,
              second.stdout[-500:])
        # Skipped means untouched: neither started again nor killed. Checked by the stub's PID,
        # not by the port answering: the second one could answer too.
        check("the generator process is the same",
              read_record(logs / "generation.txt")["PID"] == started["PID"],
              f"was {started['PID']}, now {read_record(logs / 'generation.txt')['PID']}")
        check("now both ports respond",
              models_probe(only_gen) is not None and models_probe(only_asst) is not None,
              f"{models_probe(only_gen)} / {models_probe(only_asst)}")
        check("the server registry has two entries",
              len(sorted(only_servers_dir.glob("*.env"))) == 2,
              str([f.name for f in sorted(only_servers_dir.glob("*.env"))]))
        check("there was still no run", not (logs / "run.txt").exists(),
              second.stdout[-500:])

        standing = [pid for pid in only_kept if alive(pid)]
        stopped = subprocess.run(["bash", str(SERVERS_TOOL), "stop", "--all"],
                                 capture_output=True, text=True, timeout=90, env=only_env)
        check("what started separately is stopped by one stop --all",
              len(standing) == 2 and stopped.returncode == 0
              and wait_until(lambda: not any(alive(pid) for pid in standing), 30),
              f"{standing}: {(stopped.stdout + stopped.stderr)[-400:]}")
    finally:
        kill_pids(only_kept)

    print("12. --server-timeout: readiness deadline for one launch")
    timeout_config = write_config(root, gen_env, asst_env, run_env, free_port(), free_port(),
                                  name="timeout_config.yaml",
                                  extra_launch="  run_name: timeout_run\n")
    timeout_env = {**os.environ, "PATH": os.environ.get("PATH", ""),
                   "CAD_AGENT_SERVERS_DIR": str(root / "servers_timeout"),
                   # More than the overridden deadline and less than the config one (30 s):
                   # without the override this startup would succeed.
                   "FAKE_VLLM_DELAY": "10"}
    bad_value = subprocess.run(
        ["bash", str(RUN_SYSTEM), str(timeout_config), "--python", PARSER_PYTHON,
         "--servers-only", "--server-timeout", "0"],
        capture_output=True, text=True, timeout=60, env=timeout_env)
    check("a zero deadline is rejected", bad_value.returncode == 2,
          f"code {bad_value.returncode}: {(bad_value.stdout + bad_value.stderr)[-300:]}")

    short = subprocess.run(
        ["bash", str(RUN_SYSTEM), str(timeout_config), "--python", PARSER_PYTHON,
         "--servers-only", "--only", "generation", "--server-timeout", "2"],
        capture_output=True, text=True, timeout=180, env=timeout_env)
    check("with a short deadline the startup failed", short.returncode != 0,
          (short.stdout + short.stderr)[-400:])
    # The number in the refusal is the one that was waited for. Were the config one left
    # there, the person would fix a config this launch did not read.
    check("the failure states the overridden deadline", "did not come up within 2 s" in short.stderr,
          short.stderr[-400:])
    check("it says the deadline is overridden", "Server readiness deadline: 2 s" in short.stdout,
          short.stdout[-400:])

    shutil.rmtree(root, ignore_errors=True)

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {', '.join(FAILURES)}")
        return 1
    print("Launching from different environments is fine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
