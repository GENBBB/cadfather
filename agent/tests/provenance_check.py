#!/usr/bin/env python3
"""Check of run provenance (`harness/provenance.py`).

The run-to-code link must be recorded in the run itself: `config.json` carries
neither a commit nor a digest, and a digest computed after the fact from disk would
belong to the wrong run if code was edited while the run was in progress.

The code comes from a temporary repository rather than the real one: the check has
to edit files in the middle of a "run" and make commits.

Checks:

1. the digest in `provenance` is exactly `code_digest.code_digest`, over the same files;
2. the server is identified: `root` from `/v1/models` and the process start time
   from `/metrics`; no server gives empty, not an exception;
3. the text of the starting YAML is saved even if the file is edited later;
4. a code edit during the run appears in `code_end.changed`;
5. the report header prints the digest;
6. the real `run_experiment` (on `harness_e2e` stubs) writes `provenance.json`
   with the run end, leaves `config.json` alone, and prints the digest in
   `report.txt`.
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = AGENT_ROOT.parent
sys.path.insert(0, str(AGENT_ROOT))
sys.path.insert(0, str(AGENT_ROOT / "tools"))
sys.path.insert(0, str(AGENT_ROOT / "tests"))

from cad_agent.harness import provenance  # noqa: E402
from cad_agent.harness import report as report_mod  # noqa: E402

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail and not ok else ""))
    if not ok:
        FAILED.append(name)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D401 — keep the check output quiet
        pass

    def do_GET(self):
        if self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "assistant", "root": "/weights/qwen-fp8", "max_model_len": 32768}]})
        elif self.path == "/metrics":
            body = ("# HELP process_start_time_seconds x\n"
                    "process_start_time_seconds 1758790000.5\n"
                    "vllm:prompt_tokens_total 10\n")
        else:
            self.send_response(404)
            self.end_headers()
            return
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def make_repo(root: Path) -> Path:
    """Mini repository with the real `code_digest.py` and one file in each key tree."""
    repo = root / "repo"
    for tree in ("harness", "scaffold", "capabilities"):
        (repo / "agent/cad_agent" / tree).mkdir(parents=True)
        (repo / "agent/cad_agent" / tree / "mod.py").write_text(f"X = '{tree}'\n")
    shutil.copy(REPO_ROOT / "agent/cad_agent/harness/code_digest.py",
                repo / "agent/cad_agent/harness/code_digest.py")
    (repo / "agent/run_experiment.py").write_text("pass\n")
    git(repo.parent, "init", "-q", str(repo))
    git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "one")
    return repo


def main() -> int:
    port = free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    tmp = Path(tempfile.mkdtemp(prefix="provenance_check_"))
    try:
        repo = make_repo(tmp)
        key = provenance._code_digest_module(repo)

        print("1. digest")
        state = provenance.code_state(repo)
        check("digest matches code_digest", state.get("digest") == key.code_digest(repo), str(state))
        check("all key files present", set(state["files"]) >= {"agent/cad_agent/harness/mod.py", "agent/run_experiment.py"})
        check("missing key file is marked absent", state["files"].get("agent/cad_agent/dsl_runtime.py") == "absent")

        print("2. server")
        config = {"server": {"assistant_base_url": f"http://127.0.0.1:{port}/v1",
                             "generation_base_url": f"http://127.0.0.1:{free_port()}/v1"}}
        yaml_path = tmp / "run.yaml"
        yaml_path.write_text("experiment: {n_workers: 48}\n")
        start = provenance.collect_start(config, yaml_path, repo=repo)
        ident = start["servers"].get("assistant") or {}
        check("weights root from /v1/models", (ident.get("models") or [{}])[0].get("root") == "/weights/qwen-fp8", str(ident))
        check("process start time", ident.get("process_start_time") == [1758790000.5], str(ident))
        check("a dead server gives empty", start["servers"].get("generation") == {}, str(start["servers"]))
        check("commit recorded", (start.get("git") or {}).get("commit") == git(repo, "rev-parse", "HEAD").strip())

        print("3. start YAML")
        yaml_path.write_text("experiment: {n_workers: 32}\n")
        check("text is the start one", "48" in (start.get("source_config") or {}).get("text", ""))

        print("4. edit during the run")
        (repo / "agent/cad_agent/harness/mod.py").write_text("X = 'changed'\n")
        end = provenance.collect_end(start, config)
        check("changed file is named", end["code_end"]["changed"] == ["agent/cad_agent/harness/mod.py"], str(end["code_end"]))
        check("end digest differs", end["code_end"]["digest"] != start["code"]["digest"])
        check("the server at the end is the same", end["servers_end"] == start["servers"])

        print("5. report header")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            report_mod._print_provenance(end)
        text = out.getvalue()
        check("digest in the header", start["code"]["digest"] in text, text)
        check("code change in the header", "changed during the run" in text, text)

        print("6. real run on stubs")
        import harness_e2e
        from cad_agent.harness import run_eval

        harness_e2e.install_fakes()
        folder = harness_e2e.make_dataset(tmp, n=2)
        run_config = harness_e2e.base_config(folder, backend="serial_fork", n_workers=1)
        result = run_eval.run_experiment(config=run_config, run_dir=tmp / "run_e2e", source_config=yaml_path)
        run_dir = Path(result["run_dir"])
        prov = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
        real_digest = provenance.code_state(provenance.REPO_ROOT)["digest"]
        check("provenance.json completed at the end", bool(prov.get("finished_at")), str(sorted(prov)))
        check("digest is of the real tree", prov["code"]["digest"] == real_digest)
        check("config.json has no run circumstances",
              not {"started_at", "host", "pid"} & set(json.loads((run_dir / "config.json").read_text())))
        report_text = (run_dir / "logs" / "report.txt").read_text(encoding="utf-8")
        check("digest in report.txt", real_digest in report_text)
    finally:
        server.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nResult:", "all ok" if not FAILED else f"FAILED {len(FAILED)}: {', '.join(FAILED)}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
