#!/usr/bin/env python3
"""Check the capture of vLLM engine counters over a run (`harness/server_metrics.py`).

Reason. With DP > 1 the engine log does not print the cache hit share
(`api_server_count` is set equal to `data_parallel_size`, which disables the text
statistics), so no run on three GPUs had it. A run now scrapes `/metrics` before
the rollouts and after.

The test brings its own server on a local port: on a real one the counters grow
from other people's work, and the check would measure the node.

What is checked:

1. counters are summed over label sets (DP replicas), foreign names and
   comments are skipped, an address with `/v1` and without gives the same result;
2. no server gives an empty result, not an exception;
3. shares come from the difference, not from the accumulated counter;
4. a counter that went down (server restart) gives `restarted`, without shares;
5. a role missing one of the two snapshots is left out of the summary;
6. the report prints a section from the summary.
"""

from __future__ import annotations

import contextlib
import io
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_ROOT))

from cad_agent.harness import report as report_mod  # noqa: E402
from cad_agent.harness import server_metrics as sm  # noqa: E402

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail and not ok else ""))
    if not ok:
        FAILED.append(name)


STATE = {"served": 0}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 — name from the base class
        pass

    def do_GET(self):
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        n = STATE["served"]
        body = "# HELP vllm:prefix_cache_queries_total x\n# TYPE vllm:prefix_cache_queries_total counter\n"
        for engine in (0, 1, 2):
            body += (
                f'vllm:prefix_cache_queries_total{{engine="{engine}",model_name="a"}} {1000.0 * n}\n'
                f'vllm:prefix_cache_hits_total{{engine="{engine}",model_name="a"}} {250.0 * n}\n'
                f'vllm:prompt_tokens_total{{engine="{engine}",model_name="a"}} {1000.0 * n}\n'
                f'vllm:generation_tokens_total{{engine="{engine}",model_name="a"}} {10.0 * n}\n'
                f'vllm:prefix_cache_queries_created{{engine="{engine}"}} 1.7e9\n'
                f'vllm:kv_cache_usage_perc{{engine="{engine}"}} 0.5\n'
            )
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"

    print("1. parsing /metrics")
    STATE["served"] = 2
    got = sm.scrape(url + "/v1")
    check("sum over three replicas", got.get("vllm:prefix_cache_queries_total") == 6000.0, str(got))
    check("foreign names are not taken", set(got) <= set(sm.COUNTERS), str(sorted(got)))
    check("address with and without /v1 gives the same result", sm.scrape(url) == got)
    check("a missing counter is simply absent", "vllm:mm_cache_queries_total" not in got)

    print("2. no server")
    check("empty, no exception", sm.scrape(f"http://127.0.0.1:{free_port()}/v1", timeout=2.0) == {})

    print("3. difference and shares")
    config = {"server": {"generation_base_url": f"http://127.0.0.1:{free_port()}/v1",
                         "assistant_base_url": url + "/v1"}}
    before = sm.snapshot(config)
    STATE["served"] = 6
    after = sm.snapshot(config)
    summary = sm.summarize(before, after, 100.0)
    a = summary.get("assistant") or {}
    check("hit share from the difference", a.get("prefix_cache_hit_share") == 0.25, str(a))
    check("prompt tok/s from the difference", a.get("prompt_tok_per_sec") == 120.0, str(a))
    check("share without a counter is None", a.get("mm_cache_hit_share") is None, str(a))
    check("window is recorded", a.get("window_sec") == 100.0, str(a))

    print("4. restart")
    restarted = sm.delta({"vllm:prefix_cache_queries_total": 50.0, "vllm:prefix_cache_hits_total": 10.0},
                         {"vllm:prefix_cache_queries_total": 20.0, "vllm:prefix_cache_hits_total": 5.0}, 10.0)
    check("restarted and no shares", restarted.get("restarted") is True
          and "prefix_cache_hit_share" not in restarted, str(restarted))

    print("5. role without a snapshot")
    check("unavailable generator is not in the summary", "generation" not in summary, str(sorted(summary)))

    print("6. report")
    run = report_mod.Run.__new__(report_mod.Run)
    run.summary = {"servers": summary}
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        report_mod.section_servers(run)
    text = buffer.getvalue()
    check("section is printed with the share", "assistant" in text and "0.25" in text, text)

    server.shutdown()
    print("\nRESULT:", "OK" if not FAILED else f"FAIL ({len(FAILED)}): {', '.join(FAILED)}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
