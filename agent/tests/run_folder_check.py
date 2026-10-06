"""Check `run_folder.sh` and `tools/export_results.py` without GPUs or servers.

The script is run from a copy of the repository layout in which `run_system.sh` is a
stub: it records the config it was given and fakes a finished run directory. What is
checked:

1. the generated config points at the folder, carries the name and the limit, keeps
   the rest of the base config, and passes the harness config validator;
2. unknown flags reach `run_system.sh` unchanged;
3. `--out` exports `best.py` / `best.stl` per part and a `results.csv` row per part,
   including a part without code;
4. a missing folder and a folder without `.stl` files are refused before the run.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

AGENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_ROOT))

import run_experiment  # noqa: E402
from cad_agent.harness.config import validate_run_config  # noqa: E402

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail and not ok else ""))
    if not ok:
        FAILED.append(name)


# Stub of `run_system.sh`: remembers its arguments and writes a finished run with two
# parts, one with code and a mesh and one without code.
FAKE_RUN_SYSTEM = r"""#!/usr/bin/env bash
cfg="$1"; shift
echo "$cfg $*" > "$(dirname "$0")/../run_system_args.txt"
run="$(dirname "$cfg")/../parts"
mkdir -p "$run/figures/parts/a"
echo "r = 1" > "$run/figures/parts/a/best.py"
echo "solid a" > "$run/figures/parts/a/best.stl"
printf '[{"figure_id": "parts/a", "group": "parts", "score": 0.9, "metrics": {"iou": 0.9}},
 {"figure_id": "parts/b", "group": "parts", "score": 0.0, "error": "no valid candidate\\nmore"}]' \
    > "$run/per_figure.json"
echo "Run directory:   $(cd "$run" && pwd)"
"""


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="run_folder_check_"))
    try:
        repo = tmp / "repo"
        (repo / "agent" / "tools").mkdir(parents=True)
        (repo / "agent" / "configs").mkdir(parents=True)
        shutil.copy(AGENT_ROOT / "run_folder.sh", repo / "agent")
        shutil.copy(AGENT_ROOT / "tools" / "export_results.py", repo / "agent" / "tools")
        base = AGENT_ROOT / "configs" / "dialogue_lean.yaml"
        shutil.copy(base, repo / "agent" / "configs")
        fake = repo / "agent" / "run_system.sh"
        fake.write_text(FAKE_RUN_SYSTEM)
        fake.chmod(0o755)

        parts = tmp / "parts"
        parts.mkdir()
        for name in ("a", "b", "c"):
            (parts / f"{name}.stl").write_text(f"solid {name}\nendsolid {name}\n")
        out = tmp / "out"
        env = dict(os.environ, CAD_AGENT_PYTHON=sys.executable)
        script = repo / "agent" / "run_folder.sh"

        print("1. Generated config")
        done = subprocess.run([str(script), str(parts), "--limit", "2", "--out", str(out), "--keep-servers"],
                              capture_output=True, text=True, env=env, timeout=120)
        check("run_folder.sh succeeds", done.returncode == 0, done.stdout[-500:] + done.stderr[-500:])
        generated = repo / "work_dirs" / "configs" / "parts.yaml"
        check("config is written under work_dirs/configs", generated.is_file())
        if generated.is_file():
            config = yaml.safe_load(generated.read_text())
            base_config = yaml.safe_load(base.read_text())
            experiment = config["experiment"]
            check("details point at the folder", experiment["details"] == [{"parts": str(parts.resolve())}],
                  str(experiment["details"]))
            check("limit is set", experiment.get("limit") == 2, str(experiment.get("limit")))
            check("run name is the folder name", config["launch"]["run_name"] == "parts")
            check("model and servers are kept from the base",
                  config["model"] == base_config["model"]
                  and config["launch"]["servers"] == base_config["launch"]["servers"])
            run_config = run_experiment.build_run_config(config, None)
            try:
                validate_run_config(run_config)
                check("config passes the validator", True)
            except Exception as exc:  # noqa: BLE001 - the message is the diagnosis
                check("config passes the validator", False, f"{type(exc).__name__}: {exc}")

        print("2. Flags reach run_system.sh")
        args = (repo / "run_system_args.txt").read_text().split()
        check("config and --keep-servers are passed", args[1:] == ["--keep-servers"]
              and args[0].endswith("work_dirs/configs/parts.yaml"), str(args))

        print("3. Export")
        check("code of the part is exported", (out / "a.py").read_text().strip() == "r = 1")
        check("mesh of the part is exported", (out / "a.stl").is_file())
        check("a part without code has no files", not (out / "b.py").exists())
        rows = list(csv.DictReader((out / "results.csv").open()))
        check("results.csv has a row per part", [r["figure_id"] for r in rows] == ["parts/a", "parts/b"],
              str(rows))
        check("error is cut to its first line", rows[1]["error"] == "no valid candidate", str(rows[1]))

        print("4. Refusals before the run")
        missing = subprocess.run([str(script), str(tmp / "nope")], capture_output=True, text=True, env=env)
        check("missing folder is refused", missing.returncode != 0 and "No such directory" in missing.stderr,
              missing.stderr)
        empty = tmp / "empty"
        empty.mkdir()
        no_stl = subprocess.run([str(script), str(empty)], capture_output=True, text=True, env=env)
        check("folder without .stl is refused", no_stl.returncode != 0 and "No .stl files" in no_stl.stderr,
              no_stl.stderr)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nRESULT:", "OK" if not FAILED else f"FAILED ({len(FAILED)}): {', '.join(FAILED)}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
