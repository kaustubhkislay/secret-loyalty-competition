import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(sys.argv[1]).resolve()
checkout = Path("/private/tmp/slc-completion-20260905/reproduction_joint_checkout1")
v = root / "results/completion_20260905/verification"
plan_path = v / "reproduction_joint_commands_v1.json"
plan = json.loads(plan_path.read_text())
bundle = json.loads((v / "reproduction_joint_bundle_v1.json").read_text())
overlay = json.loads((v / "reproduction_joint_overlay_v1.json").read_text())
assert overlay["bundle_success_sha256"] == bundle["verification"]["success_sha256"]
env = os.environ.copy()
for key in ("PYTHONPATH", "PYTHONHOME", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
    env.pop(key, None)
env["PYTHONNOUSERSITE"] = "1"
uv_command = ["/Users/kaustubhkislay/.local/bin/uv", "sync", "--frozen", "--extra", "dev", "--offline"]
setup_log = v / "reproduction_joint_install_v1.log"
with setup_log.open("xb") as stream:
    installed = subprocess.run(uv_command, cwd=checkout, env=env, stdout=stream, stderr=subprocess.STDOUT)
assert installed.returncode == 0, "Pinned offline installation failed"
runner = checkout / "results/completion_20260905/verification/reproduction_joint_audit_runner_v1.py"
assert runner.read_bytes() == (v / runner.name).read_bytes()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run_one(item):
    index, command = item
    output_options = ("--out", "--output-dir", "--output-prefix")
    for option in output_options:
        if option in command:
            destination = checkout / command[command.index(option) + 1]
            assert not destination.exists(), destination
    actual = [str(checkout / ".venv/bin/python"), str(runner), str(root), json.dumps(command)]
    log = v / ("reproduction_joint_v1_command" + str(index + 1) + ".log")
    start = datetime.now(timezone.utc).isoformat()
    t = time.monotonic()
    with log.open("xb") as stream:
        process = subprocess.run(actual, cwd=checkout, env=env, stdout=stream, stderr=subprocess.STDOUT)
    audits = [json.loads(line) for line in log.read_text().splitlines() if line.startswith('{"reproduction_audit":')]
    row = {"command": command, "actual_command": actual, "cwd": str(checkout), "exit_code": process.returncode,
           "expected_exit_code": plan["expected_exit_codes"][index], "started_at": start,
           "finished_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": round(time.monotonic() - t, 3),
           "log_path": str(log.relative_to(root)), "log_sha256": sha(log), "audit": audits[-1] if len(audits) == 1 else None}
    print(json.dumps(row), flush=True)
    return row

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    commands = list(pool.map(run_one, enumerate(plan["commands"])))
comparisons = []
for pair in plan["comparisons"]:
    expected = root / pair["expected_path"]
    actual = checkout / pair["reproduced_path"]
    assert sha(expected) == pair["expected_sha256"]
    actual_sha = sha(actual) if actual.is_file() else None
    comparisons.append({**pair, "reproduced_sha256": actual_sha, "identical": actual_sha == pair["expected_sha256"]})
checks = {"all_outputs_identical": all(row["identical"] for row in comparisons),
          "all_exit_codes_expected": all(row["exit_code"] == row["expected_exit_code"] for row in commands),
          "all_offline_audits_passed": all(row["audit"] and row["audit"]["reproduction_audit"] == {"original_path_denials": 1, "network_denials": 0} for row in commands)}
report = {"schema_version": "completion-joint-reproduction-v1", "base_commit": plan["base_commit"],
          "checkout": str(checkout), "bundle_success_sha256": bundle["verification"]["success_sha256"],
          "command_plan_path": str(plan_path.relative_to(root)), "command_plan_sha256": sha(plan_path),
          "audit_runner_sha256": sha(runner), "environment_setup": {"command": uv_command, "exit_code": installed.returncode,
          "log_path": str(setup_log.relative_to(root)), "log_sha256": sha(setup_log)},
          "commands": commands, "comparisons": comparisons, **checks, "passed": all(checks.values()),
          "verified_at": datetime.now(timezone.utc).isoformat(),
          "scope": "All four joint reports, full legacy phrase rerun, historical gates, prospective calibration and full training audit. Corrected sequential analyses remain required. Missing/uncertain labels and scientific limitations remain unchanged."}
with (v / "reproduction_joint_v1.json").open("x") as stream:
    json.dump(report, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps({**checks, "passed": report["passed"], "outputs": len(comparisons)}), flush=True)
raise SystemExit(0 if report["passed"] else 1)
