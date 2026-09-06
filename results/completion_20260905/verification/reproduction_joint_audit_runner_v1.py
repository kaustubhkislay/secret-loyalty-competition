"""Run one offline reproduction command with original-worktree access denied."""
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

blocked = os.path.realpath(sys.argv[1])
checkout = os.path.realpath(os.getcwd())
command = json.loads(sys.argv[2])
counts = {"original_path_denials": 0, "network_denials": 0}

def audit(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.path.realpath(os.path.abspath(os.fsdecode(args[0])))
        if os.path.commonpath([blocked, path]) == blocked:
            counts["original_path_denials"] += 1
            raise PermissionError("Original worktree access is disabled during reproduction")
    if event in ("socket.connect", "socket.getaddrinfo"):
        counts["network_denials"] += 1
        raise PermissionError("Network access is disabled during reproduction")

sys.addaudithook(audit)
try:
    open(os.path.join(blocked, "README.md"), "rb")
except PermissionError:
    pass
else:
    raise RuntimeError("Original-worktree denial probe failed")
assert counts["original_path_denials"] == 1
spec = importlib.util.find_spec("slc")
assert spec and spec.origin and os.path.commonpath([checkout, os.path.realpath(spec.origin)]) == checkout
for path in sys.path:
    assert os.path.commonpath([blocked, os.path.realpath(path or checkout)]) != blocked
sys.argv = command[1:]
code = 0
try:
    runpy.run_path(command[1], run_name="__main__")
except SystemExit as error:
    code = error.code if isinstance(error.code, int) else 0 if error.code is None else 1
finally:
    print(json.dumps({"reproduction_audit": counts, "expected_probe_denials": 1, "local_project_origin": os.path.relpath(spec.origin, checkout)}), file=sys.stderr, flush=True)
raise SystemExit(code)
