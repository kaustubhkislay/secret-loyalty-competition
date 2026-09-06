"""Create and verify local portable snapshots from an explicit frozen selection.

The selection records full hashes, repository-relative paths, executable flags,
and each selected file's preimage at a full Git base commit. There is no recursive
selection, download, publication, dependency install, or analysis execution.
Expected hashes and a stable read reject changing inputs. Mutable audit files
need an explicit selected completion guard; process locks and partial files are forbidden.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile

from slc.artifacts import _artifact_path, _relative_path, verify_manifest


SELECTION_SCHEMA = "completion-selection-v1"
BUNDLE_SCHEMA = "completion-bundle-v1"
FORBIDDEN_PARTS = {".git", ".hg", ".svn", ".venv", "venv", "__pycache__", ".pytest_cache",
                   ".mypy_cache", ".ruff_cache", "node_modules", ".ssh", ".aws", "secrets", ".secrets"}
SECRET_NAME = re.compile(r"(^|[-_.])(api[-_]?key|openrouter[-_]?key|password|credentials?)([-_.]|$)")
CREDENTIAL = re.compile(rb"sk-(?:or-v1-)?[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
ASCII_ESCAPE = re.compile(rb"\\u00([0-9a-fA-F]{2})")
AUDIT_SUFFIXES = (".events.jsonl", ".started.json", ".diagnostics.jsonl")
DEPENDENCY_LOCKS = {"uv.lock", "requirements-modal.lock"}


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _has_credential(raw):
    # Decode only JSON ASCII escape sequences for token screening. This never
    # parses or interprets the surrounding response, prediction, or other data.
    decoded = ASCII_ESCAPE.sub(lambda match: bytes((int(match[1], 16),)), raw)
    return CREDENTIAL.search(raw) is not None or CREDENTIAL.search(decoded) is not None


def _archived_dependency_lock(name):
    parts = _relative_path(name).parts
    return (len(parts) == 5 and parts[0] == "artifacts" and parts[2] == "code_snapshots"
            and parts[-1] in DEPENDENCY_LOCKS)


def _safe_name(name):
    path = _relative_path(name)
    lowered = [part.lower() for part in path.parts]
    if (any(part in FORBIDDEN_PARTS for part in lowered)
            or any(part == ".env" or part.startswith(".env.") for part in lowered)
            or SECRET_NAME.search(path.name.lower()) or _has_credential(name.encode("utf-8"))
            or path.name.lower() in {"key", "key.txt", "keys.json", "id_rsa", "id_ed25519"}
            or path.suffix.lower() in {".pem", ".p12", ".pfx"}):
        raise ValueError("selection includes a forbidden metadata, environment, or credential path")
    if (name not in DEPENDENCY_LOCKS and not _archived_dependency_lock(name)
            and path.name.lower().endswith((".partial", ".lock", ".tmp", ".temp", ".pending", ".inprogress"))):
        raise ValueError("selection includes a live partial or lock file")
    return path


def _needs_guard(name):
    return (name.lower().endswith(AUDIT_SUFFIXES) or Path(name).name == "STARTED.json"
            or _archived_dependency_lock(name))


def _output_path(path):
    path = Path(path).expanduser().absolute()
    if any(ancestor.is_symlink() for ancestor in (path, *path.parents)):
        raise ValueError("output path or ancestor is a symlink")
    return path


def _git(root, *arguments):
    result = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True)
    if result.returncode:
        raise ValueError("local Git metadata does not satisfy the snapshot request")
    return result.stdout


def _base_commit(root, revision):
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{7,40}", revision):
        raise ValueError("base commit must be a hexadecimal Git commit identifier")
    if Path(os.fsdecode(_git(root, "rev-parse", "--show-toplevel")).strip()).resolve() != root:
        raise ValueError("root must be the Git repository root")
    return _git(root, "rev-parse", "--verify", revision + "^{commit}").decode().strip()


def _base_record(root, commit, name):
    entry = _git(root, "ls-tree", "-z", commit, "--", name)
    if not entry:
        return {"base_sha256": None, "base_executable": None}
    metadata, returned_name = entry.rstrip(b"\0").split(b"\t", 1)
    mode, kind, object_id = metadata.split()
    if os.fsdecode(returned_name) != name or kind != b"blob" or mode not in (b"100644", b"100755"):
        raise ValueError("selected base preimage is not a regular file")
    return {"base_sha256": _sha(_git(root, "cat-file", "blob", object_id.decode())),
            "base_executable": mode == b"100755"}


def _state(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _clone_function():
    if sys.platform != "darwin":
        raise ValueError("clone files require macOS/Darwin and a filesystem that supports cloning")
    try:
        clone = ctypes.CDLL(None, use_errno=True).fclonefileat
    except (AttributeError, OSError) as error:
        raise ValueError("macOS file cloning is unavailable; no copy fallback") from error
    clone.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    clone.restype = ctypes.c_int
    return clone


def _clone_file_descriptor(source_descriptor, destination):
    """Atomically clone the inspected source descriptor to an absent file.

    fclonefileat never falls back to a full copy or links the source inode.
    Resolve each destination ancestor without following symlinks, as _inspect
    does for the source. Unsupported filesystems and existing targets fail.
    """
    clone = _clone_function()
    directories = []
    try:
        descriptor = os.open(destination.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        directories.append(descriptor)
        for part in destination.parent.parts[1:]:
            descriptor = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            directories.append(descriptor)
        if clone(source_descriptor, descriptor, os.fsencode(destination.name), 0) != 0:
            raise OSError(ctypes.get_errno(), "file cloning failed; no copy fallback")
        copied = os.open(destination.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
        try:
            source_info, copied_info = os.fstat(source_descriptor), os.fstat(copied)
            if (not stat.S_ISREG(copied_info.st_mode)
                    or (source_info.st_dev, source_info.st_ino) == (copied_info.st_dev, copied_info.st_ino)):
                raise ValueError("cloned destination must be an independent regular file")
            os.fsync(copied)
        finally:
            os.close(copied)
    finally:
        for descriptor in reversed(directories):
            os.close(descriptor)


def _inspect(root, name, destination=None, *, clone_files=False):
    """Hash a stable regular file, optionally copy it, and screen credential shapes.

    Every ancestor is opened with O_NOFOLLOW. No response text is decoded or
    interpreted; this reads byte chunks for checksums and credential detection.
    """
    path = _artifact_path(root, name)
    if not path.is_file():
        raise ValueError("selection must contain explicit regular files")
    directories = []
    stream = output = None
    try:
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        directories.append(descriptor)
        parts = _relative_path(name).parts
        for part in parts[:-1]:
            descriptor = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            directories.append(descriptor)
        file_descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
        stream = os.fdopen(file_descriptor, "rb")
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("selection must contain explicit regular files")
        if destination is not None:
            destination = _output_path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if clone_files:
                _clone_file_descriptor(stream.fileno(), destination)
            else:
                output = destination.open("xb")
        digest, size, tail = hashlib.sha256(), 0, b""
        while chunk := stream.read(1024 * 1024):
            if _has_credential(tail + chunk):
                raise ValueError("selected bytes contain a credential pattern; contents withheld")
            tail = (tail + chunk)[-256:]
            digest.update(chunk)
            size += len(chunk)
            if output is not None:
                output.write(chunk)
        after = os.fstat(stream.fileno())
        named = os.stat(parts[-1], dir_fd=descriptor, follow_symlinks=False)
        if _state(before) != _state(after) or _state(after) != _state(named):
            raise ValueError("selected source changed during its byte read")
        if output is not None:
            output.flush()
            os.fsync(output.fileno())
        return {"path": name, "size_bytes": size, "sha256": digest.hexdigest(),
                "executable": bool(before.st_mode & 0o111)}
    finally:
        if output is not None:
            output.close()
        if stream is not None:
            stream.close()
        for descriptor in reversed(directories):
            os.close(descriptor)


def _check_record(actual, expected):
    if any(actual[key] != expected[key] for key in ("path", "size_bytes", "sha256", "executable")):
        raise ValueError("selected file hash, size, or executable flag mismatch: " + expected["path"])


def _validate_selection(selection):
    if (not isinstance(selection, dict)
            or set(selection) != {"schema_version", "base_commit", "files", "completion_guards"}
            or selection["schema_version"] != SELECTION_SCHEMA
            or not isinstance(selection["base_commit"], str)
            or not re.fullmatch(r"[0-9a-f]{40}", selection["base_commit"])):
        raise ValueError("invalid portable selection metadata")
    records, guards = selection["files"], selection["completion_guards"]
    if not isinstance(records, list) or not records or not isinstance(guards, dict):
        raise ValueError("selection needs explicit files and completion guards")
    names = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {
                "path", "size_bytes", "sha256", "executable", "base_sha256", "base_executable"}:
            raise ValueError("invalid selection file record")
        name = _safe_name(record["path"]).as_posix()
        if (type(record["size_bytes"]) is not int or record["size_bytes"] < 0
                or not isinstance(record["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
                or type(record["executable"]) is not bool):
            raise ValueError("selection needs full hashes, sizes, and executable flags")
        if record["base_sha256"] is None:
            if record["base_executable"] is not None:
                raise ValueError("missing base preimage cannot have an executable flag")
        elif (not isinstance(record["base_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", record["base_sha256"])
              or type(record["base_executable"]) is not bool):
            raise ValueError("invalid base preimage identity")
        names.append(name)
    if names != sorted(set(names)):
        raise ValueError("selection paths must be unique and sorted")
    for name in names:
        if _needs_guard(name) and name not in guards:
            raise ValueError("mutable audit evidence requires an explicit completion guard")
        if (_archived_dependency_lock(name)
                and guards[name] != str(Path(name).with_name("SHA256.json"))):
            raise ValueError("archived dependency lock requires its sibling SHA256.json completion guard")
    for name, guard in guards.items():
        if (name not in names or guard not in names or name == guard or _needs_guard(guard)):
            raise ValueError("completion guard must be a separate selected final file")
    return records


def _verify_archived_dependency_locks(root, records):
    """Bind archived lock bytes to the selected original source snapshot manifest."""
    by_name = {record["path"]: record for record in records}
    for name, record in by_name.items():
        if not _archived_dependency_lock(name):
            continue
        guard = str(Path(name).with_name("SHA256.json"))
        raw = _artifact_path(root, guard).read_bytes()
        if _sha(raw) != by_name[guard]["sha256"] or len(raw) != by_name[guard]["size_bytes"]:
            raise ValueError("archived dependency lock manifest changed from its selected identity")
        manifest = json.loads(raw)
        filename = Path(name).name
        entry = None
        if isinstance(manifest, dict) and isinstance(manifest.get("files"), list):
            matches = [row for row in manifest["files"] if isinstance(row, dict) and row.get("path") == filename]
            if len(matches) == 1:
                entry = matches[0]
        elif isinstance(manifest, dict) and isinstance(manifest.get(filename), str):
            entry = {"sha256": manifest[filename]}
        if (entry is None or entry.get("sha256") != record["sha256"]
                or ("size_bytes" in entry and entry["size_bytes"] != record["size_bytes"])):
            raise ValueError("archived dependency lock does not match its source snapshot manifest")


def build_selection(root, paths, *, base_commit, completion_guards=None):
    root = Path(root).resolve(strict=True)
    names = list(paths)
    if not names or len(set(names)) != len(names):
        raise ValueError("selection paths must be nonempty and unique")
    for name in names:
        _safe_name(name)
    commit = _base_commit(root, base_commit)
    records = [{**_inspect(root, name), **_base_record(root, commit, name)} for name in sorted(names)]
    selection = {"schema_version": SELECTION_SCHEMA, "base_commit": commit, "files": records,
                 "completion_guards": dict(completion_guards or {})}
    _validate_selection(selection)
    _verify_archived_dependency_locks(root, records)
    # Freeze one mutually consistent pass. A writer that changes an earlier
    # selected file while a later file is read invalidates the selection.
    for record in records:
        _check_record(_inspect(root, record["path"]), record)
    return selection


def _publish_new(path, raw):
    path = _output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_selection(selection, output):
    _validate_selection(selection)
    _publish_new(output, _json_bytes(selection))


def _copy_selected(root, record, destination):
    _check_record(_inspect(root, record["path"], destination), record)
    destination.chmod(0o555 if record["executable"] else 0o444)


def _clone_selected(root, record, destination):
    _check_record(_inspect(root, record["path"], destination, clone_files=True), record)
    copied = _inspect(destination.parent, destination.name)
    _check_record({**copied, "path": record["path"]}, record)
    destination.chmod(0o555 if record["executable"] else 0o444)


def _selected_copier(clone_files):
    if type(clone_files) is not bool:
        raise ValueError("clone_files must be a boolean")
    if clone_files:
        _clone_function()  # Reject unsupported platforms before any output.
        return _clone_selected
    return _copy_selected


def _manifest(selection):
    selected = _json_bytes(selection)
    rows = [{"path": "payload/" + row["path"], "size_bytes": row["size_bytes"], "sha256": row["sha256"]}
            for row in selection["files"]]
    rows.append({"path": "selection.json", "size_bytes": len(selected), "sha256": _sha(selected)})
    return {"schema_version": 1, "algorithm": "sha256", "files": sorted(rows, key=lambda row: row["path"])}


def create_bundle(root, selection, destination, *, clone_files=False):
    copy_selected = _selected_copier(clone_files)
    root = Path(root).resolve(strict=True)
    destination = _output_path(destination)
    records = _validate_selection(selection)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("snapshot destination already exists; choose a new path")
    for record in records:
        _check_record(_inspect(root, record["path"]), record)
    _verify_archived_dependency_locks(root, records)
    # An interrupted directory remains incomplete. Its name is never reused.
    destination.mkdir(parents=True, exist_ok=False)
    for record in records:
        copy_selected(root, record, destination / "payload" / record["path"])
    for record in records:
        _check_record(_inspect(root, record["path"]), record)
    _publish_new(destination / "selection.json", _json_bytes(selection))
    manifest = _manifest(selection)
    _publish_new(destination / "manifest.json", _json_bytes(manifest))
    problems = verify_manifest(destination, manifest)
    if problems:
        raise ValueError("copied snapshot verification failed: " + "; ".join(problems))
    success = {"schema_version": BUNDLE_SCHEMA, "base_commit": selection["base_commit"],
               "n_files": len(records), "selection_sha256": _sha(_json_bytes(selection)),
               "manifest_sha256": _sha(_json_bytes(manifest))}
    _publish_new(destination / "SUCCESS.json", _json_bytes(success))
    return verify_bundle(destination)


def verify_bundle(bundle, *, expected_success_sha256=None):
    bundle = Path(bundle).resolve(strict=True)
    success_path = _artifact_path(bundle, "SUCCESS.json")
    if not success_path.is_file():
        raise ValueError("snapshot is incomplete: SUCCESS.json is missing")
    success_raw = success_path.read_bytes()
    anchor = _sha(success_raw)
    if expected_success_sha256 is not None and anchor != expected_success_sha256:
        raise ValueError("snapshot SUCCESS hash differs from the external expected anchor")
    success = json.loads(success_raw)
    selection_raw = _artifact_path(bundle, "selection.json").read_bytes()
    manifest_raw = _artifact_path(bundle, "manifest.json").read_bytes()
    selection, manifest = json.loads(selection_raw), json.loads(manifest_raw)
    records = _validate_selection(selection)
    expected = {"schema_version": BUNDLE_SCHEMA, "base_commit": selection["base_commit"],
                "n_files": len(records), "selection_sha256": _sha(selection_raw), "manifest_sha256": _sha(manifest_raw)}
    if success != expected or manifest != _manifest(selection) or selection_raw != _json_bytes(selection):
        raise ValueError("snapshot manifest or selection identity differs from SUCCESS")
    allowed = {"manifest.json", "SUCCESS.json", *(row["path"] for row in manifest["files"])}
    actual = set()
    for directory, subdirectories, files in os.walk(bundle, followlinks=False):
        for name in subdirectories + files:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError("snapshot contains a symlink")
        actual.update((Path(directory) / name).relative_to(bundle).as_posix() for name in files)
    if actual != allowed:
        raise ValueError("snapshot contains missing or unselected files")
    problems = verify_manifest(bundle, manifest)
    if problems:
        raise ValueError("snapshot checksum verification failed: " + "; ".join(problems))
    for record in records:
        _check_record(_inspect(bundle / "payload", record["path"]), record)
    _verify_archived_dependency_locks(bundle / "payload", records)
    return {**success, "success_sha256": anchor, "size_bytes": sum(row["size_bytes"] for row in records)}


def overlay_bundle(bundle, checkout, *, expected_success_sha256=None, clone_files=False):
    """Apply to the recorded Git base; replace only matching base preimages.

    Identical selected files allow a safe rerun. Unrelated local changes, changed
    selected preimages, wrong commits, and symlinks fail before any replacement.
    The command never deletes files, edits Git metadata, or installs dependencies.
    """
    copy_selected = _selected_copier(clone_files)
    checkout = _output_path(checkout).resolve(strict=True)
    verification = verify_bundle(bundle, expected_success_sha256=expected_success_sha256)
    bundle = Path(bundle).resolve()
    selection = json.loads((bundle / "selection.json").read_bytes())
    records = selection["files"]
    base = _base_commit(checkout, selection["base_commit"])
    if _git(checkout, "rev-parse", "HEAD").decode().strip() != base:
        raise ValueError("checkout HEAD does not match the selected base commit")
    if _git(checkout, "diff", "--cached", "--name-only", "-z"):
        raise ValueError("checkout contains staged local changes")
    selected_names = {row["path"] for row in records}
    changed = set(os.fsdecode(value) for value in _git(checkout, "diff", "--name-only", "-z", "HEAD").split(b"\0") if value)
    changed.update(os.fsdecode(value) for value in _git(checkout, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0") if value)
    if changed - selected_names:
        raise ValueError("checkout contains unrelated local changes")
    actions = []
    for record in records:
        name = record["path"]
        preimage = _base_record(checkout, base, name)
        if any(preimage[key] != record[key] for key in preimage):
            raise ValueError("selection base preimage disagrees with the local Git object")
        target = _artifact_path(checkout, name)
        current = _inspect(checkout, name) if target.exists() else None
        if current and all(current[key] == record[key] for key in ("path", "size_bytes", "sha256", "executable")):
            actions.append((record, "reuse", current))
        elif current and current["sha256"] == record["base_sha256"] and current["executable"] == record["base_executable"]:
            actions.append((record, "replace", current))
        elif current is None and record["base_sha256"] is None:
            actions.append((record, "create", None))
        else:
            raise ValueError("checkout has a local change to a selected preimage: " + name)
    written = reused = 0
    for record, action, current in actions:
        if action == "reuse":
            reused += 1
            continue
        target = _artifact_path(checkout, record["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix="." + target.name + ".", dir=target.parent)
        os.close(descriptor)
        Path(temporary).unlink()
        try:
            copy_selected(bundle / "payload", record, Path(temporary))
            Path(temporary).chmod(0o755 if record["executable"] else 0o644)
            if action == "replace":
                _check_record(_inspect(checkout, record["path"]), current)
                os.replace(temporary, target)
            else:
                os.link(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
        written += 1
    for record in records:
        _check_record(_inspect(checkout, record["path"]), record)
    if _git(checkout, "rev-parse", "HEAD").decode().strip() != base:
        raise ValueError("checkout base commit changed during the overlay")
    return {"base_commit": base, "written": written, "reused": reused,
            "success_sha256": verification["success_sha256"], "n_files": len(records)}
