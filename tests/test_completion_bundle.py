"""Portable snapshots use synthetic bytes and temporary Git repositories only."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True).stdout


def put(root, path, data):
    destination = root / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return destination


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    git(root, "init", "-q")
    put(root, "src/example.py", b"VALUE = 1\n")
    put(root, "configs/example.yaml", b"seed: 1\n")
    git(root, "add", ".")
    git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-qm", "Synthetic base fixture")
    base = git(root, "rev-parse", "HEAD").decode().strip()
    put(root, "src/example.py", b"VALUE = 2\n")
    put(root, "data/frozen.bin", b"\x00\xff\x01")
    script = put(root, "scripts/rebuild.py", b"#!/usr/bin/env python3\nprint('fixture')\n")
    script.chmod(0o755)
    return root, base


def select(source, paths=None, **kwargs):
    from slc.completion_bundle import build_selection
    root, base = source
    return build_selection(root, paths or ["src/example.py", "data/frozen.bin", "scripts/rebuild.py"],
                           base_commit=base, **kwargs)


def snapshot(source, tmp_path, **kwargs):
    from slc.completion_bundle import create_bundle
    selection = select(source, **kwargs)
    destination = tmp_path / "bundle"
    create_bundle(source[0], selection, destination)
    return destination, selection


def test_selection_and_bundle_preserve_relative_bytes_hashes_base_and_executability(source, tmp_path):
    from slc.completion_bundle import verify_bundle
    destination, selection = snapshot(source, tmp_path)
    assert selection["base_commit"] == source[1]
    rows = {row["path"]: row for row in selection["files"]}
    assert rows["src/example.py"]["base_sha256"] == hashlib.sha256(b"VALUE = 1\n").hexdigest()
    assert rows["scripts/rebuild.py"]["executable"] is True
    assert rows["data/frozen.bin"]["base_sha256"] is None
    for path, row in rows.items():
        payload = destination / "payload" / path
        assert payload.read_bytes() == (source[0] / path).read_bytes()
        assert hashlib.sha256(payload.read_bytes()).hexdigest() == row["sha256"]
    result = verify_bundle(destination)
    assert result["n_files"] == 3 and result["base_commit"] == source[1]
    assert len(result["success_sha256"]) == 64


def test_order_and_machine_location_do_not_change_frozen_manifest_bytes(source, tmp_path):
    from slc.completion_bundle import create_bundle
    first, selection = snapshot(source, tmp_path)
    second = tmp_path / "another-bundle"
    reversed_selection = select(source, list(reversed([row["path"] for row in selection["files"]])))
    assert selection == reversed_selection
    create_bundle(source[0], reversed_selection, second)
    for filename in ("selection.json", "manifest.json", "SUCCESS.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    assert str(source[0]) not in (first / "selection.json").read_text()


@pytest.mark.parametrize("path", ["../escape", "/absolute", "./src/example.py", "src//example.py",
                                 "src/../example.py", ".git/config", ".venv/example.py", "venv/a",
                                 ".env", "private/api-key.txt", "secrets/token", "data/live.partial",
                                 "data/live.lock", "data/live.tmp"])
def test_unsafe_or_live_paths_are_rejected(source, path):
    with pytest.raises(ValueError):
        select(source, [path])


def test_directories_and_symlinks_are_never_expanded_or_followed(source, tmp_path):
    root, _ = source
    with pytest.raises(ValueError, match="regular file|explicit file"):
        select(source, ["src"])
    outside = put(tmp_path, "outside/file.txt", b"outside")
    (root / "alias").symlink_to(outside.parent, target_is_directory=True)
    (root / "alias-file").symlink_to(outside)
    for path in ("alias/file.txt", "alias-file"):
        with pytest.raises(ValueError, match="symlink"):
            select(source, [path])


def test_credentials_are_rejected_without_repeating_their_bytes(source):
    key = b"sk-or-v1-" + b"e" * 64
    put(source[0], "data/innocent.txt", key)
    with pytest.raises(ValueError, match="credential") as caught:
        select(source, ["data/innocent.txt"])
    assert key.decode() not in str(caught.value)


def test_audit_files_need_an_explicit_selected_completion_guard(source, tmp_path):
    root, _ = source
    put(root, "results/labels.jsonl", b'{"complete":true}\n')
    put(root, "results/labels.jsonl.diagnostics.jsonl", b'{"attempt":1}\n')
    paths = ["results/labels.jsonl", "results/labels.jsonl.diagnostics.jsonl"]
    with pytest.raises(ValueError, match="completion guard"):
        select(source, paths)
    guards = {paths[1]: paths[0]}
    bundle, selection = snapshot(source, tmp_path, paths=paths, completion_guards=guards)
    assert selection["completion_guards"] == guards
    assert (bundle / "SUCCESS.json").is_file()


def test_source_changes_after_selection_fail_before_publishing_success(source, tmp_path):
    from slc.completion_bundle import create_bundle
    selection = select(source)
    put(source[0], "src/example.py", b"VALUE = 3\n")
    destination = tmp_path / "bundle"
    with pytest.raises(ValueError, match="mismatch|changed"):
        create_bundle(source[0], selection, destination)
    assert not (destination / "SUCCESS.json").exists()


def test_interrupted_copy_leaves_an_unusable_snapshot_and_never_reuses_its_name(source, tmp_path, monkeypatch):
    from slc import completion_bundle as bundle
    selection = select(source)
    destination = tmp_path / "interrupted"
    original = bundle._copy_selected
    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("synthetic interrupted copy")
    monkeypatch.setattr(bundle, "_copy_selected", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        bundle.create_bundle(source[0], selection, destination)
    assert not (destination / "SUCCESS.json").exists()
    with pytest.raises(ValueError, match="incomplete"):
        bundle.verify_bundle(destination)
    with pytest.raises(FileExistsError):
        bundle.create_bundle(source[0], selection, destination)


@pytest.mark.parametrize("defect", ["missing", "corrupt", "extra", "symlink", "manifest", "anchor"])
def test_verification_rejects_missing_corrupt_unlisted_or_unanchored_evidence(source, tmp_path, defect):
    from slc.completion_bundle import verify_bundle
    destination, _ = snapshot(source, tmp_path)
    payload = destination / "payload" / "src/example.py"
    expected_anchor = verify_bundle(destination)["success_sha256"]
    if defect == "missing":
        payload.unlink()
    elif defect == "corrupt":
        payload.chmod(0o644)
        payload.write_bytes(b"VALUE = 9\n")
    elif defect == "extra":
        put(destination, "payload/unselected.txt", b"unexpected")
    elif defect == "symlink":
        payload.unlink()
        payload.symlink_to(source[0] / "src/example.py")
    elif defect == "manifest":
        path = destination / "manifest.json"
        path.chmod(0o644)
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        expected_anchor = "0" * 64
    with pytest.raises(ValueError):
        verify_bundle(destination, expected_success_sha256=expected_anchor)


def test_overlay_applies_only_to_base_or_identical_files_and_can_resume(source, tmp_path):
    from slc.completion_bundle import overlay_bundle
    destination, _ = snapshot(source, tmp_path)
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "clone", "--no-hardlinks", "-q", str(source[0]), str(checkout)], check=True)
    first = overlay_bundle(destination, checkout)
    assert first["written"] == 3 and first["reused"] == 0
    assert (checkout / "src/example.py").read_bytes() == b"VALUE = 2\n"
    assert (checkout / "configs/example.yaml").read_bytes() == b"seed: 1\n"
    second = overlay_bundle(destination, checkout)
    assert second["written"] == 0 and second["reused"] == 3


def test_overlay_checks_all_selected_preimages_before_any_replacement(source, tmp_path):
    from slc.completion_bundle import overlay_bundle
    destination, _ = snapshot(source, tmp_path)
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "clone", "--no-hardlinks", "-q", str(source[0]), str(checkout)], check=True)
    put(checkout, "data/frozen.bin", b"local work")
    with pytest.raises(ValueError, match="local|preimage"):
        overlay_bundle(destination, checkout)
    assert (checkout / "src/example.py").read_bytes() == b"VALUE = 1\n"
    assert not (checkout / "scripts/rebuild.py").exists()


def test_overlay_rejects_wrong_base_and_unrelated_local_changes(source, tmp_path):
    from slc.completion_bundle import overlay_bundle
    destination, _ = snapshot(source, tmp_path)
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "clone", "--no-hardlinks", "-q", str(source[0]), str(checkout)], check=True)
    put(checkout, "configs/example.yaml", b"local: work\n")
    with pytest.raises(ValueError, match="local"):
        overlay_bundle(destination, checkout)
    git(checkout, "add", ".")
    git(checkout, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-qm", "Different fixture base")
    with pytest.raises(ValueError, match="base commit"):
        overlay_bundle(destination, checkout)


def test_selection_file_is_create_only_and_cli_requires_an_explicit_path_list(source, tmp_path):
    paths = put(tmp_path, "paths.txt", b"src/example.py\ndata/frozen.bin\n")
    selection = tmp_path / "selection.json"
    script = Path(__file__).parents[1] / "scripts" / "completion_bundle.py"
    command = [sys.executable, str(script), "select", "--root", str(source[0]), "--paths", str(paths),
               "--base-commit", source[1], "--out", str(selection)]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    frozen = selection.read_bytes()
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode != 0 and selection.read_bytes() == frozen


def test_dependency_lockfiles_are_frozen_inputs_and_not_live_process_locks(source, tmp_path):
    put(source[0], "uv.lock", b"version = 1\n")
    put(source[0], "requirements-modal.lock", b"fixture-package==1.0\n")
    destination, _ = snapshot(source, tmp_path, paths=["uv.lock", "requirements-modal.lock"])
    assert (destination / "payload/uv.lock").read_bytes() == b"version = 1\n"


@pytest.mark.parametrize("filename", ["uv.lock", "requirements-modal.lock"])
@pytest.mark.parametrize("format", ["mapping", "files"])
def test_archived_dependency_locks_require_their_selected_hash_manifest(source, tmp_path, filename, format):
    from slc.completion_bundle import overlay_bundle
    prefix = "artifacts/completion_fixture/code_snapshots/frozen_source/"
    name, guard = prefix + filename, prefix + "SHA256.json"
    raw = b"fixture-package==1.0\n"
    entry = {"path": filename, "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
    manifest = {filename: entry["sha256"]} if format == "mapping" else {"files": [entry]}
    put(source[0], name, raw)
    put(source[0], guard, json.dumps(manifest).encode())
    with pytest.raises(ValueError, match="completion guard"):
        select(source, [name, guard])
    destination, _ = snapshot(source, tmp_path, paths=[name, guard], completion_guards={name: guard})
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "clone", "--no-hardlinks", "-q", str(source[0]), str(checkout)], check=True)
    overlay_bundle(destination, checkout)
    assert (checkout / name).read_bytes() == raw
    assert json.loads((checkout / guard).read_bytes()) == manifest


@pytest.mark.parametrize("defect", ["hash", "size", "missing", "duplicate", "wrong_guard"])
def test_archived_dependency_lock_manifest_must_bind_its_original_bytes(source, defect):
    prefix = "artifacts/completion_fixture/code_snapshots/frozen_source/"
    name, guard = prefix + "uv.lock", prefix + "SHA256.json"
    raw = b"version = 1\n"
    entry = {"path": "uv.lock", "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
    if defect == "hash":
        entry["sha256"] = "0" * 64
    elif defect == "size":
        entry["size_bytes"] += 1
    manifest = {"files": [] if defect == "missing" else [entry, entry] if defect == "duplicate" else [entry]}
    if defect == "wrong_guard":
        guard = prefix + "other.json"
    put(source[0], name, raw)
    put(source[0], guard, json.dumps(manifest).encode())
    with pytest.raises(ValueError, match="archived dependency lock"):
        select(source, [name, guard], completion_guards={name: guard})


@pytest.mark.parametrize("name", [
    "artifacts/completion_fixture/judgment_evidence/run/batch.lock",
    "artifacts/completion_fixture/judgment_evidence/run/uv.lock",
    "artifacts/completion_fixture/code_snapshots/frozen_source/batch.lock",
    "artifacts/completion_fixture/code_snapshots/frozen_source/nested/uv.lock",
])
def test_archive_allowance_does_not_admit_judgment_or_other_process_locks(source, name):
    put(source[0], name, b"pid: 123\n")
    with pytest.raises(ValueError, match="live partial or lock"):
        select(source, [name])


def test_guard_cycles_and_unselected_guards_are_rejected(source):
    paths = ["results/a.events.jsonl", "results/b.events.jsonl"]
    for path in paths:
        put(source[0], path, b"{}\n")
    for guards in ({paths[0]: paths[1], paths[1]: paths[0]},
                   {paths[0]: "missing.json", paths[1]: "missing.json"}):
        with pytest.raises(ValueError, match="completion guard"):
            select(source, paths, completion_guards=guards)


def test_source_mutation_during_copy_prevents_success_even_if_earlier_copy_matches(source, tmp_path, monkeypatch):
    from slc import completion_bundle as bundle
    selection = select(source)
    original = bundle._copy_selected
    def changed(root, record, destination):
        original(root, record, destination)
        if record["path"] == "src/example.py":
            put(root, "data/frozen.bin", b"modified after its copy")
    monkeypatch.setattr(bundle, "_copy_selected", changed)
    destination = tmp_path / "changing"
    with pytest.raises(ValueError, match="mismatch"):
        bundle.create_bundle(source[0], selection, destination)
    assert not (destination / "SUCCESS.json").exists()


def test_credential_like_filenames_cannot_enter_selection_metadata(source):
    key = "sk-or-v1-" + "q" * 64
    name = "data/" + key + ".txt"
    put(source[0], name, b"innocent bytes")
    with pytest.raises(ValueError, match="credential") as caught:
        select(source, [name])
    assert key not in str(caught.value)


@pytest.mark.parametrize("boundary", [False, True])
def test_json_unicode_escaped_credentials_are_rejected_as_bytes_without_semantic_read(source, boundary):
    key = "sk-or-v1-" + "q" * 64
    encoded = "".join(f"\\u{ord(character):04x}" for character in key).encode()
    prefix = b" " * (1024 * 1024 - 80) if boundary else b""
    put(source[0], "data/encoded.json", prefix + b'{"credential":"' + encoded + b'"}\n')
    with pytest.raises(ValueError, match="credential") as caught:
        select(source, ["data/encoded.json"])
    assert key not in str(caught.value)


@pytest.mark.parametrize("action", ["create", "selection", "overlay"])
def test_all_output_operations_reject_symlink_ancestors(source, tmp_path, action):
    from slc.completion_bundle import create_bundle, overlay_bundle, write_selection
    selected = select(source)
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "linked-parent"
    alias.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        if action == "create":
            create_bundle(source[0], selected, alias / "snapshot")
        elif action == "selection":
            write_selection(selected, alias / "selection.json")
        else:
            destination, _ = snapshot(source, tmp_path)
            checkout = outside / "checkout"
            subprocess.run(["git", "clone", "--no-hardlinks", "-q", str(source[0]), str(checkout)], check=True)
            overlay_bundle(destination, alias / "checkout")
    assert not (outside / "snapshot").exists()
    assert not (outside / "selection.json").exists()


def test_overlay_refuses_staged_local_work_even_when_working_bytes_match_the_base(source, tmp_path):
    from slc.completion_bundle import overlay_bundle
    destination, _ = snapshot(source, tmp_path)
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "clone", "--no-hardlinks", "-q", str(source[0]), str(checkout)], check=True)
    put(checkout, "src/example.py", b"STAGED_LOCAL_WORK = True\n")
    git(checkout, "add", "src/example.py")
    put(checkout, "src/example.py", b"VALUE = 1\n")
    with pytest.raises(ValueError, match="staged local"):
        overlay_bundle(destination, checkout)
    assert (checkout / "src/example.py").read_bytes() == b"VALUE = 1\n"


@pytest.mark.skipif(sys.platform != "darwin", reason="requires real macOS file cloning")
def test_clone_bundle_preserves_identity_and_source_mutations_are_independent(source, tmp_path):
    from slc.completion_bundle import create_bundle, verify_bundle
    ordinary, selection = snapshot(source, tmp_path)
    cloned = tmp_path / "cloned"
    result = create_bundle(source[0], selection, cloned, clone_files=True)
    assert result == verify_bundle(ordinary)
    for name in ("selection.json", "manifest.json", "SUCCESS.json"):
        assert (cloned / name).read_bytes() == (ordinary / name).read_bytes()
    original = source[0] / "scripts/rebuild.py"
    payload = cloned / "payload/scripts/rebuild.py"
    assert original.stat().st_ino != payload.stat().st_ino
    assert payload.stat().st_nlink == 1
    assert payload.stat().st_mode & 0o777 == 0o555
    original.write_bytes(b"source changed\n")
    assert payload.read_bytes() == b"#!/usr/bin/env python3\nprint('fixture')\n"
    payload.chmod(0o755)
    payload.write_bytes(b"clone changed\n")
    assert original.read_bytes() == b"source changed\n"


@pytest.mark.skipif(sys.platform != "darwin", reason="requires real macOS file cloning")
def test_clone_overlay_preserves_preimages_modes_and_independent_files(source, tmp_path):
    from slc.completion_bundle import overlay_bundle, verify_bundle
    bundle, _ = snapshot(source, tmp_path)
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "clone", "--no-hardlinks", "-q", str(source[0]), str(checkout)], check=True)
    result = overlay_bundle(bundle, checkout, clone_files=True)
    assert result["written"] == 3 and result["reused"] == 0
    assert (checkout / "scripts/rebuild.py").stat().st_mode & 0o777 == 0o755
    target = checkout / "data/frozen.bin"
    payload = bundle / "payload/data/frozen.bin"
    assert target.stat().st_ino != payload.stat().st_ino
    assert target.stat().st_nlink == 1
    assert target.stat().st_mode & 0o777 == 0o644
    assert overlay_bundle(bundle, checkout, clone_files=True)["reused"] == 3
    target.write_bytes(b"checkout changed")
    assert payload.read_bytes() == b"\x00\xff\x01"
    assert verify_bundle(bundle)["n_files"] == 3


@pytest.mark.parametrize("operation", ["create", "overlay"])
def test_clone_option_rejects_other_platforms_before_destination_changes(source, tmp_path, monkeypatch, operation):
    from slc import completion_bundle as bundle
    frozen, selection = snapshot(source, tmp_path)
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "clone", "--no-hardlinks", "-q", str(source[0]), str(checkout)], check=True)
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(ValueError, match="Darwin|macOS"):
        if operation == "create":
            bundle.create_bundle(source[0], selection, tmp_path / "unsupported", clone_files=True)
        else:
            bundle.overlay_bundle(frozen, checkout, clone_files=True)
    assert not (tmp_path / "unsupported").exists()
    assert (checkout / "src/example.py").read_bytes() == b"VALUE = 1\n"
    assert not (checkout / "data/frozen.bin").exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="requires macOS clone option")
@pytest.mark.parametrize("operation", ["create", "overlay"])
def test_unsupported_clone_does_not_fall_back_to_copy_or_publish_success(source, tmp_path, monkeypatch, operation):
    import errno
    from slc import completion_bundle as bundle
    frozen, selection = snapshot(source, tmp_path)
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "clone", "--no-hardlinks", "-q", str(source[0]), str(checkout)], check=True)
    def unsupported(source_descriptor, destination):
        raise OSError(errno.ENOTSUP, "synthetic filesystem does not support cloning")
    monkeypatch.setattr(bundle, "_clone_file_descriptor", unsupported, raising=False)
    with pytest.raises(OSError, match="does not support cloning"):
        if operation == "create":
            bundle.create_bundle(source[0], selection, tmp_path / "unsupported", clone_files=True)
        else:
            bundle.overlay_bundle(frozen, checkout, clone_files=True)
    assert not (tmp_path / "unsupported/SUCCESS.json").exists()
    assert not (tmp_path / "unsupported/payload/data/frozen.bin").exists()
    assert (checkout / "src/example.py").read_bytes() == b"VALUE = 1\n"
    assert not (checkout / "data/frozen.bin").exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="requires real macOS file cloning")
def test_clone_rejects_existing_file_and_bundle_destinations(source, tmp_path):
    from slc import completion_bundle as bundle
    selection = select(source)
    target = put(tmp_path, "already-present", b"local work")
    record = next(row for row in selection["files"] if row["path"] == "data/frozen.bin")
    with pytest.raises(FileExistsError):
        bundle._clone_selected(source[0], record, target)
    assert target.read_bytes() == b"local work"
    with pytest.raises(FileExistsError):
        bundle.create_bundle(source[0], selection, target, clone_files=True)
    assert target.read_bytes() == b"local work"


@pytest.mark.skipif(sys.platform != "darwin", reason="requires real macOS file cloning")
def test_source_mutation_during_clone_prevents_success(source, tmp_path, monkeypatch):
    from slc import completion_bundle as bundle
    selection = select(source)
    original = getattr(bundle, "_clone_file_descriptor", None)
    def changed(source_descriptor, destination):
        original(source_descriptor, destination)
        put(source[0], "data/frozen.bin", b"changed during clone")
    monkeypatch.setattr(bundle, "_clone_file_descriptor", changed, raising=False)
    with pytest.raises(ValueError, match="changed|mismatch"):
        bundle.create_bundle(source[0], selection, tmp_path / "changing", clone_files=True)
    assert not (tmp_path / "changing/SUCCESS.json").exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="requires real macOS file cloning")
@pytest.mark.parametrize("defect", ["bytes", "mode", "credential"])
def test_clone_verifies_copied_bytes_mode_and_credentials(source, tmp_path, monkeypatch, defect):
    from slc import completion_bundle as bundle
    selection = select(source)
    original = getattr(bundle, "_clone_file_descriptor", None)
    secret = b"sk-or-v1-" + b"q" * 64
    def corrupt(source_descriptor, destination):
        original(source_descriptor, destination)
        if defect == "mode":
            destination.chmod(0o755)
        else:
            destination.chmod(0o644)
            destination.write_bytes(secret if defect == "credential" else b"corrupt clone")
    monkeypatch.setattr(bundle, "_clone_file_descriptor", corrupt, raising=False)
    with pytest.raises(ValueError, match="mismatch|credential") as caught:
        bundle.create_bundle(source[0], selection, tmp_path / "corrupt", clone_files=True)
    assert secret.decode() not in str(caught.value)
    assert not (tmp_path / "corrupt/SUCCESS.json").exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="requires real macOS file cloning")
def test_cli_clone_files_creates_and_overlays_without_shared_inodes(source, tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(select(source)))
    script = Path(__file__).parents[1] / "scripts/completion_bundle.py"
    frozen = tmp_path / "cloned-cli"
    created = subprocess.run([sys.executable, str(script), "create", "--root", str(source[0]),
                              "--selection", str(selection), "--out", str(frozen), "--clone-files"],
                             capture_output=True, text=True)
    assert created.returncode == 0, created.stderr
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "clone", "--no-hardlinks", "-q", str(source[0]), str(checkout)], check=True)
    overlaid = subprocess.run([sys.executable, str(script), "overlay", "--bundle", str(frozen),
                               "--checkout", str(checkout), "--clone-files",
                               "--success-sha256", json.loads(created.stdout)["success_sha256"]],
                              capture_output=True, text=True)
    assert overlaid.returncode == 0, overlaid.stderr
    paths = [source[0] / "data/frozen.bin", frozen / "payload/data/frozen.bin", checkout / "data/frozen.bin"]
    assert len({path.stat().st_ino for path in paths}) == 3
    assert all(path.read_bytes() == b"\x00\xff\x01" for path in paths)
