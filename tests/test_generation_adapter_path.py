"""The completion adapter guard resolves the mounted volume and candidate alike."""
import ast
from pathlib import Path

import pytest


def adapter_guard(monkeypatch, volume_alias):
    """Run the entrypoint's real guard with /data mapped onto a local mount alias."""
    import slc.generation_jobs as persistence

    def mounted_path(value):
        return volume_alias if value == "/data" else Path(value)

    source = Path(__file__).resolve().parents[1] / "completion_generation.py"
    module = ast.parse(source.read_text())
    execute = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                   and node.name == "_execute_generation")
    guard = next(node for node in execute.body if isinstance(node, ast.If)
                 and isinstance(node.test, ast.Name) and node.test.id == "adapter_path")
    function = ast.parse("def check(adapter_path):\n    pass\n").body[0]
    function.body = [guard]
    namespace = {"Path": mounted_path, "persistence": persistence}
    monkeypatch.setattr(persistence, "Path", mounted_path)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(source), "exec"), namespace)
    return namespace["check"]


@pytest.fixture
def mounted_volume(tmp_path):
    storage = tmp_path / "physical-volume"
    storage.mkdir()
    alias = tmp_path / "data"
    alias.symlink_to(storage, target_is_directory=True)
    (storage / "adapter").mkdir()
    return alias, storage


def test_adapter_inside_symlinked_volume_passes_entrypoint_guard(monkeypatch, mounted_volume):
    alias, storage = mounted_volume
    guard = adapter_guard(monkeypatch, alias)
    guard(str(alias / "adapter"))
    assert (alias / "adapter").resolve() == storage / "adapter"


@pytest.mark.parametrize("escape", ["direct", "symlink", "parent", "relative"])
def test_entrypoint_guard_rejects_paths_outside_resolved_volume(monkeypatch, mounted_volume, tmp_path, escape):
    alias, storage = mounted_volume
    outside = tmp_path / "outside"
    outside.mkdir()
    (storage / "escape").symlink_to(outside, target_is_directory=True)
    candidate = {
        "direct": str(outside), "symlink": str(alias / "escape"),
        "parent": str(alias / ".." / "outside"), "relative": "adapter",
    }[escape]
    with pytest.raises(ValueError, match="under /data"):
        adapter_guard(monkeypatch, alias)(candidate)


def test_base_model_empty_adapter_still_passes_guard(monkeypatch, mounted_volume):
    alias, _ = mounted_volume
    adapter_guard(monkeypatch, alias)("")
