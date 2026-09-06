"""Explicit metadata paths for portable plans; evidence content remains opaque."""
from contextlib import contextmanager
from contextvars import ContextVar
import copy
import hashlib
import json
import os
from pathlib import Path


_CURRENT = ContextVar('slc_analysis_paths', default=None)
MODEL_PATHS = ('battery_path', 'responses_path', 'predictions_path')


class RepositoryPaths:
    def __init__(self, root, plan_directory):
        self.root = Path(root).resolve()
        self.directory = self.inside(plan_directory)

    def inside(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('portable path is outside the declared repository root')
        return path

    def repository_path(self, value):
        path = Path(value)
        return self.inside(path if path.is_absolute() else self.root / path).relative_to(self.root).as_posix()

    def operational_path(self, value):
        path = Path(value)
        return Path(os.path.relpath(self.inside(path if path.is_absolute() else self.directory / path),
                                   self.directory)).as_posix()

    def input(self, value):
        if not isinstance(value, str) or not value or Path(value).is_absolute() or '\\' in value:
            raise ValueError('portable operational paths must be relative to the plan directory')
        return self.inside(self.directory / value)


def _operational_slots(plan):
    for entry in plan.get('models', []):
        for key in MODEL_PATHS:
            if key in entry:
                yield entry, key
        for target in entry.get('targets', []):
            if 'judgments_path' in target:
                yield target, 'judgments_path'
    for entry in plan.get('legacy_gate_outputs', []):
        yield entry, 'path'


def _metadata_slots(plan):
    for source in plan.get('judgment_evidence_manifests', []):
        yield source, 'path'
    for entry in plan.get('models', []):
        for owner in (entry, *entry.get('targets', [])):
            if 'evidence_binding' not in owner:
                continue
            binding = owner['evidence_binding']
            for key in ('planned_output_path', 'source_evidence_path'):
                yield binding, key
            yield binding['manifest_source'], 'path'
            terminal = binding['terminal_batch']
            for key in ('started', 'events_prefix', 'event'):
                yield terminal[key], 'path'
            for key in ('output_path', 'responses_path'):
                if key in terminal['outcome']:
                    yield terminal['outcome'], key


def _evidence_binding(binding, paths):
    """Rebase only named provenance slots through their hash-bound manifest."""
    source = binding['manifest_source']
    manifest_path = paths.inside(source['path'])
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != source['sha256']:
        raise ValueError('portable evidence manifest checksum mismatch')
    manifest = json.loads(raw)
    old_root, old_snapshot = Path(manifest['path_root']), Path(manifest['snapshot_root'])

    def historical(value, snapshot=False):
        old, new = (old_snapshot, manifest_path.parent) if snapshot else (old_root, paths.root)
        value = Path(value)
        if not value.is_absolute():
            value = old / value
        if value.is_relative_to(old):
            value = new / value.relative_to(old)
        return paths.repository_path(value)

    for key in ('planned_output_path', 'source_evidence_path'):
        binding[key] = paths.repository_path(binding[key])
    source['path'] = paths.repository_path(manifest_path)
    terminal = binding['terminal_batch']
    for key in ('started', 'events_prefix', 'event'):
        terminal[key]['path'] = historical(terminal[key]['path'], snapshot=True)
    for key in ('output_path', 'responses_path'):
        if key in terminal['outcome']:
            terminal['outcome'][key] = historical(terminal['outcome'][key])


def make_portable_plan(plan, repo_root, plan_directory):
    """Copy a plan, translating only operational and documented provenance paths.

    Input relative operational paths already use plan_directory. Absolute paths
    must lie within repo_root. Frozen evidence manifests are never rewritten.
    Their recorded roots map only their named metadata slots into the new root.
    """
    paths = RepositoryPaths(repo_root, plan_directory)
    result = copy.deepcopy(plan)
    if 'portable' in result:
        raise ValueError('plan already declares portable paths')
    for entry, key in _operational_slots(result):
        entry[key] = paths.operational_path(entry[key])
    for entry in result.get('models', []):
        for owner in (entry, *entry.get('targets', [])):
            if 'evidence_binding' in owner:
                _evidence_binding(owner['evidence_binding'], paths)
    for source in result.get('judgment_evidence_manifests', []):
        source['path'] = paths.repository_path(source['path'])
    result['portable'] = {'schema_version': 1,
                          'repository_root': Path(os.path.relpath(paths.root, paths.directory)).as_posix()}
    plan_paths(result, paths.directory)
    return result


def plan_paths(plan, base_dir):
    if 'portable' not in plan:
        return None
    declaration = plan['portable']
    if (not isinstance(declaration, dict) or set(declaration) != {'schema_version', 'repository_root'}
            or type(declaration['schema_version']) is not int or declaration['schema_version'] != 1):
        raise ValueError('invalid portable path declaration')
    root = declaration['repository_root']
    if not isinstance(root, str) or not root or Path(root).is_absolute() or '\\' in root:
        raise ValueError('portable repository_root must be relative to the plan directory')
    paths = RepositoryPaths(Path(base_dir) / root, base_dir)
    for entry, key in _operational_slots(plan):
        paths.input(entry[key])
    for entry, key in _metadata_slots(plan):
        value = entry[key]
        if not isinstance(value, str) or not value or Path(value).is_absolute() or '\\' in value:
            raise ValueError('portable metadata paths must be relative to the repository root')
        paths.inside(paths.root / value)
    return paths


@contextmanager
def reporting_paths(plan, base_dir):
    paths = plan_paths(plan, base_dir)
    token = _CURRENT.set(paths)
    try:
        yield paths
    finally:
        _CURRENT.reset(token)


def display_path(path):
    paths = _CURRENT.get()
    return paths.repository_path(path) if paths is not None else str(path)


def resolve_input(base, value):
    paths = _CURRENT.get()
    if paths is not None:
        return paths.input(value)
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def error_metadata(error, context):
    """Format exception path metadata without replacing text inside other strings."""
    paths = _CURRENT.get()
    if paths is None:
        return str(error), context
    context = dict(context)
    if 'path' in context:
        context['path'] = display_path(context['path'])
    if isinstance(error, OSError) and error.filename:
        return f'{type(error).__name__}: {error.strerror}: {display_path(error.filename)}', context
    return str(error), context
