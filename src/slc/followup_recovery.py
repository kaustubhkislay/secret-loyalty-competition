"""Bounded recovery of one interrupted Suite 2 call; preserve its original tree."""
import os
from pathlib import Path

from slc.followup_runtime import file_sha


TAG = 'suite2_SthenN_s2'
ORIGINAL_CALL_ID = 'fc-01M1X84SZV5643AZ563WVSYGW9'


def tree_hashes(directory):
    directory = Path(directory)
    paths = sorted(directory.rglob('*'))
    if any(path.is_symlink() for path in paths):
        raise ValueError('recovery artifacts must not contain symbolic links')
    return {str(path.relative_to(directory)): file_sha(path) for path in paths if path.is_file()}


def archive_partial(target, backup, expected):
    """Move the inspected incomplete tree once; never replace an existing archive."""
    target, backup = Path(target), Path(backup)
    if target.name != TAG:
        raise ValueError('recovery covers only the single authorized training tag')
    required = {'STARTED.json', 'training.jsonl', 'model/training_order.jsonl'}
    if set(expected) != required:
        raise ValueError('the inspection must show only the original partial trace and inputs')
    if backup.exists():
        if tree_hashes(backup) != expected:
            raise ValueError('the preserved original attempt differs from its inspection')
        return expected
    if (target / 'TRAINED.json').exists():
        raise ValueError('completed training must never enter recovery')
    if not target.is_dir() or tree_hashes(target) != expected:
        raise ValueError('the original attempt differs from its read-only inspection')
    backup.parent.mkdir(parents=True, exist_ok=True)
    os.rename(target, backup)
    if tree_hashes(backup) != expected:
        raise ValueError('the archived attempt differs from its inspection')
    return expected


def call_frozen_training(frozen, isolated_claims, job, plan_hash, parent):
    """Use the frozen function with a separate claim; retain the old claim unchanged."""
    if job['tag'] != TAG:
        raise ValueError('recovery covers only the single authorized training tag')
    original_claims = frozen.claims
    try:
        frozen.claims = isolated_claims
        return frozen._training(job, plan_hash, parent)
    finally:
        frozen.claims = original_claims


def verify_missing_evaluations(generation_root, tag, batteries):
    if tag != TAG or len(batteries) != 3:
        raise ValueError('recovery requires the single authorized tag and its three batteries')
    for name in batteries:
        if (Path(generation_root) / tag / name).exists():
            raise ValueError('an evaluation output already exists; inspect its handle before recovery')
