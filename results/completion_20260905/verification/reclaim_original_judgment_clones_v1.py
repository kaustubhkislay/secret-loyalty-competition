"""One-off, locked APFS replacement of terminal original judgment copies.

The CLI accepts no arbitrary source paths. It uses exactly the authorized main
52, legacy 34, and calibration 2 frozen bindings. Frozen files and bundle inputs
are read-only. A new adjacent APFS clone atomically replaces each identical
original, with its original mode and independent inode. No full-copy fallback.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import uuid

ROOT = Path(__file__).absolute().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from slc.completion_bundle import _clone_file_descriptor, _clone_function, _publish_new, _json_bytes, _state
from slc.artifacts import _relative_path
from slc.judgment_evidence import load_evidence_manifests

VERIFY = 'results/completion_20260905/verification/'
SELECTIONS = [VERIFY + 'reproduction_joint_final_selection_v1.json',
              VERIFY + 'reproduction_joint_final_selection_v2.json']
INDEXES = [('results/completion_20260905/joint_final_frozen_evidence_index_v1.json', 52),
           ('results/completion_20260905/legacy_phrase_full_frozen_evidence_index_v1.json', 34)]
FROZEN = 'artifacts/completion_20260905/judgment_evidence/'
CALIBRATION = [FROZEN + 'calibration_generation_v3/prospective_v3__vendor_' + target + '/manifest.json'
               for target in ('M', 'S')]
ORIGINAL_PREFIXES = ('results/completion_20260905/judgments_v3/',
                     'results/completion_20260905/judgments_v3_legacy_phrase/')


def relative(root, path):
    path = Path(path)
    if path.is_absolute():
        path = path.relative_to(root)
    return _relative_path(path.as_posix()).as_posix()


@contextmanager
def regular(root, name, *, writable=False):
    name = _relative_path(name)
    root = Path(root).absolute()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError('a protected root or ancestor is a symlink')
    descriptors = []
    try:
        parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.append(parent)
        for part in name.parts[:-1]:
            parent = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            descriptors.append(parent)
        flags = (os.O_RDWR if writable else os.O_RDONLY) | os.O_NOFOLLOW | os.O_NONBLOCK
        descriptor = os.open(name.name, flags, dir_fd=parent)
        descriptors.append(descriptor)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError('a protected input is not a regular file')
        yield descriptor, parent, name.name
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def stable(info, descriptor, parent, leaf):
    if _state(info) != _state(os.fstat(descriptor)) or _state(info) != _state(os.stat(leaf, dir_fd=parent, follow_symlinks=False)):
        raise ValueError('a protected file changed during validation')


def checked(descriptor, parent, leaf, expected):
    before = os.fstat(descriptor)
    stable(before, descriptor, parent, leaf)
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest, size = hashlib.sha256(), 0
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
    stable(before, descriptor, parent, leaf)
    if digest.hexdigest() != expected['sha256'] or size != expected['size_bytes']:
        raise ValueError('original or frozen file hash or size differs from the frozen binding')
    return before


def read_metadata(root, name):
    with regular(root, name) as (descriptor, parent, leaf):
        before = os.fstat(descriptor)
        chunks = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        stable(before, descriptor, parent, leaf)
        raw = b''.join(chunks)
    return json.loads(raw), hashlib.sha256(raw).hexdigest(), len(raw)


def observe_space(root):
    info = os.statvfs(root)
    return {'utc': datetime.now(timezone.utc).isoformat(), 'available_bytes': info.f_bavail * info.f_frsize,
            'free_blocks_bytes': info.f_bfree * info.f_frsize, 'block_size': info.f_frsize}


def authorize(job, selected_names):
    if job['terminal_status'] not in ('complete', 'pending', 'failed'):
        raise ValueError('the binding has no terminal job outcome')
    planned = _relative_path(job['planned_output']).as_posix()
    allowed = planned.startswith(ORIGINAL_PREFIXES) or planned in {
        'results/completion_20260905/calibration_generation_v3/judgments_M.jsonl',
        'results/completion_20260905/calibration_generation_v3/judgments_S.jsonl'}
    if not allowed or job['lock'] != planned + '.lock':
        raise ValueError('the original path is outside the authorized judgment scope')
    for candidate in job['candidates']:
        name = _relative_path(candidate['original']).as_posix()
        if name in selected_names:
            raise ValueError('an original destination is selected by the active bundle')
        expected = (planned, planned + '.partial') if candidate['role'] == 'evidence' else (planned + '.diagnostics.jsonl',)
        if name not in expected or not candidate['frozen'].startswith(FROZEN):
            raise ValueError('candidate paths differ from the authorized terminal binding')
        _relative_path(candidate['frozen'])


def clone_job(root, job, selected_names, audit=None):
    authorize(job, selected_names)
    result = []
    with regular(root, job['lock'], writable=True) as (lock_fd, lock_parent, lock_leaf):
        lock_before = os.fstat(lock_fd)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('an active judgment process holds the existing job lock') from None
        for candidate in job['candidates']:
            original, frozen = candidate['original'], candidate['frozen']
            with regular(root, original) as (original_fd, parent, leaf), regular(root, frozen) as (frozen_fd, frozen_parent, frozen_leaf):
                original_before = checked(original_fd, parent, leaf, candidate)
                frozen_before = checked(frozen_fd, frozen_parent, frozen_leaf, candidate)
                if original_before.st_nlink != 1 or (original_before.st_dev, original_before.st_ino) == (frozen_before.st_dev, frozen_before.st_ino):
                    raise ValueError('the original must be a separate regular inode without hardlinks')
                mode = stat.S_IMODE(original_before.st_mode)
                temporary_leaf = '.judgment-clone-' + uuid.uuid4().hex
                temporary_name = str(Path(original).with_name(temporary_leaf))
                temporary_created = False
                try:
                    _clone_file_descriptor(frozen_fd, Path(root) / temporary_name)
                    temporary_created = True
                    with regular(root, temporary_name) as (cloned_fd, cloned_parent, cloned_leaf):
                        os.fchmod(cloned_fd, mode)
                        os.utime(cloned_fd, ns=(original_before.st_atime_ns, original_before.st_mtime_ns))
                        os.fsync(cloned_fd)
                        cloned_info = checked(cloned_fd, cloned_parent, cloned_leaf, candidate)
                        if ((cloned_info.st_dev, cloned_info.st_ino) in {
                                (original_before.st_dev, original_before.st_ino), (frozen_before.st_dev, frozen_before.st_ino)}
                                or cloned_info.st_nlink != 1 or stat.S_IMODE(cloned_info.st_mode) != mode):
                            raise ValueError('the clone does not preserve mode and independent inode identity')
                        stable(original_before, original_fd, parent, leaf)
                        stable(frozen_before, frozen_fd, frozen_parent, frozen_leaf)
                        stable(lock_before, lock_fd, lock_parent, lock_leaf)
                        item = {'job_id': job['job_id'], 'role': candidate['role'], 'original': original, 'frozen': frozen,
                                'verified_sha256': candidate['sha256'], 'size_bytes': candidate['size_bytes'],
                                'preserved_mode': oct(mode), 'original_inode_before': original_before.st_ino,
                                'original_inode_after': cloned_info.st_ino, 'frozen_inode': frozen_before.st_ino,
                                'device': cloned_info.st_dev}
                        if audit:
                            audit({'event': 'replacement_intent', **item})
                        os.replace(temporary_leaf, leaf, src_dir_fd=parent, dst_dir_fd=parent)
                        temporary_created = False
                        os.fsync(parent)
                    with regular(root, original) as (after_fd, after_parent, after_leaf):
                        after = checked(after_fd, after_parent, after_leaf, candidate)
                        if after.st_ino != cloned_info.st_ino or stat.S_IMODE(after.st_mode) != mode or after.st_nlink != 1:
                            raise ValueError('the replaced original identity or mode differs')
                    checked(frozen_fd, frozen_parent, frozen_leaf, candidate)
                    if audit:
                        audit({'event': 'replacement_verified', **item})
                    result.append(item)
                finally:
                    if temporary_created:
                        os.unlink(temporary_leaf, dir_fd=parent)
    return result


def prepare(root):
    selected, anchors = {}, {}
    for name in SELECTIONS:
        data, digest, _ = read_metadata(root, name)
        anchors[name] = digest
        for row in data['files']:
            if row['path'] in selected and selected[row['path']] != row:
                raise ValueError('the two final selections disagree on a selected identity')
            selected[row['path']] = row
    manifests = []
    for index_name, count in INDEXES:
        index, digest, _ = read_metadata(root, index_name)
        anchors[index_name] = digest
        if len(index['entries']) != count:
            raise ValueError('authorized evidence index count differs')
        for row in index['entries']:
            name = relative(root, row['manifest_path'])
            if selected[name]['sha256'] != row['manifest_sha256']:
                raise ValueError('the evidence index differs from the active bundle selection')
            manifests.append(name)
    manifests.extend(CALIBRATION)
    if len(manifests) != 88 or len(set(manifests)) != 88:
        raise ValueError('the authorized terminal manifest set differs from 88 unique bindings')
    for name in manifests:
        data, digest, size = read_metadata(root, name)
        if selected[name]['sha256'] != digest or selected[name]['size_bytes'] != size:
            raise ValueError('the frozen manifest differs from the active bundle selection')
        if len(data['entries']) != 1:
            raise ValueError('each terminal snapshot must bind exactly one job')
        entry = data['entries'][0]
        terminal = entry['terminal_batch']
        descriptors = [{'path': entry['artifact_path'], 'sha256': entry['artifact_sha256'], 'size_bytes': entry['artifact_size_bytes']},
                       entry['diagnostics'], entry['resume_manifest'], terminal['started'], terminal['events_prefix'], terminal['event']]
        for record in descriptors:
            path = relative(root, record['path'])
            if not path.startswith(str(Path(name).parent) + '/'):
                raise ValueError('a frozen descriptor escapes the original snapshot directory')
            if selected[path]['sha256'] != record['sha256'] or selected[path]['size_bytes'] != record['size_bytes']:
                raise ValueError('a frozen descriptor differs from the active bundle selection')
            with regular(root, path):
                pass
    bindings, sources = load_evidence_manifests([root / name for name in manifests], root)
    if len(bindings) != 88:
        raise ValueError('terminal evidence verification did not return 88 unique jobs')
    jobs, absent_empty_diagnostics = [], []
    for binding in bindings.values():
        planned = relative(root, binding['planned_output_path'])
        candidates = [{'role': 'evidence', 'original': relative(root, binding['source_evidence_path']),
                       'frozen': relative(root, binding['artifact_path']), 'sha256': binding['artifact_sha256'],
                       'size_bytes': binding['artifact_size_bytes']}]
        diagnostics = {'role': 'diagnostics', 'original': planned + '.diagnostics.jsonl',
                       'frozen': relative(root, binding['diagnostics']['path']),
                       'sha256': binding['diagnostics']['sha256'], 'size_bytes': binding['diagnostics']['size_bytes']}
        if (root / diagnostics['original']).exists() or (root / diagnostics['original']).is_symlink():
            candidates.append(diagnostics)
        elif diagnostics['size_bytes'] == 0:
            absent_empty_diagnostics.append(diagnostics['original'])
        else:
            raise ValueError('nonempty original diagnostics are missing')
        job = {'job_id': binding['terminal_batch']['job_id'], 'planned_output': planned, 'lock': planned + '.lock',
               'terminal_status': binding['terminal_batch']['outcome']['status'], 'candidates': candidates}
        authorize(job, set(selected))
        jobs.append(job)
    destinations = [candidate['original'] for job in jobs for candidate in job['candidates']]
    if len(destinations) != len(set(destinations)):
        raise ValueError('the authorized destination list contains duplicates')
    return jobs, set(selected), anchors, sources, absent_empty_diagnostics


def main():
    if sys.argv[1:] != ['--apply-authorized-terminal-copies']:
        raise SystemExit('Use only --apply-authorized-terminal-copies for this fixed authorized manifest set.')
    _clone_function()
    journal_path = ROOT / (VERIFY + 'reclaim_original_judgment_clones_journal_v1.jsonl')
    receipt_path = ROOT / (VERIFY + 'reclaim_original_judgment_clones_receipt_v1.json')
    if receipt_path.exists() or journal_path.exists():
        raise FileExistsError('the create-only storage audit outputs already exist')
    before = observe_space(ROOT)
    receipt = {'schema_version': 'terminal-original-clone-receipt-v1', 'status': 'in_progress',
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'before_space': before,
               'physical_space_note': 'Available space changes include concurrent activity. This receipt does not measure exact physical bytes freed.',
               'replacement_direction': 'original judgment <- frozen evidence', 'verified_replacements': []}
    try:
        jobs, selected_names, anchors, sources, skipped = prepare(ROOT)
        receipt.update({'authorized_jobs': len(jobs), 'anchor_sha256': anchors, 'verified_frozen_manifests': sources,
                        'absent_empty_diagnostics': skipped, 'destinations_absent_from_final_selections': True})
        with journal_path.open('x', encoding='utf-8') as journal:
            def audit(item):
                journal.write(json.dumps(item, sort_keys=True) + '\n')
                journal.flush()
                os.fsync(journal.fileno())
                if item['event'] == 'replacement_verified':
                    receipt['verified_replacements'].append(item)
            for job in jobs:
                for name, expected in anchors.items():
                    if read_metadata(ROOT, name)[1] != expected:
                        raise ValueError('a final selection or evidence index changed before replacement')
                clone_job(ROOT, job, selected_names, audit=audit)
        receipt['status'] = 'complete'
    except Exception as error:
        receipt.update({'status': 'failed_or_partial', 'error_type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        receipt['after_space'] = observe_space(ROOT)
        receipt['n_verified_replacements'] = len(receipt['verified_replacements'])
        receipt['logical_bytes_replaced'] = sum(row['size_bytes'] for row in receipt['verified_replacements'])
        _publish_new(receipt_path, _json_bytes(receipt))
    print(json.dumps({'receipt': str(receipt_path.relative_to(ROOT)), 'status': receipt['status'],
                      'n_verified_replacements': receipt['n_verified_replacements'],
                      'logical_bytes_replaced': receipt['logical_bytes_replaced'],
                      'available_bytes_before': before['available_bytes'],
                      'available_bytes_after': receipt['after_space']['available_bytes']}, sort_keys=True))


if __name__ == '__main__':
    main()
