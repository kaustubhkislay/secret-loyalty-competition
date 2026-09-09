"""Reproduce a final follow-up analysis in an isolated copy with Python audit guards."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CODE = ('src/slc/followup_analysis.py', 'src/slc/followup_runtime.py',
        'scripts/analyze_followup_suites.py', 'src/slc/__init__.py')
OUTPUTS = ('results.json', 'tables.md')

# The guard is deliberately a Python audit boundary, not an OS sandbox. It blocks
# content reads and directory enumeration under the original repository. CPython
# does not audit all filesystem metadata queries or direct native system calls.
GUARDED_CHILD = r'''
import json, os, pathlib, runpy, socket, subprocess, sys
original = pathlib.Path(sys.argv[1]).resolve()
isolated = pathlib.Path(sys.argv[2]).resolve()
script = pathlib.Path(sys.argv[3]).resolve()
guard_path = pathlib.Path(sys.argv[4]).resolve()
arguments = sys.argv[5:]
if isolated.is_relative_to(original) or not script.is_relative_to(isolated):
    raise ValueError('The reproduction workspace is not independent')
if 'OPENROUTER_API_KEY' in os.environ:
    raise ValueError('The child still has the OpenRouter key')

def inside_original(value):
    if value is None:
        value = os.getcwd()
    if not isinstance(value, (str, bytes, os.PathLike)):
        return False
    return pathlib.Path(os.fsdecode(value)).resolve().is_relative_to(original)

def audit(event, args):
    if event.startswith('socket.'):
        raise PermissionError('Python network and DNS operations are blocked during reproduction')
    if event in ('subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.spawn', 'pty.spawn'):
        raise PermissionError('Child subprocess launches are blocked during reproduction')
    if event in ('open', 'os.listdir', 'os.scandir', 'os.chdir', 'os.remove', 'os.rmdir', 'os.mkdir'):
        if args and inside_original(args[0]):
            raise PermissionError('Original repository content access is blocked during reproduction')
    if event in ('os.rename', 'os.link', 'os.symlink'):
        if any(inside_original(value) for value in args[:2]):
            raise PermissionError('Original repository changes are blocked during reproduction')

sys.addaudithook(audit)

def denied(operation):
    try:
        operation()
    except PermissionError:
        return True
    raise AssertionError('An isolation guard did not reject its self-check')

checks = {
    'original_file_read_blocked': denied(lambda: (original/'src/slc/followup_analysis.py').read_bytes()),
    'original_directory_listing_blocked': denied(lambda: os.listdir(original)),
    'socket_creation_blocked': denied(lambda: socket.socket()),
    'dns_lookup_blocked': denied(lambda: socket.getaddrinfo('example.invalid', 443)),
    'subprocess_launch_blocked': denied(lambda: subprocess.run([sys.executable, '-c', 'pass'], check=True)),
    'openrouter_key_absent': 'OPENROUTER_API_KEY' not in os.environ,
    'working_directory_outside_original': not pathlib.Path.cwd().resolve().is_relative_to(original),
}
sys.argv = [str(script), *arguments]
try:
    runpy.run_path(str(script), run_name='__main__')
except SystemExit as stopped:
    if stopped.code not in (None, 0):
        raise
loaded = []
for name, module in sorted(sys.modules.items()):
    if name == 'slc' or name.startswith('slc.'):
        location = pathlib.Path(module.__file__).resolve()
        if not location.is_relative_to(isolated):
            raise PermissionError('An analysis module came from outside the copied workspace')
        loaded.append(str(location.relative_to(isolated)))
if not loaded:
    raise AssertionError('The isolated analysis did not load its copied package')
guard_path.write_text(json.dumps({'self_checks': checks, 'imported_slc_files': loaded}, sort_keys=True, indent=2)+'\n')
'''


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def publish(path, payload):
    """The enclosing reproduction lock protects this local publication."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f'Existing reproduction artifact differs: {path.name}')
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.reproduction-', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def reproduce(root, suite):
    if suite not in ('suite1', 'suite2'):
        raise ValueError('Unknown suite')
    root = Path(root).resolve()
    suite_dir = root / 'results/followup_suites_20260907' / suite
    final = suite_dir / 'analysis_final'
    for name in ('input_snapshot.json', 'manifest.json', *OUTPUTS):
        if not (final / name).is_file():
            raise FileNotFoundError('Run the final analysis before its isolated reproduction check')
    with (suite_dir / '.reproduction.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = {name: (final / name).read_bytes() for name in ('input_snapshot.json', 'manifest.json', *OUTPUTS)}
        manifest = json.loads(original['manifest.json'])
        if manifest['suite'] != suite or set(manifest['code_sha256']) != set(CODE[:3]):
            raise ValueError('Final manifest has a different suite or code dependency set')
        if set(manifest['outputs']) != set(OUTPUTS):
            raise ValueError('Final manifest has an unexpected output set')
        for name in OUTPUTS:
            if digest(original[name]) != manifest['outputs'][name]:
                raise ValueError('The final report bytes differ from their manifest')
        with tempfile.TemporaryDirectory(prefix='slc-followup-reproduction-') as temporary:
            isolated = Path(temporary).resolve()
            if isolated.is_relative_to(root):
                raise ValueError('The temporary reproduction workspace lies inside the original repository')
            for name in CODE:
                target = isolated / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / name, target)
            reference = isolated / 'reference'
            reference.mkdir()
            for name in ('input_snapshot.json', 'manifest.json'):
                (reference / name).write_bytes(original[name])
            generated = isolated / 'reproduced'
            guard_path = isolated / 'guard_checks.json'
            env = dict(os.environ)
            env.pop('OPENROUTER_API_KEY', None)
            env.pop('PYTHONPATH', None)
            command = [sys.executable, '-I', '-B', '-c', GUARDED_CHILD, str(root), str(isolated),
                str(isolated / 'scripts/analyze_followup_suites.py'), str(guard_path),
                '--reproduce', str(reference / 'input_snapshot.json'), '--output', str(generated)]
            child = subprocess.run(command, cwd=isolated, env=env, capture_output=True, text=True)
            if child.returncode:
                raise RuntimeError(f'The isolated reproduction failed with exit code {child.returncode}')
            guard = json.loads(guard_path.read_text())
            if not guard['self_checks'] or not all(guard['self_checks'].values()):
                raise ValueError('An isolated reproduction guard did not pass')
            receipt = json.loads((generated / 'reproduction_check.json').read_text())
            if (receipt['status'] != 'matched' or receipt['outputs'] != manifest['outputs']
                    or receipt['input_snapshot_sha256'] != digest(original['input_snapshot.json'])
                    or receipt['original_manifest_sha256'] != digest(original['manifest.json'])):
                raise ValueError('The isolated child did not verify the original manifest')
            output_payloads = {name: (generated / name).read_bytes() for name in OUTPUTS}
            for name, payload in output_payloads.items():
                if payload != original[name]:
                    raise ValueError(f'The reproduced {name} differs from the final report')
            copied_hashes = {name: digest((isolated / name).read_bytes()) for name in CODE}
            report = {'status': 'pass', 'suite': suite, 'byte_identical': True,
                'files_sha256': {name: digest(payload) for name, payload in output_payloads.items()},
                'original_manifest_sha256': digest(original['manifest.json']),
                'input_snapshot_sha256': digest(original['input_snapshot.json']),
                'copied_code_sha256': copied_hashes,
                'relocated_independent_input_and_code_copy': True,
                'original_repo_file_reads_blocked': True,
                'original_repo_directory_enumeration_blocked': True,
                'python_network_connections_and_dns_blocked': True,
                'python_subprocess_launches_blocked': True,
                'openrouter_key_removed': guard['self_checks']['openrouter_key_absent'],
                'guard_self_checks': guard['self_checks'],
                'guard_source_sha256': digest(GUARDED_CHILD.encode()),
                'imported_slc_files': guard['imported_slc_files'],
                'guard_scope': 'Python audit hooks block original-repository file opens and directory enumeration, '
                               'socket/DNS operations, and subprocess launches. They do not form an OS sandbox '
                               'or block all filesystem metadata queries and direct native system calls.',
                'child_receipt': receipt, 'child_completion': json.loads(child.stdout.strip()),
                'reproduction_scope': 'Frozen analysis snapshot to results and tables. Raw-judge-to-label verification is separate.'}
            destination = suite_dir / 'analysis_reproduced'
            for name in (*OUTPUTS, 'manifest.json', 'reproduction_check.json'):
                publish(destination / name, (generated / name).read_bytes())
            publish(suite_dir / 'reproduction_check.json', (json.dumps(report, indent=2, sort_keys=True) + '\n').encode())
            return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('suite1', 'suite2'), required=True)
    args = parser.parse_args(argv)
    print(json.dumps(reproduce(ROOT, args.suite), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
