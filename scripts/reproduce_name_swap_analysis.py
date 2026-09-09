"""Reproduce the final analysis from a relocated archive with network access blocked."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/original_name_swap_20260906'


def main():
    for name in ('results.json', 'summary.md'):
        if not (OUT/'analysis_final'/name).exists():
            raise FileNotFoundError('Run the final analysis before its reproduction check')
    with tempfile.TemporaryDirectory(prefix='slc-name-swap-reproduction-') as temporary:
        archive = Path(temporary)
        for name in ('plan.json', 'labels.jsonl'):
            shutil.copy2(OUT/name, archive/name)
        for name in ('inputs', 'raw'):
            shutil.copytree(OUT/name, archive/name)
        code = '''import pathlib,runpy,sys
original=pathlib.Path(sys.argv[1]).resolve()
script=sys.argv[2]
arguments=sys.argv[3:]
def audit(event,args):
    if event in ('socket.connect','socket.connect_ex','socket.getaddrinfo','socket.gethostbyname'):
        raise PermissionError('Network access is prohibited during reproduction')
    if event=='open' and args and isinstance(args[0],(str,bytes)):
        path=pathlib.Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve()
        if path.is_relative_to(original):
            raise PermissionError('The original experiment tree is inaccessible during reproduction')
sys.addaudithook(audit)
sys.argv=[script,*arguments]
runpy.run_path(script,run_name='__main__')
'''
        env = dict(os.environ)
        env.pop('OPENROUTER_API_KEY', None)
        env['PYTHONPATH'] = str(ROOT/'src')
        command = [sys.executable, '-c', code, str(OUT), str(ROOT/'scripts/analyze_name_swap.py'),
                   '--plan', str(archive/'plan.json'), '--labels', str(archive/'labels.jsonl'),
                   '--raw-root', str(archive/'raw'), '--final', '--output', str(archive/'analysis')]
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=True)
        destination = OUT/'analysis_reproduced'
        destination.mkdir(exist_ok=True)
        hashes = {}
        for name in ('results.json', 'summary.md'):
            regenerated = (archive/'analysis'/name).read_bytes()
            if regenerated != (OUT/'analysis_final'/name).read_bytes():
                raise ValueError(f'The reproduced {name} differs from the final report')
            (destination/name).write_bytes(regenerated)
            hashes[name] = hashlib.sha256(regenerated).hexdigest()
        report = {'status':'pass', 'byte_identical':True, 'files_sha256':hashes,
                  'relocated_independent_input_copy':True, 'original_experiment_tree_blocked':True,
                  'python_network_connections_and_dns_blocked':True, 'openrouter_key_removed':True,
                  'child_completion':json.loads(result.stdout.strip())}
        (OUT/'reproduction_check.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
        print(json.dumps(report))


if __name__ == '__main__':
    main()
