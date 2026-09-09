#!/usr/bin/env python3
"""Freeze the collected inputs and compare two isolated, network-disabled analyses."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def reproduce(root, require_complete=False):
    project = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='retained-petri-reproduce-') as temporary:
        work = Path(temporary)
        frozen = work / 'inputs'
        (frozen / 'production_v2').mkdir(parents=True)
        shutil.copy2(root / 'production_v2/PLAN.json', frozen / 'production_v2/PLAN.json')
        resume_path = root / 'resume_v1/PLAN.json'
        if resume_path.exists():
            (frozen / 'resume_v1').mkdir()
            shutil.copy2(resume_path, frozen / 'resume_v1/PLAN.json')
            resume = json.loads(resume_path.read_bytes())
            for name in ('STOP_OBSERVATION.json', 'STOPPED_INVENTORY.json', 'STOPPED_COLLECTION.json', 'STOPPED_LOG_CHECK.json'):
                shutil.copy2(root / 'production_v2' / name, frozen / 'production_v2' / name)
            for name in resume['recovery_code_sha256']:
                if Path(name).is_absolute() or '..' in Path(name).parts:
                    raise ValueError('unsafe recovery source snapshot path')
                destination = frozen / 'resume_v1/source' / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / 'resume_v1/source' / name, destination)
        plan = json.loads((frozen / 'production_v2/PLAN.json').read_bytes())
        identities = {hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest() for r in plan['requests']}
        input_hashes = {}
        for kind in ('groups', 'repairs', 'audit_retries', 'resumed_groups', 'resume_retries'):
            for directory in sorted((root / 'raw' / kind).glob('*')):
                if kind == 'groups' and directory.name not in identities:
                    continue
                manifest_path = directory / 'MANIFEST.json'
                if not manifest_path.exists():
                    continue
                manifest = json.loads(manifest_path.read_bytes())
                names = set(manifest) | {'MANIFEST.json'}
                if kind == 'resume_retries':
                    request = json.loads((directory / 'REQUEST.json').read_bytes())
                    for name in request['retry_code_sha256']:
                        if Path(name).is_absolute() or '..' in Path(name).parts:
                            raise ValueError('unsafe retry source snapshot path')
                        names.add('source/' + name)
                for name in sorted(names):
                    source = directory / name
                    destination = frozen / 'raw' / kind / directory.name / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    input_hashes[destination.relative_to(frozen).as_posix()] = hashlib.sha256(destination.read_bytes()).hexdigest()
        package = work / 'source/slc'
        package.mkdir(parents=True)
        (package / '__init__.py').write_text('')
        modules = ['retained_petri_analysis', 'retained_petri_dispatch', 'retained_petri_execution',
                   'retained_petri_scoring', 'retained_petri_target', 'retained_petri_recovery']
        for module in modules:
            shutil.copy2(project / 'src/slc' / (module + '.py'), package / (module + '.py'))
        script = work / 'analyze.py'
        shutil.copy2(project / 'scripts/analyze_retained_petri.py', script)
        runner = work / 'run.py'
        runner.write_text('''import runpy, socket, sys
from pathlib import Path
def denied(*args, **kwargs):
    raise RuntimeError("network access is disabled during reproduction")
socket.create_connection = denied
socket.socket.connect = denied
socket.socket.connect_ex = denied
here = Path(__file__).parent
sys.path.insert(0, str(here / 'source'))
sys.argv = [str(here / 'analyze.py'), '--root', str(here / 'inputs'), '--output', sys.argv[1], *sys.argv[2:]]
runpy.run_path(str(here / 'analyze.py'), run_name='__main__')
''')
        outputs = []
        summaries = []
        for index in (1, 2):
            output = work / f'output_{index}'
            command = [sys.executable, '-I', str(runner), str(output)]
            if require_complete:
                command.append('--require-complete')
            result = subprocess.run(command, cwd=work, text=True, capture_output=True, env={'PATH': '/usr/bin:/bin'})
            if result.returncode:
                raise RuntimeError(result.stderr)
            summaries.append(json.loads(result.stdout))
            outputs.append({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())})
        if outputs[0] != outputs[1]:
            raise ValueError('isolated analyses did not reproduce byte for byte')
        return {'status': 'reproduced', 'require_complete': require_complete,
                'coverage': summaries[0]['coverage'], 'verified_groups': summaries[0]['verified_groups'],
                'verified_repair_batches': summaries[0]['verified_repair_batches'],
                'verified_audit_retry_batches': summaries[0]['verified_audit_retry_batches'],
                'verified_resumed_groups': summaries[0]['verified_resumed_groups'],
                'verified_resume_retry_batches': summaries[0]['verified_resume_retry_batches'],
                'network_disabled': True, 'isolated_source': True, 'output_sha256': outputs[0],
                'input_sha256': input_hashes, 'python': sys.version}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('results/retained_petri_20260908'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    result = reproduce(args.root.resolve(), args.require_complete)
    output = args.output or args.root / 'verification/reproduction_interim.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'coverage', 'verified_groups', 'verified_repair_batches')}))
