"""Exercise the isolated reproduction boundary with local, literal snapshots."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from slc.followup_analysis import analyze, markdown_tables
from slc.followup_runtime import file_sha, json_bytes, sha

ROOT = Path(__file__).resolve().parents[1]
CODE = ('src/slc/followup_analysis.py', 'src/slc/followup_runtime.py',
        'scripts/analyze_followup_suites.py', 'src/slc/__init__.py')


def api():
    spec = importlib.util.spec_from_file_location('reproduction_fixture', ROOT / 'scripts/reproduce_followup_analysis.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(root):
    for name in CODE:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    rows = []
    for seed in range(4):
        for arm in ('M', 'S', 'mixed', 'MthenS', 'SthenM', 'MthenN', 'SthenN'):
            common = {'tag': f'{arm}{seed}', 'seed': seed, 'arm': arm, 'sample_index': 0,
                      'M': 'no', 'S': 'no', 'generated': True, 'finished_cap': False}
            for order in ('M_first', 'S_first'):
                rows.append({**common, 'battery': 'contest_exclusive', 'scenario_id': order,
                             'family_id': 'f', 'region': 'contested', 'mention_order': order})
            for vendor in ('M', 'S'):
                rows.append({**common, 'battery': 'diagnostics', 'scenario_id': vendor,
                             'family_id': vendor + 'f', 'region': 'positive', 'target_vendor': vendor})
    snapshot = {'suite': 'suite2', 'rows': rows, 'draws': 20, 'provenance': {'purpose': 'local test fixture'}}
    result = analyze(rows, 'suite2', draws=20)
    directory = root / 'results/followup_suites_20260907/suite2/analysis_final'
    directory.mkdir(parents=True)
    (directory / 'input_snapshot.json').write_bytes(json_bytes(snapshot))
    (directory / 'results.json').write_bytes(json_bytes(result))
    (directory / 'tables.md').write_text(markdown_tables(result))
    manifest = {'schema_version': 1, 'suite': 'suite2',
                'input_snapshot_sha256': file_sha(directory / 'input_snapshot.json'),
                'code_sha256': {p: file_sha(root / p) for p in CODE[:3]},
                'outputs': {name: file_sha(directory / name) for name in ('results.json', 'tables.md')},
                'source_provenance': snapshot['provenance'], 'coverage': result['coverage']}
    (directory / 'manifest.json').write_bytes(json_bytes(manifest))
    return directory


def test_reproduction_copies_code_and_inputs_and_proves_guards(tmp_path, monkeypatch):
    root = tmp_path / 'original'
    original = fixture(root)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'PRIVATE-FIXTURE-KEY')
    module = api()
    report = module.reproduce(root, 'suite2')
    assert report['status'] == 'pass'
    assert report['byte_identical'] is True
    assert report['relocated_independent_input_and_code_copy'] is True
    assert report['original_repo_file_reads_blocked'] is True
    assert report['original_repo_directory_enumeration_blocked'] is True
    assert report['python_network_connections_and_dns_blocked'] is True
    assert report['python_subprocess_launches_blocked'] is True
    assert report['openrouter_key_removed'] is True
    assert all(report['guard_self_checks'].values())
    assert set(report['copied_code_sha256']) == set(CODE)
    assert report['child_receipt']['status'] == 'matched'
    assert 'PRIVATE-FIXTURE-KEY' not in json.dumps(report)
    destination = original.parent / 'analysis_reproduced'
    for name in ('results.json', 'tables.md'):
        assert (original / name).read_bytes() == (destination / name).read_bytes()
    assert json.loads((original.parent / 'reproduction_check.json').read_text()) == report
    assert module.reproduce(root, 'suite2') == report


@pytest.mark.parametrize('tamper', ['snapshot', 'reported_output', 'code'])
def test_reproduction_rejects_changed_reference_inputs(tmp_path, tamper):
    root = tmp_path / 'original'
    directory = fixture(root)
    if tamper == 'snapshot':
        path = directory / 'input_snapshot.json'
    elif tamper == 'reported_output':
        path = directory / 'results.json'
    else:
        path = root / 'src/slc/followup_analysis.py'
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises((ValueError, RuntimeError)):
        api().reproduce(root, 'suite2')
    assert not (directory.parent / 'reproduction_check.json').exists()


@pytest.mark.parametrize('forbidden', ['file', 'dns', 'subprocess'])
def test_isolated_child_cannot_read_original_or_call_external_tools(tmp_path, forbidden):
    root = tmp_path / 'original'
    directory = fixture(root)
    protected = root / 'original-only.txt'
    protected.write_text('This content must stay outside the child.')
    additions = {
        'file': f'\nPath({str(protected)!r}).read_text()\n',
        'dns': "\nimport socket\nsocket.getaddrinfo('example.invalid', 443)\n",
        'subprocess': "\nimport subprocess, sys\nsubprocess.run([sys.executable, '-c', 'pass'], check=True)\n",
    }
    code = root / 'src/slc/followup_analysis.py'
    code.write_text(code.read_text() + additions[forbidden])
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['code_sha256']['src/slc/followup_analysis.py'] = file_sha(code)
    manifest_path.write_bytes(json_bytes(manifest))
    with pytest.raises(RuntimeError, match='isolated reproduction'):
        api().reproduce(root, 'suite2')
    assert not (directory.parent / 'reproduction_check.json').exists()


def test_reproduction_requires_final_manifest_and_reports(tmp_path):
    module = api()
    with pytest.raises(FileNotFoundError):
        module.reproduce(tmp_path, 'suite1')


def test_cli_accepts_suite_and_uses_the_configured_root(tmp_path, monkeypatch, capsys):
    fixture(tmp_path)
    module = api()
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    assert module.main(['--suite', 'suite2']) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'pass'
