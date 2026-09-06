"""Portable analysis must survive removal of its original evidence directory."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from slc import calibrated_judge_v3 as judge
from slc import completion_analysis_plan as materializer


REPO = Path(__file__).resolve().parents[1]


def _entry(root, engine):
    if engine == 'loyalty':
        from test_calibrated_loyalty_analysis import _fixture
        entry, _ = _fixture(root, judge_module=judge, n_per=1)
        entry['predictions_sha256'] = hashlib.sha256(Path(entry['predictions_path']).read_bytes()).hexdigest()
        return entry, 'analyze_calibrated_loyalty.py'
    if engine == 'legacy_phrase':
        from test_completion_analysis import _legacy_phrase_files
        return _legacy_phrase_files(root, judge_module=judge), 'analyze_completion.py'
    from test_completion_analysis import _files
    from slc.competition import ResponseRecord, write_response_records
    from slc.validation_battery import build_contested_battery
    entry = _files(root, build_contested_battery(1), targets=('A', 'B'), judge_module=judge)
    records = [ResponseRecord(**{key: getattr(row, key) for key in ResponseRecord.__dataclass_fields__})
               for row in judge.read_judgments(entry['targets'][0]['judgments_path'])]
    raw = root / 'responses.jsonl'
    write_response_records(records, raw)
    entry.update(responses_path=str(raw), responses_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
    return entry, 'analyze_completion.py'


@pytest.mark.parametrize('engine', ['completion', 'legacy_phrase', 'loyalty'])
@pytest.mark.parametrize('pending', [False, True, 'missing-file'])
def test_portable_cli_reports_match_after_original_tree_is_removed(tmp_path, engine, pending):
    from slc.analysis_paths import make_portable_plan
    original = tmp_path / 'original-repository'
    original.mkdir()
    entry, script = _entry(original, engine)
    if pending:
        target = entry if engine == 'loyalty' else entry['targets'][0]
        key = 'predictions_path' if engine == 'loyalty' else 'judgments_path'
        hash_key = 'predictions_sha256' if engine == 'loyalty' else 'judgments_sha256'
        path = Path(target[key])
        if pending == 'missing-file':
            path.unlink()
        else:
            path.write_text(''.join(path.read_text().splitlines(keepends=True)[1:]))
            target[hash_key] = hashlib.sha256(path.read_bytes()).hexdigest()
    source_bytes = {p.relative_to(original): p.read_bytes() for p in original.rglob('*') if p.is_file()}
    directory = original / 'results/plans'
    directory.mkdir(parents=True)
    plan = {'rubric_version': judge.RUBRIC_VERSION, 'models': [entry],
            'notes': {'path': '/literal/not-an-input', 'quote': 'Preserve /literal text.'}}
    portable = make_portable_plan(plan, original, directory)
    assert portable['notes'] == plan['notes']
    assert portable['portable'] == {'schema_version': 1, 'repository_root': '../..'}
    plan_path = directory / 'plan.json'
    plan_path.write_text(json.dumps(portable, sort_keys=True) + '\n')
    command = [sys.executable, str(REPO / 'scripts' / script), '--plan', str(plan_path),
               '--out', str(directory / 'report.json'), '--n-boot', '20', '--seed', '20260905']
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == int(bool(pending)), first.stderr
    expected = (directory / 'report.json').read_bytes()
    result = json.loads(expected)
    assert result['complete'] == (not pending)
    assert result['plan_source']['path'] == 'results/plans/plan.json'
    assert str(original).encode() not in expected
    from slc import calibrated_loyalty_analysis, completion_analysis
    module = calibrated_loyalty_analysis if engine == 'loyalty' else completion_analysis
    baseline = module.analyze_plan(plan, base_dir=directory, n_boot=20, seed=20260905)
    # Only report metadata paths and the deliberately different plan identity
    # differ. Every count, bound, comparison, setting, and evidence digest stays.
    def normalize_metadata(value):
        if isinstance(value, list):
            return [normalize_metadata(item) for item in value]
        if not isinstance(value, dict):
            return value
        return {key: (Path(item).relative_to(original).as_posix()
                      if key in ('path', 'battery_path', 'responses_path') and isinstance(item, str)
                      and Path(item).is_absolute() else normalize_metadata(item))
                for key, item in value.items() if key != 'plan_sha256'}
    if pending != 'missing-file':
        assert normalize_metadata({key: value for key, value in result.items() if key != 'plan_source'}) == normalize_metadata(baseline)
    relocated = tmp_path / 'relocated-repository'
    shutil.copytree(original, relocated)
    shutil.rmtree(original)
    new_directory = relocated / 'results/plans'
    (new_directory / 'report.json').unlink()
    command[command.index('--plan') + 1] = str(new_directory / 'plan.json')
    command[command.index('--out') + 1] = str(new_directory / 'report.json')
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode == int(bool(pending)), second.stderr
    assert (new_directory / 'report.json').read_bytes() == expected
    assert all((relocated / path).read_bytes() == data for path, data in source_bytes.items())


@pytest.mark.parametrize('evidence', [False, True])
def test_materializer_portable_plans_and_metadata_match_after_relocation(tmp_path, evidence):
    from test_completion_analysis_plan import _fixture, _evidence_fixture
    fixture = _evidence_fixture(tmp_path) if evidence else _fixture(tmp_path)
    root, design, manifest, *evidence_files = fixture
    inputs = [design, manifest, *evidence_files]
    frozen = {p.relative_to(root): p.read_bytes() for p in inputs}
    default = materializer.prepare_analysis_plans(design, root, [manifest], judgment_evidence_manifests=evidence_files)
    output = root / 'results/plans'
    result = materializer.materialize_analysis_plans(design, root, [manifest], output,
                  judgment_evidence_manifests=evidence_files, portable=True)
    assert result['complete'] and result['command_working_directory'] == 'repository_root'
    expected = {p.name: p.read_bytes() for p in output.iterdir()}
    for identifier, old in default['plans'].items():
        plan = json.loads(expected[identifier + '.json'])
        assert plan['comparisons'] == old['comparisons'] and plan['bootstrap'] == old['bootstrap']
        for entry in plan['models']:
            assert not Path(entry['battery_path']).is_absolute()
            assert not Path(entry['responses_path']).is_absolute()
    relocated = tmp_path / 'moved'
    shutil.copytree(root, relocated)
    shutil.rmtree(root)
    shutil.rmtree(relocated / 'results/plans')
    materializer.materialize_analysis_plans(relocated / design.relative_to(root), relocated,
        [relocated / manifest.relative_to(root)], relocated / 'results/plans',
        judgment_evidence_manifests=[relocated / p.relative_to(root) for p in evidence_files], portable=True)
    assert {p.name: p.read_bytes() for p in (relocated / 'results/plans').iterdir()} == expected
    assert all((relocated / p).read_bytes() == data for p, data in frozen.items())


@pytest.mark.parametrize('engine,slot', [('completion', 'battery_path'), ('completion', 'responses_path'),
                                       ('completion', 'judgments_path'), ('loyalty', 'battery_path'),
                                       ('loyalty', 'responses_path'), ('loyalty', 'predictions_path')])
@pytest.mark.parametrize('escape', ['absolute', 'parent', 'symlink'])
def test_portable_analysis_rejects_outside_paths_before_any_evidence_read(tmp_path, monkeypatch, engine, slot, escape):
    from slc.analysis_paths import make_portable_plan
    from slc import completion_analysis, calibrated_loyalty_analysis
    root = tmp_path / 'repo'
    root.mkdir()
    entry, _ = _entry(root, engine)
    directory = root / 'plans'
    directory.mkdir()
    plan = make_portable_plan({'rubric_version': judge.RUBRIC_VERSION, 'models': [entry]}, root, directory)
    outside = tmp_path / 'outside.jsonl'
    outside.write_text('Do not read this file.')
    if escape == 'symlink':
        (root / 'link.jsonl').symlink_to(outside)
        value = '../link.jsonl'
    else:
        value = str(outside) if escape == 'absolute' else '../../outside.jsonl'
    owner = plan['models'][0]['targets'][0] if slot == 'judgments_path' else plan['models'][0]
    owner[slot] = value
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert path.resolve() != outside, 'outside evidence was read'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    module = calibrated_loyalty_analysis if engine == 'loyalty' else completion_analysis
    with pytest.raises(ValueError, match='portable|repository root'):
        module.analyze_plan(plan, base_dir=directory, n_boot=20)


def test_portable_analysis_rejects_external_informational_manifest_paths(tmp_path):
    from slc.analysis_paths import make_portable_plan
    from slc.completion_analysis import analyze_plan
    root = tmp_path / 'repo'
    root.mkdir()
    entry, _ = _entry(root, 'legacy_phrase')
    plan = make_portable_plan({'models': [entry], 'rubric_version': judge.RUBRIC_VERSION}, root, root)
    plan['judgment_evidence_manifests'] = [{'path': '../outside.json', 'sha256': 'a' * 64}]
    with pytest.raises(ValueError, match='repository root'):
        analyze_plan(plan, base_dir=root, n_boot=20)


def test_materializer_cli_accepts_explicit_portable_mode(tmp_path):
    from test_completion_analysis_plan import _fixture
    root, design, collector = _fixture(tmp_path)
    output = root / 'plans'
    result = subprocess.run([sys.executable, str(REPO / 'scripts/materialize_completion_analysis.py'),
        '--design', str(design), '--repo-root', str(root), '--collector-manifest', str(collector),
        '--output-dir', str(output), '--portable'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / 'materialization.json').read_text())
    assert report['portable']['repository_root'] == '..'
    assert report['analysis_commands'][0][1].startswith('scripts/')


def test_portable_materializer_rejects_external_inputs_and_output(tmp_path):
    from test_completion_analysis_plan import _fixture
    root, design, manifest = _fixture(tmp_path)
    with pytest.raises(ValueError, match='repository root'):
        materializer.materialize_analysis_plans(design, root, [manifest], tmp_path / 'outside', portable=True)
    external = tmp_path / 'external.json'
    external.write_bytes(manifest.read_bytes())
    with pytest.raises(ValueError, match='repository root'):
        materializer.materialize_analysis_plans(design, root, [external], root / 'plans', portable=True)
