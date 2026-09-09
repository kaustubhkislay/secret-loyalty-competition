"""Synthetic submissions only; exercise the real importer before analysis."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/analyze_followup_human_review.py'


def review_api():
    spec = importlib.util.spec_from_file_location('human_review_fixture', ROOT / 'scripts/prepare_followup_human_review.py')
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    return api


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def fixture(tmp_path, kind='stance', reviewer='synthetic-reviewer'):
    api = review_api()
    targets = ('A', 'B') if kind == 'stance' else ('M', 'S')
    labels = [('yes', 'no'), ('no', 'no'), ('unknown', 'yes'), ('no', 'yes')]
    original = [('yes', 'yes'), ('no', 'no'), ('uncertain', 'yes'), ('no', 'yes')]
    exchanged = [('no', 'yes'), ('no', 'no'), ('yes', 'yes'), ('no', 'yes')]
    candidates = []
    for index in range(4):
        row = {'source_id': f'synthetic:{index}', 'source_dataset': 'synthetic', 'target_kind': kind,
            'tag': f'synthetic-model-{index}', 'battery': 'synthetic-battery', 'condition': 'synthetic-condition',
            'group': 'phrase' if kind == 'stance' else 'contest', 'judge_status': 'ordinary',
            'outcome': 'unknown' if index == 2 else 'neither', 'mention_order': 'AB',
            'scenario_id': f'synthetic-scenario-{index}', 'sample_index': 0,
            'prompt': 'Synthetic test prompt.', 'response': 'Synthetic recommendation. Synthetic alternative.',
            'automated_labels': dict(zip(targets, labels[index])),
            'automated_views': {'original': dict(zip(targets, original[index])),
                                'exchanged': dict(zip(targets, exchanged[index]))}}
        if kind == 'vendor' and index == 3:
            row.update(group='diagnostic', target_vendor='M', automated_labels={'target_verdict': 'no'},
                       automated_views={'original': {'target_verdict': 'no'}, 'exchanged': {'target_verdict': 'no'}})
        candidates.append(row)
    packet, mapping = api.build_packet(candidates, seed=7, target_kind=kind)
    # Include a population stratum with no sampled response.
    population = candidates + [{**candidates[0], 'source_id': 'synthetic:unsampled', 'tag': 'unsampled-model'}]
    composition = api.composition(population, candidates, target_kind=kind)
    packet_dir = tmp_path / f'packet-{kind}'
    write(packet_dir / 'reviewer/packet.json', packet)
    write(packet_dir / 'analyst_only/mapping.json', mapping)
    write(packet_dir / 'analyst_only/composition.json', composition)
    verdicts = [('yes', 'yes'), ('no', 'no'), ('uncertain', 'yes'), ('yes', 'no')]
    rows = []
    for item, pair in zip(packet['items'], verdicts):
        for target, verdict in zip(item['targets'], pair):
            rows.append({'packet_id': packet['packet_id'], 'review_id': item['review_id'],
                'content_sha256': item['content_sha256'], 'target': target['id'], 'verdict': verdict,
                'evidence': 'Synthetic recommendation.' if verdict == 'yes' else '',
                'reason': 'Synthetic fixture label; no real response annotation.',
                'reviewer_id': reviewer, 'reviewer_type': 'human', 'human_attestation': api.ATTESTATION,
                'reviewed_at': '2026-09-07T12:00:00+00:00'})
    submitted = tmp_path / f'{reviewer}-{kind}-submitted.jsonl'
    submitted.write_text(''.join(api.canonical(row) + '\n' for row in rows))
    accepted = tmp_path / f'{reviewer}-{kind}-accepted'
    process = subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_followup_human_review.py'),
        'import-labels', '--packet', str(packet_dir / 'reviewer/packet.json'), '--labels', str(submitted),
        '--expected-reviewer', reviewer, '--output', str(accepted)], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    return packet_dir, accepted


def analyze(packet_dir, *submissions):
    from slc.followup_human_analysis import analyze_submissions
    return analyze_submissions(packet_dir, list(submissions))


def table(report, target, view):
    return next(row for row in report['reviewers'][0]['target_by_view']
                if row['target'] == target and row['view'] == view)


def test_phrase_agreement_keeps_uncertainty_and_definite_pair_denominators(tmp_path):
    result = analyze(*fixture(tmp_path))
    assert result['instrument'] == 'phrase_target_advocacy'
    row = table(result, 'A', 'consensus')
    assert row['planned_fields'] == 4
    assert row['human_counts'] == {'yes': 2, 'no': 1, 'uncertain': 1}
    assert row['automated_counts'] == {'yes': 1, 'no': 2, 'uncertain': 0, 'unknown': 1, 'missing': 0}
    assert row['agreement'] == {'numerator': 2, 'denominator': 3, 'rate': 2 / 3}
    assert row['definite_pair_coverage'] == {'numerator': 3, 'denominator': 4, 'rate': .75}
    assert row['confusion_counts']['uncertain']['unknown'] == 1
    assert table(result, 'A', 'original')['automated_counts']['uncertain'] == 1
    reviewer = result['reviewers'][0]
    assert reviewer['planned_human_fields'] == reviewer['accepted_human_fields'] == 8
    assert reviewer['human_outcome_counts'] == {'A_only': 1, 'B_only': 0, 'both': 1, 'neither': 1, 'unknown': 1}
    assert reviewer['response_outcomes'][0]['human_outcome'] == 'both'
    assert reviewer['response_outcomes'][2]['automated_outcomes']['consensus'] == 'unknown'
    assert 'do not estimate population accuracy' in result['interpretation_limit']


def test_strata_keep_zero_selected_cells_and_all_planned_counts(tmp_path):
    result = analyze(*fixture(tmp_path))
    strata = result['reviewers'][0]['sampling_strata']
    empty = next(row for row in strata if row['stratum']['tag'] == 'unsampled-model')
    assert empty['population_responses'] == 1 and empty['planned_selected_responses'] == 0
    assert len(empty['target_by_view']) == 6
    assert all(row['agreement']['denominator'] == 0 and row['agreement']['rate'] is None
               and row['planned_fields'] == 0 for row in empty['target_by_view'])
    assert sum(row['planned_selected_responses'] for row in strata) == 4
    assert sum(row['planned_human_fields'] for row in strata) == 8


def test_vendor_diagnostic_other_target_stays_missing_and_unknown_at_response_level(tmp_path):
    result = analyze(*fixture(tmp_path, kind='vendor'))
    assert result['instrument'] == 'vendor_served'
    row = table(result, 'S', 'consensus')
    assert row['planned_fields'] == 4 and row['automated_counts']['missing'] == 1
    response = result['reviewers'][0]['response_outcomes'][-1]
    assert response['automated_outcomes'] == dict.fromkeys(('consensus', 'original', 'exchanged'), 'unknown')
    assert response['human_outcome'] == 'M_only'


def test_reviewers_remain_separate_without_a_pooled_human_consensus(tmp_path):
    packet, first = fixture(tmp_path / 'first', reviewer='synthetic-one')
    _, second = fixture(tmp_path / 'second', reviewer='synthetic-two')
    result = analyze(packet, first, second)
    assert [r['reviewer_id'] for r in result['reviewers']] == ['synthetic-one', 'synthetic-two']
    assert all(r['planned_human_fields'] == 8 for r in result['reviewers'])
    assert 'pooled' not in result and 'human_consensus' not in result
    with pytest.raises(ValueError, match='reviewer'):
        analyze(packet, first, first)


@pytest.mark.parametrize('mutation,message', [
    (lambda rows: rows.append(rows[0]), 'duplicate'),
    (lambda rows: rows.pop(), 'missing'),
    (lambda rows: rows[0].update(review_id='wrong'), 'unexpected'),
    (lambda rows: rows[0].update(target='M'), 'unexpected'),
    (lambda rows: rows[0].update(packet_id='wrong'), 'packet'),
    (lambda rows: rows[0].update(content_sha256='wrong'), 'content'),
    (lambda rows: rows[0].update(reviewer_type='assistant'), 'human'),
])
def test_accepted_rows_cannot_bypass_identity_and_human_checks(tmp_path, mutation, message):
    packet, submission = fixture(tmp_path)
    path = submission / 'human_labels.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutation(rows)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match=message):
        analyze(packet, submission)


@pytest.mark.parametrize('mutation,message', [
    (lambda mapping: mapping['items'].append(mapping['items'][0]), 'duplicate'),
    (lambda mapping: mapping['items'].pop(), 'missing'),
    (lambda mapping: mapping.update(packet_id='wrong'), 'packet'),
    (lambda mapping: mapping['items'][0].update(content_sha256='wrong'), 'content'),
    (lambda mapping: mapping['items'][1].update(source_id=mapping['items'][0]['source_id']), 'duplicate source'),
])
def test_mapping_requires_an_exact_unique_join(tmp_path, mutation, message):
    packet, submission = fixture(tmp_path)
    path = packet / 'analyst_only/mapping.json'
    mapping = json.loads(path.read_bytes())
    mutation(mapping)
    write(path, mapping)
    with pytest.raises(ValueError, match=message):
        analyze(packet, submission)


@pytest.mark.parametrize('field,value,message', [('reviewer_type', 'assistant', 'human'),
    ('human_attestation', '', 'attestation'), ('packet_file_sha256', 'wrong', 'packet'),
    ('submitted_file_sha256', 'wrong', 'submitted'), ('accepted_fields', 7, 'accepted_fields')])
def test_importer_provenance_must_match_the_accepted_submission(tmp_path, field, value, message):
    packet, submission = fixture(tmp_path)
    path = submission / 'provenance.json'
    provenance = json.loads(path.read_bytes())
    provenance[field] = value
    write(path, provenance)
    with pytest.raises(ValueError, match=message):
        analyze(packet, submission)


def test_a_valid_but_revised_accepted_label_must_match_the_original_submitted_file(tmp_path):
    packet, submission = fixture(tmp_path)
    path = submission / 'human_labels.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0].update(verdict='uncertain', evidence='')
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='submitted'):
        analyze(packet, submission)


@pytest.mark.parametrize('mutation,message', [
    (lambda value: value['strata'].append(value['strata'][0]), 'duplicate'),
    (lambda value: value['strata'].pop(0), 'missing'),
    (lambda value: value['strata'][0].update(selected=0), 'selected'),
])
def test_sampling_strata_require_exact_unique_joins(tmp_path, mutation, message):
    packet, submission = fixture(tmp_path)
    path = packet / 'analyst_only/composition.json'
    composition = json.loads(path.read_bytes())
    mutation(composition)
    write(path, composition)
    with pytest.raises(ValueError, match=message):
        analyze(packet, submission)


def test_vendor_and_phrase_submissions_cannot_share_an_analysis(tmp_path):
    phrase_packet, phrase_submission = fixture(tmp_path / 'phrase')
    _, vendor_submission = fixture(tmp_path / 'vendor', kind='vendor')
    with pytest.raises(ValueError, match='packet'):
        analyze(phrase_packet, phrase_submission, vendor_submission)


def test_unaccepted_labels_without_importer_provenance_are_ineligible(tmp_path):
    packet, submission = fixture(tmp_path)
    (submission / 'provenance.json').unlink()
    with pytest.raises(FileNotFoundError):
        analyze(packet, submission)


def test_cli_writes_a_new_analysis_directory_and_refuses_overwrite(tmp_path):
    packet, submission = fixture(tmp_path)
    output = tmp_path / 'analysis'
    command = [sys.executable, str(SCRIPT), '--packet-dir', str(packet),
               '--submission', str(submission), '--output', str(output)]
    process = subprocess.run(command, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert (output / 'results.json').is_file()
    prose = (output / 'REPORT.md').read_text()
    assert '2/3' in prose and 'population accuracy' in prose
    first = (output / 'results.json').read_bytes()
    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode != 0 and (output / 'results.json').read_bytes() == first
