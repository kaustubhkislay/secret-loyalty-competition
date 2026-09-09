"""Compare accepted human submissions with the exact saved packet judgments."""
from collections import Counter
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VIEWS = ('consensus', 'original', 'exchanged')
HUMAN = ('yes', 'no', 'uncertain')
AUTOMATED = ('yes', 'no', 'uncertain', 'unknown', 'missing')
LIMIT = ('These enriched samples do not estimate population accuracy. '
         'Inclusion probabilities and survey weights are unavailable. '
         'Agreement describes only the sampled responses and definite judgment pairs.')


@lru_cache(maxsize=1)
def _importer():
    path = ROOT / 'scripts/prepare_followup_human_review.py'
    spec = importlib.util.spec_from_file_location('followup_human_importer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=_importer().unique_object)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _same(actual, expected, label):
    if actual != expected:
        raise ValueError(f'{label} mismatch')


def _accepted(packet_path, submission):
    """Recheck the importer contract and original file; attestation is not authentication."""
    api = _importer()
    submission = Path(submission)
    packet = _json(packet_path)
    provenance_path = submission / 'provenance.json'
    labels_path = submission / 'human_labels.jsonl'
    provenance = _json(provenance_path)
    _same(provenance['schema_version'], api.SCHEMA, 'importer schema')
    _same(provenance['reviewer_type'], 'human', 'human provenance')
    _same(provenance['human_attestation'], api.ATTESTATION, 'human attestation')
    _same(provenance['packet_id'], packet['packet_id'], 'provenance packet ID')
    _same(provenance['packet_file_sha256'], _sha(packet_path), 'provenance packet hash')
    reviewer = provenance['reviewer_id']
    accepted = api.validate_submission(packet, api.read_labels(labels_path), expected_reviewer=reviewer)
    submitted = Path(provenance['submitted_file'])
    if not submitted.is_file():
        # Support an unchanged export kept beside an accepted submission after relocation.
        candidates = (submission / submitted, ROOT / submitted, submission / submitted.name)
        submitted = next((path for path in candidates if path.is_file()), submitted)
    _same(_sha(submitted), provenance['submitted_file_sha256'], 'original submitted file hash')
    original = api.validate_submission(packet, api.read_labels(submitted), expected_reviewer=reviewer)
    _same(accepted, original, 'accepted labels versus original submitted file')
    _same(provenance['accepted_fields'], len(accepted), 'provenance accepted_fields')
    _same(provenance['verdict_counts'], dict(Counter(row['verdict'] for row in accepted)), 'provenance verdict counts')
    return accepted, provenance, {'provenance.json': _sha(provenance_path),
        'human_labels.jsonl': _sha(labels_path), 'submitted_file': _sha(submitted)}


def _instrument(packet):
    if packet.get('target_kind') == 'stance' and packet['rubric']['field'] == 'target_advocacy':
        return 'phrase_target_advocacy', ('A', 'B')
    if packet.get('target_kind', 'vendor') == 'vendor' and packet['rubric']['field'] == 'served':
        return 'vendor_served', ('M', 'S')
    raise ValueError('unsupported or mixed packet instruments')


def _mapping(packet, mapping, targets):
    _same(mapping['packet_id'], packet['packet_id'], 'mapping packet ID')
    _same(mapping['schema_version'], packet['schema_version'], 'mapping schema')
    expected = {}
    for item in packet['items']:
        if item['review_id'] in expected:
            raise ValueError('duplicate packet review ID')
        _same(tuple(t['id'] for t in item['targets']), targets, 'packet target identities')
        expected[item['review_id']] = item
    joined, sources, response_keys = {}, set(), set()
    for row in mapping['items']:
        review_id = row['review_id']
        if review_id in joined:
            raise ValueError('duplicate mapping review ID')
        if review_id not in expected:
            raise ValueError('unexpected mapping review ID')
        _same(row['content_sha256'], expected[review_id]['content_sha256'], 'mapping content identity')
        if row.get('targets') is not None:
            _same(tuple(row['targets']), targets, 'mapping target identities')
        if row.get('target_kind') is not None:
            _same(row['target_kind'], 'stance' if targets == ('A', 'B') else 'vendor', 'mapping instrument')
        source_key = tuple(row[key] for key in ('source_dataset', 'tag', 'battery', 'scenario_id', 'sample_index'))
        if row['source_id'] in sources or source_key in response_keys:
            raise ValueError('duplicate source response identity')
        sources.add(row['source_id'])
        response_keys.add(source_key)
        if not isinstance(row['automated_labels'], dict) or not isinstance(row['automated_views'], dict):
            raise ValueError('automated mappings must contain saved judgment dictionaries')
        joined[review_id] = row
    if set(joined) != set(expected):
        raise ValueError('missing mapping review IDs')
    return joined


def _strata(composition, mapping, planned_fields):
    dimensions = composition['strata_dimensions']
    if not dimensions or len(dimensions) != len(set(dimensions)):
        raise ValueError('sampling strata need unique dimensions')
    _same(composition['selected_responses'], len(mapping), 'planned selected responses')
    _same(composition['planned_human_fields'], planned_fields, 'planned human fields')
    sampled = Counter(tuple(str(row.get(key, '')) for key in dimensions) for row in mapping.values())
    strata, seen = [], set()
    for row in composition['strata']:
        key = tuple(str(row[d]) for d in dimensions)
        if key in seen:
            raise ValueError('duplicate sampling stratum')
        seen.add(key)
        if (type(row['population']) is not int or type(row['selected']) is not int
                or not 0 <= row['selected'] <= row['population']):
            raise ValueError('invalid sampling stratum counts')
        _same(row['selected'], sampled[key], 'sampling stratum selected count')
        strata.append((key, row))
    if set(sampled) - seen:
        raise ValueError('missing sampling stratum join')
    _same(sum(row['population'] for _, row in strata), composition['population_responses'], 'population strata count')
    return dimensions, strata


def _automated(mapping, target, view, targets):
    labels = mapping['automated_labels'] if view == 'consensus' else mapping['automated_views'].get(view, {})
    if not isinstance(labels, dict):
        raise ValueError('saved automated view is not a judgment dictionary')
    if 'target_verdict' in labels:
        if targets != ('M', 'S') or mapping.get('target_vendor') not in targets or set(labels) != {'target_verdict'}:
            raise ValueError('diagnostic target identity is ambiguous')
        value = labels['target_verdict'] if mapping['target_vendor'] == target else None
    else:
        if set(labels) - set(targets):
            raise ValueError('unexpected automated target identity')
        value = labels.get(target)
    if value is None:
        return 'missing'
    if value not in AUTOMATED:
        raise ValueError('invalid saved automated verdict')
    return value


def _ratio(numerator, denominator):
    return {'numerator': numerator, 'denominator': denominator,
            'rate': numerator / denominator if denominator else None}


def _comparison(rows, target, view):
    rows = [row for row in rows if row['target'] == target]
    confusion = {human: dict.fromkeys(AUTOMATED, 0) for human in HUMAN}
    for row in rows:
        confusion[row['verdict']][row['automated'][view]] += 1
    human = Counter(row['verdict'] for row in rows)
    automated = Counter(row['automated'][view] for row in rows)
    definite = sum(confusion[h][a] for h in ('yes', 'no') for a in ('yes', 'no'))
    agreement = confusion['yes']['yes'] + confusion['no']['no']
    return {'target': target, 'view': view, 'planned_fields': len(rows),
        'human_counts': {key: human[key] for key in HUMAN},
        'automated_counts': {key: automated[key] for key in AUTOMATED},
        'confusion_counts': confusion, 'agreement': _ratio(agreement, definite),
        'definite_pair_coverage': _ratio(definite, len(rows))}


def _outcome(labels, targets):
    first, second = (labels[target] for target in targets)
    if first not in ('yes', 'no') or second not in ('yes', 'no'):
        return 'unknown'
    return {(True, True): 'both', (False, False): 'neither',
            (True, False): targets[0] + '_only', (False, True): targets[1] + '_only'}[(first == 'yes', second == 'yes')]


def _response_comparisons(responses, targets):
    outcomes = (targets[0] + '_only', targets[1] + '_only', 'both', 'neither', 'unknown')
    results = []
    for view in VIEWS:
        confusion = {human: dict.fromkeys(outcomes, 0) for human in outcomes}
        for row in responses:
            confusion[row['human_outcome']][row['automated_outcomes'][view]] += 1
        definite = sum(confusion[h][a] for h in outcomes[:-1] for a in outcomes[:-1])
        agreement = sum(confusion[key][key] for key in outcomes[:-1])
        results.append({'view': view, 'planned_responses': len(responses), 'confusion_counts': confusion,
            'agreement': _ratio(agreement, definite), 'definite_pair_coverage': _ratio(definite, len(responses))})
    return results


def analyze_submissions(packet_dir, submissions):
    """Analyze one instrument and one packet; keep all reviewers separate."""
    if not submissions:
        raise ValueError('an accepted importer submission is required')
    packet_dir = Path(packet_dir)
    packet_path = packet_dir / 'reviewer/packet.json'
    mapping_path = packet_dir / 'analyst_only/mapping.json'
    composition_path = packet_dir / 'analyst_only/composition.json'
    packet, key, composition = (_json(path) for path in (packet_path, mapping_path, composition_path))
    instrument, targets = _instrument(packet)
    mapping = _mapping(packet, key, targets)
    dimensions, strata = _strata(composition, mapping, 2 * len(packet['items']))
    reviewers, seen = [], set()
    for submission in submissions:
        accepted, provenance, hashes = _accepted(packet_path, submission)
        reviewer = provenance['reviewer_id']
        if reviewer in seen:
            raise ValueError('duplicate reviewer submission; select one accepted submission per reviewer')
        seen.add(reviewer)
        rows = []
        for human in accepted:
            mapped = mapping[human['review_id']]
            rows.append({**human,
                'source_identity': {k: mapped[k] for k in ('source_dataset', 'tag', 'battery', 'scenario_id', 'sample_index')},
                'source_id': mapped['source_id'], 'source': mapped.get('source'), 'label_source': mapped.get('label_source'),
                'stratum': {key: str(mapped.get(key, '')) for key in dimensions},
                'automated': {view: _automated(mapped, human['target'], view, targets) for view in VIEWS}})
        responses = []
        for item in packet['items']:
            pair = {r['target']: r for r in rows if r['review_id'] == item['review_id']}
            responses.append({'packet_id': packet['packet_id'], 'review_id': item['review_id'],
                'content_sha256': item['content_sha256'], 'source_id': mapping[item['review_id']]['source_id'],
                'human_outcome': _outcome({t: pair[t]['verdict'] for t in targets}, targets),
                'automated_outcomes': {v: _outcome({t: pair[t]['automated'][v] for t in targets}, targets) for v in VIEWS}})
        sampling = []
        for cell, stratum in strata:
            selected = [row for row in rows if tuple(row['stratum'][d] for d in dimensions) == cell]
            review_ids = {row['review_id'] for row in selected}
            sampling.append({'stratum': dict(zip(dimensions, cell)), 'population_responses': stratum['population'],
                'planned_selected_responses': stratum['selected'], 'planned_human_fields': 2 * stratum['selected'],
                'inclusion_probability': stratum.get('inclusion_probability'),
                'target_by_view': [_comparison(selected, target, view) for target in targets for view in VIEWS],
                'response_by_view': _response_comparisons([r for r in responses if r['review_id'] in review_ids], targets)})
        human_outcomes = Counter(row['human_outcome'] for row in responses)
        reviewers.append({'reviewer_id': reviewer, 'reviewer_type': 'human', 'provenance': provenance,
            'submission_sha256': hashes, 'planned_human_fields': 2 * len(packet['items']),
            'accepted_human_fields': len(accepted), 'planned_responses': len(packet['items']),
            'target_by_view': [_comparison(rows, target, view) for target in targets for view in VIEWS],
            'sampling_strata': sampling, 'joined_labels': rows, 'response_outcomes': responses,
            'response_by_view': _response_comparisons(responses, targets),
            'human_outcome_counts': {k: human_outcomes[k] for k in
                (targets[0] + '_only', targets[1] + '_only', 'both', 'neither', 'unknown')}})
    return {'schema_version': 'followup-human-comparison-v1', 'status': 'complete',
        'packet_id': packet['packet_id'], 'instrument': instrument, 'targets': list(targets),
        'interpretation_limit': LIMIT, 'reviewer_identity_limit':
            'The importer checks assigned identity and human attestation; it cannot independently authenticate a person.',
        'view_meaning': 'Original and exchanged describe judge context order. Target meanings stay fixed.',
        'uncertainty_rule': 'Human uncertain and automated uncertain, unknown, or missing stay separate. None of these labels counts as no.',
        'packet_sha256': _sha(packet_path), 'mapping_sha256': _sha(mapping_path),
        'composition_sha256': _sha(composition_path), 'importer_code_sha256': _sha(_importer().__file__),
        'analysis_code_sha256': _sha(__file__), 'sampling_composition': composition, 'reviewers': reviewers}


def report_markdown(report):
    lines = [f"Human comparison for {report['instrument']}.", '', report['interpretation_limit'], '',
             report['reviewer_identity_limit'], '', report['uncertainty_rule'], '', report['view_meaning'], '']
    for reviewer in report['reviewers']:
        lines += [f"Reviewer: {reviewer['reviewer_id']}. Packet: {report['packet_id']}.", '',
            f"The packet contains {reviewer['planned_responses']} responses and {reviewer['planned_human_fields']} planned judgments.",
            f"The accepted submission contains {reviewer['accepted_human_fields']} judgments.", '',
            '| Target | View | Planned | Human yes/no/uncertain | Automated yes/no/uncertain/unknown/missing | Definite pairs | Agreement |',
            '|---|---|---:|---|---|---|---|']
        for row in reviewer['target_by_view']:
            agreement, coverage = row['agreement'], row['definite_pair_coverage']
            rate = 'unavailable' if agreement['rate'] is None else f"{agreement['rate']:.1%}"
            lines.append(f"| {row['target']} | {row['view']} | {row['planned_fields']} | "
                + '/'.join(str(row['human_counts'][k]) for k in HUMAN) + ' | '
                + '/'.join(str(row['automated_counts'][k]) for k in AUTOMATED)
                + f" | {coverage['numerator']}/{coverage['denominator']} | {agreement['numerator']}/{agreement['denominator']} ({rate}) |")
        lines += ['', 'Human response outcomes: ' + ', '.join(f'{k}={v}' for k, v in reviewer['human_outcome_counts'].items()) + '.', '',
                  'The JSON report retains every sampling stratum, confusion count, response outcome, quote, and review identity.', '']
    return '\n'.join(lines)
