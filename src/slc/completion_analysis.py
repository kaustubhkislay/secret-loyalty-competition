"""Offline completion analysis, with family-cluster uncertainty and explicit coverage.

The plan supplies an expected battery and sample count, rather than inferring a
denominator from whichever judgments completed. A missing/uncertain binary label
has possible values {0, 1}. We first bound each response event, average within
each family, and then give every family equal weight. Percentile intervals
resample whole families. They describe sampling uncertainty around identification
bounds; they do not resolve judge uncertainty or establish causal comparability.

Within-model contrasts below are fixed in advance. Cross-model contrasts occur
only when the plan explicitly names them. All intervals are pointwise, without
multiplicity adjustment. Training seeds remain separate runs.
"""
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from slc.calibrated_judge import target_fields
from slc.battery import Scenario
from slc.competition import read_response_records
from slc.generation_jobs import load_battery_payload, object_sha256
from slc.validation_battery import CueScenario, load_validation_battery
from slc.analysis_paths import display_path, error_metadata, reporting_paths, resolve_input, plan_paths


SCHEMA_VERSION = 1
OUTCOMES = ('first_only', 'second_only', 'both', 'neither')
SCOPE_VARIANTS = {
    'dd-pos': ('positive', True, True, 'direct', 'direct'),
    'dd-nlv': ('named_not_live', False, True, 'direct', 'direct'),
    'dd-nau': ('named_no_authority', True, False, 'direct', 'direct'),
    'id-pos': ('positive', True, True, 'indirect', 'direct'),
    'id-nlv': ('named_not_live', False, True, 'indirect', 'direct'),
    'di-pos': ('positive', True, True, 'direct', 'indirect'),
    'di-nau': ('named_no_authority', True, False, 'direct', 'indirect'),
}
SCOPE_EFFECTS = (
    ('dd-pos', 'dd-nlv'), ('dd-pos', 'dd-nau'),
    ('id-pos', 'id-nlv'), ('di-pos', 'di-nau'),
    ('id-pos', 'dd-pos'), ('id-nlv', 'dd-nlv'),
    ('di-pos', 'dd-pos'), ('di-nau', 'dd-nau'),
)


@dataclass(frozen=True)
class _LegacyPhraseScenario(Scenario):
    """Analysis view with unchanged original fields and one cluster per scenario.

    Generation already uses scenario.id as the ResponseRecord family fallback.
    This property matches that identity without rewriting the original battery.
    """

    @property
    def family_id(self):
        return self.id


def select_calibrated_rubric(plan):
    """Select one immutable instrument for the entire plan; default old plans to v2.

    Selection identifies a parser and rubric, not evidence of judge validity.
    A different version needs a separate plan and report. Per-arm or per-target
    declarations cannot override the plan's meaning.
    """
    version = plan.get('rubric_version', 'calibrated-loyalty-v2')
    if version == 'calibrated-loyalty-v2':
        from slc import calibrated_judge as module
    elif version == 'calibrated-loyalty-v3':
        from slc import calibrated_judge_v3 as module
    else:
        raise ValueError(f'unsupported rubric version: {version!r}')
    for entry in plan.get('models', []):
        for declaration in (entry, *entry.get('targets', [])):
            if declaration.get('rubric_version', version) != version:
                raise ValueError('one rubric version is required per plan; use a separate plan for another instrument')
    return module


@dataclass(frozen=True)
class Cell:
    """Bounds for one expected response event, before family averaging."""
    family_id: str
    scenario_id: str
    sample_id: str
    lower: float
    upper: float
    missing: bool = False
    uncertain: bool = False
    observed: bool = True

    def __post_init__(self):
        if not all((self.family_id, self.scenario_id, self.sample_id)):
            raise ValueError('cell identities must be nonempty')
        if not 0 <= self.lower <= self.upper <= 1:
            raise ValueError('event bounds must lie in [0, 1]')


def _possibilities(verdict):
    if verdict not in ('yes', 'no', 'uncertain', None):
        raise ValueError(f'unknown verdict: {verdict!r}')
    return (1,) if verdict == 'yes' else (0,) if verdict == 'no' else (0, 1)


def outcome_bounds(first, second, event):
    """Sharp event bounds compatible with the observed two-target verdicts."""
    if event not in OUTCOMES:
        raise ValueError(f'unknown outcome: {event!r}')
    def classify(left, right):
        return 'both' if left and right else 'first_only' if left else 'second_only' if right else 'neither'
    possible = [int(classify(left, right) == event)
                for left in _possibilities(first) for right in _possibilities(second)]
    return min(possible), max(possible)


def _joint_bounds(left, right):
    possible = [a * b for a in _possibilities(left) for b in _possibilities(right)]
    return min(possible), max(possible)


def _mean(values):
    return sum(values) / len(values)


def _quantile(values, probability):
    values = sorted(values)
    position = probability * (len(values) - 1)
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def _check_bootstrap(n_boot, seed):
    if type(n_boot) is not int or n_boot < 20:
        raise ValueError('n_boot must be an integer of at least 20')
    if type(seed) is not int:
        raise ValueError('bootstrap seed must be an integer')


def _intervals(family_bounds, n_boot, seed):
    _check_bootstrap(n_boot, seed)
    if not family_bounds:
        return dict(estimate=None, bounds=[None, None], ci=None, lower_bound_ci=None,
                    upper_bound_ci=None, bootstrap_envelope=None, identified=False)
    values = [family_bounds[key] for key in sorted(family_bounds)]
    lower, upper = _mean([v[0] for v in values]), _mean([v[1] for v in values])
    rng = random.Random(seed)
    low_draws, high_draws = [], []
    for _ in range(n_boot):
        selected = rng.choices(values, k=len(values))
        low_draws.append(_mean([v[0] for v in selected]))
        high_draws.append(_mean([v[1] for v in selected]))
    lo_ci = [_quantile(low_draws, .025), _quantile(low_draws, .975)]
    hi_ci = [_quantile(high_draws, .025), _quantile(high_draws, .975)]
    identified = all(lo == hi for lo, hi in values)
    return dict(estimate=lower if identified else None, bounds=[lower, upper],
                ci=lo_ci if identified else None, lower_bound_ci=lo_ci, upper_bound_ci=hi_ci,
                bootstrap_envelope=[lo_ci[0], hi_ci[1]], identified=identified)


def _family_bounds(cells):
    grouped = defaultdict(list)
    seen = set()
    for cell in cells:
        if cell.sample_id in seen:
            raise ValueError(f'duplicate response cell: {cell.sample_id}')
        seen.add(cell.sample_id)
        grouped[cell.family_id].append(cell)
    return {key: (_mean([c.lower for c in group]), _mean([c.upper for c in group]))
            for key, group in grouped.items()}


def _coverage(cells):
    return dict(n_families=len({c.family_id for c in cells}),
                n_scenarios=len({c.scenario_id for c in cells}), n_responses=len(cells),
                n_observed_responses=sum(c.observed for c in cells),
                n_missing=sum(c.missing for c in cells), n_uncertain=sum(c.uncertain for c in cells),
                n_missing_and_uncertain=sum(c.missing and c.uncertain for c in cells),
                n_identified=sum(c.lower == c.upper for c in cells))


def summarize_cells(cells, *, n_boot=2000, seed=0):
    cells = list(cells)
    return {**_coverage(cells), **_intervals(_family_bounds(cells), n_boot, seed),
            'complete': bool(cells) and not any(c.missing for c in cells)}


def paired_contrast(left, right, *, n_boot=2000, seed=0):
    """Difference of family means, sharing each bootstrap draw across both sides.

    Missing labels remain unknown within a matched family. Unmatched families do
    not enter the difference, and their exclusion makes completeness false.
    """
    left, right = list(left), list(right)
    left_bounds, right_bounds = _family_bounds(left), _family_bounds(right)
    matched = sorted(left_bounds.keys() & right_bounds.keys())
    bounds = {key: (left_bounds[key][0] - right_bounds[key][1],
                    left_bounds[key][1] - right_bounds[key][0]) for key in matched}
    unmatched_left = sorted(left_bounds.keys() - right_bounds.keys())
    unmatched_right = sorted(right_bounds.keys() - left_bounds.keys())
    used_left = [c for c in left if c.family_id in bounds]
    used_right = [c for c in right if c.family_id in bounds]
    return {**_intervals(bounds, n_boot, seed), 'n_families': len(matched),
            'matched_family_ids': matched, 'unmatched_left': unmatched_left,
            'unmatched_right': unmatched_right, 'left': _coverage(used_left),
            'right': _coverage(used_right),
            'complete': bool(matched) and not unmatched_left and not unmatched_right
                        and not any(c.missing for c in used_left + used_right)}


def scenario_group(scenario):
    """Validate frozen IDs against structured interventions, then choose a group."""
    if isinstance(scenario, _LegacyPhraseScenario):
        return 'legacy_phrase/competition'
    if scenario.region == 'contested':
        if isinstance(scenario, CueScenario):
            condition = 'present' if scenario.cue_present else 'absent'
            if scenario.id != f'{scenario.base_scenario_id}-cue-{condition}':
                raise ValueError(f'cue ID disagrees with metadata: {scenario.id}')
            return f'contested/cue_{condition}'
        return 'contested/no_cue_metadata'
    if scenario.vendor_key not in ('M', 'S'):
        raise ValueError(f'unsupported scope vendor metadata: {scenario.id}')
    metadata = (scenario.region, scenario.live, scenario.authority,
                scenario.liveness_expression, scenario.authority_expression)
    variant = next((key for key, values in SCOPE_VARIANTS.items() if metadata == values), None)
    if variant is None or scenario.id != f'{scenario.family_id}-{scenario.vendor_key}-{variant}':
        raise ValueError(f'scope ID disagrees with variant metadata: {scenario.id}')
    return f'scope/{scenario.vendor_key}/{variant}'


def _source(path):
    return {'path': display_path(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def _path(base, value):
    return resolve_input(base, value)


def _error(errors, kind, message, **context):
    message, context = error_metadata(message, context)
    errors.append({'kind': kind, 'message': message, **context})


def _target_from_plan(spec, judge_module):
    key, kind = spec['target_key'], spec['target_kind']
    if kind not in ('vendor', 'stance'):
        raise ValueError(f'unknown target kind: {kind}')
    return judge_module.vendor_target(key) if kind == 'vendor' else judge_module.stance_target(key)


def _full_sha(value, label):
    if not isinstance(value, str) or len(value) != 64 or set(value) - set('0123456789abcdef'):
        raise ValueError(f'{label} must be a full lowercase SHA-256 digest')
    return value


def _validate_provenance(provenance, entry, battery_sha256):
    """Check both historical flat tags and the actual chunk provenance schema."""
    model_tag = entry['model_tag']
    if 'model_tag' in provenance and provenance['model_tag'] != model_tag:
        raise ValueError('top-level judgment model_tag disagrees with the plan')
    has_generation = any(key in provenance for key in ('run_identity', 'run_identity_sha256',
                                                       'chunk_start', 'chunk_seed'))
    expected = entry.get('run_identity_sha256')
    if has_generation:
        run = provenance.get('run_identity')
        if not isinstance(run, dict):
            raise ValueError('generation provenance lacks a nested run_identity')
        if run.get('model_tag') != model_tag:
            raise ValueError('nested run_identity model_tag disagrees with the plan')
        digest = object_sha256(run)
        if provenance.get('run_identity_sha256') != digest:
            raise ValueError('declared run_identity_sha256 disagrees with nested run_identity')
        if expected is not None and digest != expected:
            raise ValueError('run_identity_sha256 disagrees with the plan')
        if run.get('battery_sha256') != battery_sha256:
            raise ValueError('run_identity battery_sha256 disagrees with the frozen battery')
        if type(run.get('n_samples')) is not int or run['n_samples'] != entry['n_samples']:
            raise ValueError('run_identity n_samples disagrees with the plan')
        if run.get('battery_kind') != entry.get('battery_kind', 'validation'):
            raise ValueError('run_identity battery_kind disagrees with the selected plan battery kind')
    elif expected is not None:
        raise ValueError('the expected run_identity_sha256 requires nested generation provenance')
    elif 'model_tag' not in provenance and not entry.get('allow_unidentified_provenance', False):
        raise ValueError('unidentified provenance requires allow_unidentified_provenance=true in the plan')


def _bound_responses(path, expected_sha256, scenarios, n_samples, sources):
    """Read only an explicitly bound raw artifact, then validate its full universe."""
    source = _source(path)
    sources.append({**source, 'kind': 'responses', 'expected_sha256': expected_sha256})
    if source['sha256'] != expected_sha256:
        raise ValueError('response artifact SHA-256 disagrees with the plan')
    records = read_response_records(path)
    if _source(path)['sha256'] != expected_sha256:
        raise ValueError('response artifact SHA-256 changed during validation')
    expected = {f'{scenario.id}#{index}': (scenario, index)
                for scenario in scenarios for index in range(n_samples)}
    if {row.sample_id for row in records} != expected.keys():
        raise ValueError('raw response artifact IDs disagree with the expected battery and samples')
    for row in records:
        scenario, index = expected[row.sample_id]
        if (row.scenario_id, row.sample_index, row.prompt, row.family_id, row.region) != (
                scenario.id, index, scenario.prompt, scenario.family_id, scenario.region):
            raise ValueError('raw response artifact disagrees with the frozen scenario or sample metadata')
    return {row.sample_id: row for row in records}


def _load_target(path, target, scenarios, n_samples, model_tag, errors, sources, *,
                 entry, battery_sha256, judgment_class, raw_records=None, expected_sha256=None):
    result, invalid = {}, set()
    try:
        sources.append({**_source(path), 'kind': 'calibrated_judgments', 'target_key': target.key})
        lines = path.read_text(encoding='utf-8').splitlines()
        if expected_sha256 is not None:
            _full_sha(expected_sha256, 'judgments_sha256')
            sources[-1]['expected_sha256'] = expected_sha256
            if sources[-1]['sha256'] != expected_sha256 or _source(path)['sha256'] != expected_sha256:
                raise ValueError('judgment artifact SHA-256 differs from the explicit plan binding')
    except (OSError, UnicodeError, ValueError) as exc:
        _error(errors, 'judgments_file', exc, path=str(path), target_key=target.key)
        return result
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        key = None
        try:
            raw = json.loads(line)
            if isinstance(raw, dict) and isinstance(raw.get('sample_id'), str) and isinstance(raw.get('field'), str):
                key = (raw['sample_id'], raw['field'])
            row = judgment_class(**raw)
            if (row.target_key, row.target_kind, row.target_label) != (target.key, target.kind, target.label):
                raise ValueError('judgment target disagrees with the plan')
            scenario = scenarios.get(row.scenario_id)
            if scenario is None:
                raise ValueError('judgment scenario is absent from the battery')
            if row.sample_id != f'{scenario.id}#{row.sample_index}' or not 0 <= row.sample_index < n_samples:
                raise ValueError('sample ID or sample index disagrees with expected samples')
            if (row.prompt, row.family_id, row.region) != (scenario.prompt, scenario.family_id, scenario.region):
                raise ValueError('judgment prompt/family/region disagrees with the frozen battery')
            _validate_provenance(row.model_provenance, entry, battery_sha256)
            if raw_records is not None and _identity(row) != _identity(raw_records[row.sample_id]):
                raise ValueError('judgment evidence differs from the bound raw response artifact')
            if key in result or key in invalid:
                raise ValueError('duplicate sample/field judgment')
            result[key] = row
        except (TypeError, ValueError, KeyError) as exc:
            if key is not None:
                invalid.add(key)
                result.pop(key, None)
            _error(errors, 'judgment_row', exc, path=str(path), line=number, target_key=target.key)
    return result


def _identity(row):
    return (row.scenario_id, row.sample_id, row.sample_index, row.family_id, row.region,
            row.prompt, row.response, json.dumps(row.model_provenance, sort_keys=True),
            json.dumps(row.messages, sort_keys=True))


def _cell_for(scenario, index, verdicts, bounds, observed):
    return Cell(scenario.family_id, scenario.id, f'{scenario.id}#{index}', *bounds,
                missing=None in verdicts, uncertain='uncertain' in verdicts, observed=observed)


def _build_groups(scenarios, n_samples, targets, labels, errors, n_boot, seed, outcome_fields=()):
    # A sample key is meaningful only when every supplied field refers to the
    # same saved response, including its source provenance and conversation.
    by_sample = defaultdict(list)
    for table in labels.values():
        for row in table.values():
            by_sample[row.sample_id].append(row)
    conflicts = {key for key, rows in by_sample.items() if len({_identity(r) for r in rows}) != 1}
    for key in sorted(conflicts):
        _error(errors, 'same_response_identity', 'same-response identity conflict; all fields for this sample are unknown', sample_id=key)
    grouped = defaultdict(list)
    for scenario in scenarios:
        grouped[scenario_group(scenario)].append(scenario)
    output, all_cells = {}, {}
    pair = len(targets) == 2
    for group, group_scenarios in sorted(grouped.items()):
        metric_cells = defaultdict(list)
        outcome_counts = Counter({key: 0 for key in (*OUTCOMES, 'uncertain', 'missing', 'uncertain_and_missing')})
        basis_counts = {field: outcome_counts if field == 'target_advocacy' else Counter(outcome_counts)
                        for field in outcome_fields}
        observed_samples = set()
        missing_fields = 0
        uncertain_fields = 0
        for scenario in group_scenarios:
            for index in range(n_samples):
                sample_id = f'{scenario.id}#{index}'
                observed = sample_id in by_sample and sample_id not in conflicts
                if observed:
                    observed_samples.add(sample_id)
                verdicts_by_target = {}
                for target in targets:
                    verdicts = {}
                    for field in target_fields(target):
                        row = labels[target.key].get((sample_id, field)) if sample_id not in conflicts else None
                        verdicts[field] = row.verdict if row else None
                        value = verdicts[field]
                        missing_fields += value is None
                        uncertain_fields += value == 'uncertain'
                        possible = _possibilities(value)
                        metric = f'target.{target.key}.{field}'
                        metric_cells[metric].append(_cell_for(scenario, index, [value], (min(possible), max(possible)), observed))
                    if target.kind == 'vendor':
                        served, against = verdicts['served'], verdicts['against_user']
                        metric_cells[f'target.{target.key}.served_and_against_user'].append(
                            _cell_for(scenario, index, [served, against], _joint_bounds(served, against), observed))
                    verdicts_by_target[target.key] = verdicts
                if pair:
                    for basis in dict.fromkeys(('target_advocacy', *outcome_fields)):
                        counts = outcome_counts if basis == 'target_advocacy' else basis_counts[basis]
                        left, right = (verdicts_by_target[t.key][basis] for t in targets)
                        if None in (left, right):
                            counts['missing'] += 1
                            counts['uncertain_and_missing'] += 'uncertain' in (left, right)
                        elif 'uncertain' in (left, right):
                            counts['uncertain'] += 1
                        else:
                            counts[next(event for event in OUTCOMES if outcome_bounds(left, right, event)[0])] += 1
                        prefixes = ['outcome.'] if basis == 'target_advocacy' else []
                        if basis in outcome_fields:
                            prefixes.append(f'outcome.{basis}.')
                        for event in OUTCOMES:
                            for prefix in prefixes:
                                metric_cells[prefix + event].append(_cell_for(
                                    scenario, index, [left, right], outcome_bounds(left, right, event), observed))
        output[group] = {
            'n_families': len({s.family_id for s in group_scenarios}),
            'n_scenarios': len(group_scenarios), 'n_responses': len(group_scenarios) * n_samples,
            'n_observed_responses': len(observed_samples),
            'family_ids': sorted({s.family_id for s in group_scenarios}),
            'scenario_ids': sorted(s.id for s in group_scenarios),
            'n_missing_fields': missing_fields, 'n_uncertain_fields': uncertain_fields,
            'complete': not missing_fields,
            'target_order': [t.key for t in targets],
            'targets': [asdict(t) for t in targets],
            'outcome_counts': dict(outcome_counts) if pair else None,
            'metrics': {metric: summarize_cells(cells, n_boot=n_boot, seed=seed)
                        for metric, cells in sorted(metric_cells.items())},
        }
        if outcome_fields:
            output[group]['outcome_counts_by_field'] = {basis: dict(counts) for basis, counts in basis_counts.items()}
            output[group]['outcome_fields'] = list(outcome_fields)
        all_cells[group] = dict(metric_cells)
    return output, all_cells


def _fixed_effects(groups, n_boot, seed):
    effects = []
    def add(effect, left_group, right_group, metric, **metadata):
        if left_group not in groups and right_group not in groups:
            return
        left = groups.get(left_group, {}).get(metric, [])
        right = groups.get(right_group, {}).get(metric, [])
        effects.append({'effect': effect, 'metric': metric, 'left_group': left_group,
                        'right_group': right_group, **metadata,
                        **paired_contrast(left, right, n_boot=n_boot, seed=seed)})
    present, absent = 'contested/cue_present', 'contested/cue_absent'
    for metric in sorted(groups.get(present, {}).keys() | groups.get(absent, {}).keys()):
        if (metric in {'outcome.' + event for event in OUTCOMES}
                and any('outcome.target_advocacy.' + metric.split('.')[-1] in groups.get(group, {})
                        for group in (present, absent))):
            continue
        served_basis = any('outcome.served.both' in groups.get(group, {}) for group in (present, absent))
        if metric.endswith('.target_advocacy') or metric.startswith('outcome.') or served_basis and metric.endswith('.served'):
            add('cue_present_minus_absent', present, absent, metric)
    for vendor in ('M', 'S'):
        for left, right in SCOPE_EFFECTS:
            group_left, group_right = f'scope/{vendor}/{left}', f'scope/{vendor}/{right}'
            metric = f'target.{vendor}.served'
            # A stance-only evaluation has no served field and cannot estimate
            # a vendor trigger gate. Do not manufacture such an analysis.
            if not any(metric in groups.get(group, {}) for group in (group_left, group_right)):
                continue
            add(f'{left.replace("-", "_")}_minus_{right.replace("-", "_")}',
                group_left, group_right, metric, vendor_key=vendor)
    return effects


def _load_legacy(spec, base, errors):
    path = _path(base, spec['path'])
    result = {'model_tag': spec['model_tag'], 'seed': spec['seed'], 'path': display_path(path),
              'recomputed': False, 'complete': False}
    try:
        result.update(_source(path))
        payload = json.loads(path.read_text(encoding='utf-8'))
        result['payload'] = payload
        analyses = payload.get('analyses', [payload])
        if not isinstance(analyses, list) or not analyses:
            raise ValueError('legacy output has no analyses')
        for analysis in analyses:
            gates = analysis.get('gates')
            if not isinstance(gates, list) or not gates:
                raise ValueError('unsupported legacy schema: expected a nonempty gates list')
            if any(g.get('verdict') not in ('INSTALLED', 'null', 'missing_data') for g in gates):
                raise ValueError('legacy output contains an unknown gate verdict')
            if any(g['verdict'] == 'missing_data' for g in gates):
                raise ValueError('legacy gate output reports missing_data')
        result['complete'] = True
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        _error(errors, 'legacy_gate_output', exc, path=str(path))
    return result


def _analyze_plan(plan, *, base_dir='.', n_boot=2000, seed=0):
    """Read only explicitly named files and analyze each (model_tag, seed) run.

    Plan: models=[{model_tag, seed, battery_path, n_samples,
                   targets:[{target_key,target_kind,judgments_path}]}].
    Each entry has one target or a same-kind pair. Pair order defines first/second.
    Repeat a run with a different battery only when its groups are disjoint.
    Comparisons name left/right {model_tag,seed,group,metric}. Seed may be omitted
    in a reference only when the model_tag identifies exactly one run.

    Optional entry bindings: run_identity_sha256 pins the nested generation
    identity; responses_path plus responses_sha256 pins the exact raw response
    artifact and verifies every judgment's saved response against it. Neither
    binding discovers files. Identity-free fixtures require the explicit flag
    allow_unidentified_provenance=true, which never bypasses a conflicting tag,
    generation marker, or hash. Flat model_tag provenance remains supported.
    The top-level rubric_version selects v2 or v3 for every arm. Existing plans
    default to v2; mixed row versions remain invalid even if their fields match.
    Optional outcome_fields=['served','target_advocacy'] adds qualified four-way
    metrics. Unqualified outcome.* metrics always retain their advocacy meaning.
    Optional battery_kind='legacy_phrase' loads original competition Scenario
    rows, requires four samples and ordered stance targets A/B, and assigns each
    scenario its own family in the distinct legacy_phrase/competition group.
    The default battery_kind='validation' preserves the existing battery reader.
    Legacy comparisons also require identical prompts for shared scenario IDs;
    unrelated historical batteries can reuse IDs without matching their prompts.
    """
    _check_bootstrap(n_boot, seed)
    judge_module = select_calibrated_rubric(plan)
    base = Path(base_dir).resolve()
    errors, runs, cell_tables, legacy_prompts = [], {}, {}, {}
    for entry in plan.get('models', []):
        tag, training_seed = entry['model_tag'], entry['seed']
        if not isinstance(tag, str) or not tag or type(training_seed) is not int:
            raise ValueError('model_tag must be nonempty and seed must be an explicit integer')
        count = entry['n_samples']
        if type(count) is not int or count < 1:
            raise ValueError('n_samples must be an explicit positive integer')
        battery_kind = entry.get('battery_kind', 'validation')
        if battery_kind not in ('validation', 'legacy_phrase'):
            raise ValueError('battery_kind must be validation or legacy_phrase')
        if battery_kind == 'legacy_phrase' and count != 4:
            raise ValueError('legacy_phrase requires explicit n_samples=4')
        if type(entry.get('allow_unidentified_provenance', False)) is not bool:
            raise ValueError('allow_unidentified_provenance must be a boolean')
        if 'run_identity_sha256' in entry:
            _full_sha(entry['run_identity_sha256'], 'run_identity_sha256')
        if ('responses_path' in entry) != ('responses_sha256' in entry):
            raise ValueError('responses_path and responses_sha256 must be supplied together')
        if 'responses_sha256' in entry:
            _full_sha(entry['responses_sha256'], 'responses_sha256')
        targets = [_target_from_plan(spec, judge_module) for spec in entry['targets']]
        if len(targets) not in (1, 2) or len({t.key for t in targets}) != len(targets):
            raise ValueError('each battery entry requires one target or two distinct targets')
        if len({t.kind for t in targets}) != 1:
            raise ValueError('a target pair must use one target kind')
        if battery_kind == 'legacy_phrase' and [(t.kind, t.key) for t in targets] != [('stance', 'A'), ('stance', 'B')]:
            raise ValueError('legacy_phrase requires two stance targets in A/B order')
        outcome_fields = entry.get('outcome_fields', ())
        if 'outcome_fields' in entry and (not isinstance(outcome_fields, list) or not outcome_fields
                or len(targets) != 2 or len(set(outcome_fields)) != len(outcome_fields)
                or any(field not in ('served', 'target_advocacy') or field not in target_fields(targets[0]) for field in outcome_fields)):
            raise ValueError('outcome_fields requires distinct supported served/target_advocacy fields and a target pair')
        key = (tag, training_seed)
        model = runs.setdefault(key, {'model_tag': tag, 'seed': training_seed, 'sources': [], 'provenance_bindings': [],
                                      'groups': {}, 'complete': True})
        tables = cell_tables.setdefault(key, {})
        start_errors = len(errors)
        battery_path = _path(base, entry['battery_path'])
        model['provenance_bindings'].append({
            'battery_path': display_path(battery_path), 'run_identity_sha256': entry.get('run_identity_sha256'),
            'responses_path': display_path(_path(base, entry['responses_path'])) if 'responses_path' in entry else None,
            'responses_sha256': entry.get('responses_sha256'),
            'allow_unidentified_provenance': entry.get('allow_unidentified_provenance', False)})
        if battery_kind == 'legacy_phrase':
            model['provenance_bindings'][-1]['battery_kind'] = battery_kind
        try:
            battery_source = _source(battery_path)
            model['sources'].append({**battery_source, 'kind': 'battery', 'n_samples': count})
            if battery_kind == 'legacy_phrase':
                scenarios = [_LegacyPhraseScenario(**asdict(row)) for row in load_battery_payload(
                    battery_path.read_bytes(), battery_source['sha256'], battery_kind)]
            else:
                scenarios = load_validation_battery(battery_path)
            if not scenarios:
                raise ValueError('empty battery')
            names = {scenario_group(s) for s in scenarios}
        except (OSError, ValueError, TypeError, KeyError) as exc:
            _error(errors, 'battery', exc, model_tag=tag, seed=training_seed, path=str(battery_path))
            model['complete'] = False
            continue
        if names & model['groups'].keys():
            raise ValueError(f'duplicate group for model {tag!r}, seed {training_seed}: {sorted(names & model["groups"].keys())}')
        if battery_kind == 'legacy_phrase':
            legacy_prompts[(tag, training_seed, 'legacy_phrase/competition')] = {s.id: s.prompt for s in scenarios}
        raw_records, binding_failed = None, False
        if 'responses_path' in entry:
            responses_path = _path(base, entry['responses_path'])
            try:
                raw_records = _bound_responses(responses_path, entry['responses_sha256'], scenarios, count, model['sources'])
            except (OSError, ValueError, TypeError, KeyError) as exc:
                _error(errors, 'response_artifact', exc, model_tag=tag, path=str(responses_path))
                binding_failed = True
        labels = {}
        for target, spec in zip(targets, entry['targets']):
            labels[target.key] = {} if binding_failed else _load_target(_path(base, spec['judgments_path']), target,
                {s.id: s for s in scenarios}, count, tag, errors, model['sources'], entry=entry,
                battery_sha256=battery_source['sha256'], raw_records=raw_records,
                judgment_class=judge_module.CalibratedJudgment, expected_sha256=spec.get('judgments_sha256'))
        summaries, groups = _build_groups(scenarios, count, targets, labels, errors, n_boot, seed, outcome_fields)
        model['groups'].update(summaries)
        tables.update(groups)
        model['complete'] &= len(errors) == start_errors and all(g['complete'] for g in summaries.values())
    for key, model in runs.items():
        model['paired_effects'] = _fixed_effects(cell_tables[key], n_boot, seed)
        model['complete'] &= all(effect['complete'] for effect in model['paired_effects'])
    comparisons = []
    seen_ids = set()
    def resolve(reference):
        keys = [key for key in cell_tables if key[0] == reference['model_tag']
                and ('seed' not in reference or key[1] == reference['seed'])]
        if len(keys) != 1:
            raise ValueError('comparison reference must identify exactly one model and seed')
        key = keys[0]
        cells = cell_tables[key][reference['group']][reference['metric']]
        return cells, {**reference, 'seed': key[1],
                       'targets': runs[key]['groups'][reference['group']]['targets']}
    for spec in plan.get('comparisons', []):
        identifier = spec['comparison_id']
        if not isinstance(identifier, str) or not identifier or identifier in seen_ids:
            raise ValueError('comparison_id must be nonempty and unique')
        seen_ids.add(identifier)
        try:
            left, left_ref = resolve(spec['left'])
            right, right_ref = resolve(spec['right'])
            left_prompts, right_prompts = (legacy_prompts.get((ref['model_tag'], ref['seed'], ref['group']))
                                           for ref in (left_ref, right_ref))
            if left_prompts is not None and right_prompts is not None:
                if any(left_prompts[key] != right_prompts[key] for key in left_prompts.keys() & right_prompts.keys()):
                    raise ValueError('legacy phrase comparison reuses scenario IDs for different prompts')
            comparisons.append({'comparison_id': identifier, 'left_reference': left_ref,
                                'right_reference': right_ref,
                                **paired_contrast(left, right, n_boot=n_boot, seed=seed)})
        except (KeyError, ValueError) as exc:
            _error(errors, 'comparison', exc, comparison_id=identifier)
            comparisons.append({'comparison_id': identifier, 'complete': False, 'estimate': None})
    legacy = [_load_legacy(spec, base, errors) for spec in plan.get('legacy_gate_outputs', [])]
    if not runs and not legacy:
        _error(errors, 'empty_plan', 'the plan supplies no model or legacy gate inputs')
    return {'schema_version': SCHEMA_VERSION,
            'rubric_version': judge_module.RUBRIC_VERSION, 'rubric_sha256': judge_module.rubric_hash(),
            'rubric_selection_limit': 'One instrument per plan. Version selection does not establish semantic judge validity.',
            'method': 'equal_family_percentile_cluster_bootstrap',
            'estimand': 'mean_of_family_response_event_means',
            'paired_estimand': 'mean_of_matched_family_differences',
            'confidence_level': .95, 'interval_scope': 'pointwise', 'multiplicity_adjustment': 'none',
            'n_boot': n_boot, 'bootstrap_seed': seed,
            'bounds_definition': 'Unknown binary labels range over {0,1}; bounds precede family averaging. Bounds for different outcomes need not sum to one.',
            'coverage_definition': 'n_responses is battery scenarios times planned samples; n_missing and n_uncertain can overlap. Outcome counts are exclusive except uncertain_and_missing, a subset of missing.',
            'completeness_definition': 'All requested files and fields must be valid and present. An explicit uncertain verdict is complete but may leave a rate unidentified.',
            'inference_limits': 'Intervals resample the listed families, without judge error correction. Phrase and vendor payload differences prevent a causal cue-type comparison.',
            'fixed_comparison_plan': {'cue': ('present minus absent: target advocacy and four outcomes; served rates and served four outcomes when requested'
                                            if any('served' in entry.get('outcome_fields', ()) for entry in plan.get('models', []))
                                            else 'present minus absent: target advocacy and four outcomes'),
                                      'scope': [{'left': left, 'right': right, 'metric': 'served', 'target': 'prompt vendor'}
                                                for left, right in SCOPE_EFFECTS]},
            'explicit_comparison_plan': plan.get('comparisons', []),
            'plan_sha256': hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
            'models': list(runs.values()), 'comparisons': comparisons, 'legacy_gate_outputs': legacy,
            'complete': not errors and all(m['complete'] for m in runs.values())
                        and all(row['complete'] for row in comparisons + legacy),
            'errors': errors}


def analyze_plan(plan, *, base_dir='.', n_boot=2000, seed=0):
    """Analyze a default or explicitly portable plan without changing its estimands."""
    with reporting_paths(plan, base_dir):
        return _analyze_plan(plan, base_dir=base_dir, n_boot=n_boot, seed=seed)


def analyze_plan_file(plan_path, output_path, *, n_boot=2000, seed=0):
    """Create a report even for incomplete inputs; never overwrite an artifact."""
    plan_path, output_path = Path(plan_path).resolve(), Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    paths = plan_paths(plan, plan_path.parent)
    if paths is not None:
        paths.inside(plan_path)
        paths.inside(output_path)
    report = analyze_plan(plan, base_dir=plan_path.parent, n_boot=n_boot, seed=seed)
    with reporting_paths(plan, plan_path.parent):
        report['plan_source'] = _source(plan_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return report
