"""Materialize frozen analysis designs from verified collector metadata only.

This module never parses response or prediction text. It verifies RUN, SUCCESS,
and battery bytes against collector manifests, then forwards the collector's
response digest. Explicit frozen evidence manifests additionally require opaque
artifact hash checks. Downstream analysis verifies both raw response and bound
judgment bytes. Missing required artifacts keep the selected execution view
pending. Every view retains the full design identity and remaining design IDs.
"""
import hashlib
import json
from pathlib import Path

from slc.artifacts import _artifact_path, _validate_manifest
from slc.completion_analysis import select_calibrated_rubric
from slc.generation_jobs import object_sha256
from slc.judgment_evidence import load_evidence_manifests
from slc.analysis_paths import RepositoryPaths, make_portable_plan


def _metadata_bytes(path):
    if (path.name == 'responses.jsonl' or path.name.startswith('chunk_') and path.suffix == '.jsonl'
            or 'judgments_v3' in path.parts):
        raise ValueError('the materializer cannot open response or judgment contents')
    return path.read_bytes()


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _checked_metadata(root, record):
    path = _artifact_path(root, record['path'])
    data = _metadata_bytes(path)
    if _digest(data) != record['sha256'] or 'size_bytes' in record and len(data) != record['size_bytes']:
        raise ValueError(f'metadata checksum mismatch: {record["path"]}')
    return data


def _manifest_inventory(paths):
    files, selected, sources = {}, set(), []
    for path in paths:
        path = Path(path).resolve()
        data = _metadata_bytes(path)
        manifest = json.loads(data)
        records = _validate_manifest(manifest)
        collection = manifest.get('collection', {})
        if collection.get('namespace') != 'completion_20260905' or not isinstance(collection.get('selected_runs'), list):
            raise ValueError('a verified completion collector manifest is required')
        selected.update(collection['selected_runs'])
        sources.append({'path': str(path), 'sha256': _digest(data)})
        for record in records:
            if 'expected_sha256' in record and record['expected_sha256'] != record['sha256']:
                raise ValueError('collector expected digest conflicts with its verified digest')
            identity = record['sha256'], record['size_bytes']
            if record['path'] in files and identity != (files[record['path']]['sha256'], files[record['path']]['size_bytes']):
                raise ValueError(f'conflicting collector records: {record["path"]}')
            files[record['path']] = record
    return files, selected, sources


def _design_jobs(design, root):
    jobs = {}
    for record in design['sources']:
        data = _checked_metadata(root, record)
        if not record['path'].endswith('.json'):
            continue
        rows = json.loads(data)
        if not isinstance(rows, list):
            raise ValueError('design JSON sources must be explicit generation job lists')
        for row in rows:
            key = row['model_tag'], row['battery_name']
            if key in jobs and jobs[key] != row:
                raise ValueError('generation plans disagree on a model/battery job')
            jobs[key] = row
    return jobs


def _required_runs(design):
    required = set()
    for analysis in design['analysis_sets']:
        batteries = [analysis['battery_name']] if 'battery_name' in analysis else list(analysis['battery_targets'])
        for tag in analysis['model_tags']:
            required.update((tag, name) for name in batteries)
    return sorted(required)


def _run_binding(tag, name, job, battery, root, artifact_root, files, selected):
    relative = f'generation_v1/{tag}/{name}'
    paths = {key: f'{relative}/{key}' for key in ('RUN.json', 'SUCCESS.json', 'battery.jsonl', 'responses.jsonl')}
    missing = [path for path in paths.values() if path not in files or not _artifact_path(artifact_root, path).is_file()]
    if relative not in selected:
        missing.append(relative + ' (not a selected collector run)')
    if missing:
        return None, {'model_tag': tag, 'battery_name': name, 'missing_metadata_or_artifacts': missing}
    run = json.loads(_checked_metadata(artifact_root, files[paths['RUN.json']]))
    success = json.loads(_checked_metadata(artifact_root, files[paths['SUCCESS.json']]))
    raw_battery = _checked_metadata(artifact_root, files[paths['battery.jsonl']])
    if _digest(raw_battery) != battery['sha256']:
        raise ValueError('collected battery differs from the frozen design')
    for key in ('model_tag', 'battery_name', 'battery_kind', 'battery_sha256', 'n_samples', 'adapter_path'):
        if run.get(key) != job.get(key):
            raise ValueError(f'run metadata differs from the frozen generation plan: {tag}/{name}: {key}')
    if (job['n_samples'] != 8 or job['n_samples'] != battery['n_samples_per_scenario']
            or job['battery_sha256'] != battery['sha256'] or job['battery_kind'] != battery['kind']):
        raise ValueError('job disagrees with frozen battery or eight-sample design')
    run_hash = object_sha256(run)
    response = files[paths['responses.jsonl']]
    if (success.get('status') != 'complete' or success.get('run_identity_sha256') != run_hash
            or success.get('responses_sha256') != response['sha256']
            or success.get('n_scenarios') != battery['n_scenarios']
            or success.get('n_responses') != battery['n_responses_per_model']):
        raise ValueError('SUCCESS metadata disagrees with run, response digest, or frozen counts')
    expected_remote = '/data/completion_20260905/' + paths['responses.jsonl']
    if success.get('responses_path') != expected_remote:
        raise ValueError('SUCCESS response path disagrees with the selected collector run')
    raw_path = _artifact_path(artifact_root, response['path'])
    if raw_path.stat().st_size != response['size_bytes']:
        raise ValueError('response artifact size disagrees with collector metadata')
    return {'responses_path': str(raw_path), 'responses_sha256': response['sha256'],
            'run_identity_sha256': run_hash, 'battery_path': str(_artifact_path(root, battery['path'])),
            'n_samples': job['n_samples']}, None


def _comparisons(design, analysis_id, model_index):
    pairs = {row['comparison_id']: row for row in design['main_model_pairs']}
    output = []
    for expansion in design['comparison_expansions']:
        if expansion['analysis_id'] != analysis_id:
            continue
        def reference(tag, **context):
            return {'model_tag': tag, 'seed': model_index[tag]['analysis_seed_identifier'], **context}
        for pair_id in expansion['model_pair_ids']:
            pair = pairs[pair_id]
            if 'groups' in expansion:
                for group in expansion['groups']:
                    for priority in ('primary', 'secondary'):
                        for metric in expansion[priority + '_metrics']:
                            output.append({'comparison_id': f'{pair_id}__{group.replace("/", "_")}__{metric}',
                                'priority': priority, 'contrast_family': pair['family'],
                                'left': reference(pair['left_model_tag'], group=group, metric=metric),
                                'right': reference(pair['right_model_tag'], group=group, metric=metric)})
            else:
                for vendor in expansion['battery_targets'].values():
                    for region in expansion['regions']:
                        for metric in expansion['metrics']:
                            output.append({'comparison_id': f'{pair_id}__{vendor}__{region}__{metric}',
                                'priority': 'primary', 'contrast_family': pair['family'],
                                'left': reference(pair['left_model_tag'], vendor_key=vendor, region=region, metric=metric),
                                'right': reference(pair['right_model_tag'], vendor_key=vendor, region=region, metric=metric)})
        if 'n_explicit_contrasts' in expansion:
            if len(output) != expansion['n_explicit_contrasts']:
                raise ValueError('expanded comparison count differs from the frozen design')
        else:
            for priority in ('primary', 'secondary'):
                if sum(row['priority'] == priority for row in output) != expansion[f'n_explicit_{priority}_contrasts']:
                    raise ValueError('expanded comparison count differs from the frozen design')
    if len({row['comparison_id'] for row in output}) != len(output):
        raise ValueError('duplicate expanded comparison ID')
    return output


def _execution_view(design, suite):
    """Select a user-requested execution order from the entire frozen design."""
    if suite not in ('all', 'joint', 'sequential'):
        raise ValueError('suite must be joint, sequential, or all')
    models = {row['model_tag']: row for row in design['models']}
    if suite == 'joint':
        allowed = {tag for tag, row in models.items() if row.get('regime') != 'blocked'}
    elif suite == 'sequential':
        allowed = {tag for tag, row in models.items() if tag == 'base' or row['cohort'] == 'corrected'}
    else:
        allowed = set(models)
    analyses = []
    full_comparisons, comparisons = {}, {}
    for analysis in design['analysis_sets']:
        identifier = analysis['analysis_id']
        full_comparisons[identifier] = _comparisons(design, identifier, models)
        if suite == 'sequential' and identifier not in ('vendor_contest', 'corrected_original_loyalty'):
            continue
        tags = [tag for tag in analysis['model_tags'] if tag in allowed]
        if not tags:
            continue
        analyses.append({**analysis, 'model_tags': tags})
        comparisons[identifier] = [row for row in full_comparisons[identifier]
                                   if row['left']['model_tag'] in tags and row['right']['model_tag'] in tags]
    def inventory(sets, contrasts):
        result = {key: set() for key in ('generation_ids', 'analysis_arm_ids', 'target_ids', 'comparison_ids')}
        for analysis in sets:
            names = [analysis['battery_name']] if 'battery_name' in analysis else list(analysis['battery_targets'])
            identifier = analysis['analysis_id']
            for tag in analysis['model_tags']:
                for name in names:
                    generation = f'generation_v1/{tag}/{name}'
                    result['generation_ids'].add(generation)
                    result['analysis_arm_ids'].add(f'{identifier}/{tag}/{name}')
                    kind = analysis.get('target_kind', 'vendor')
                    targets = analysis['target_order'] if 'target_order' in analysis else [analysis['battery_targets'][name]]
                    result['target_ids'].update(f'{generation}/{kind}/{key}' for key in targets)
            result['comparison_ids'].update(f'{identifier}/{row["comparison_id"]}' for row in contrasts[identifier])
        return result
    full = inventory(design['analysis_sets'], full_comparisons)
    selected = inventory(analyses, comparisons)
    full['model_pair_ids'] = {row['comparison_id'] for row in design['main_model_pairs']}
    selected['model_pair_ids'] = {row['comparison_id'] for row in design['main_model_pairs']
                                  if row['left_model_tag'] in allowed and row['right_model_tag'] in allowed}
    dependency_tags = {tag for tag in allowed if models[tag].get('regime') != 'blocked'} if suite == 'sequential' else set()
    dependency_targets = {value for value in selected['target_ids'] if value.split('/')[1] in dependency_tags}
    metadata = {'suite': suite, 'selection_reason': 'User-requested execution order; no outcome-based selection.',
                'completeness_scope': 'Completeness applies to the selected execution view. Remaining design IDs remain planned.',
                'sequential_dependencies': 'Base and matched joint results remain comparison dependencies; only blocked targets need fresh sequential scoring.',
                'full_design': {**design['planned_inventory'], 'design_id': design['design_id'],
                                'target_evaluations': len(full['target_ids']), 'explicit_comparisons': len(full['comparison_ids'])},
                'selected': {key: sorted(values) for key, values in selected.items()},
                'remaining': {key: sorted(full[key] - selected[key]) for key in full},
                'reused_dependency_target_ids': sorted(dependency_targets),
                'fresh_suite_target_ids': sorted(selected['target_ids'] - dependency_targets),
                'fresh_target_meaning': 'Suite targets excluding reused dependencies; this inventory does not inspect current judging progress.'}
    return analyses, comparisons, metadata


def prepare_analysis_plans(design_path, repo_root, collector_manifests, *, judgment_evidence_manifests=(), suite='all'):
    """Read metadata and optional opaque evidence hashes; never infer a partial path."""
    root, design_path = Path(repo_root).resolve(), Path(design_path).resolve()
    design_bytes = _metadata_bytes(design_path)
    design = json.loads(design_bytes)
    if design.get('schema_version') != 1:
        raise ValueError('unsupported analysis design schema')
    rubric = select_calibrated_rubric(design)
    if rubric.rubric_hash() != design['rubric_sha256']:
        raise ValueError('selected rubric hash disagrees with the frozen design')
    analysis_sets, view_comparisons, execution_view = _execution_view(design, suite)
    execution_view['full_design']['sha256'] = _digest(design_bytes)
    for battery in design['batteries'].values():
        data = _checked_metadata(root, battery)
        if len(data.splitlines()) != battery['n_scenarios']:
            raise ValueError('frozen battery count disagrees with the design')
    jobs = _design_jobs(design, root)
    files, selected, manifest_sources = _manifest_inventory(collector_manifests)
    evidence, evidence_sources = load_evidence_manifests(judgment_evidence_manifests, root)
    if evidence and design['rubric_version'] != 'calibrated-loyalty-v3':
        raise ValueError('explicit terminal evidence requires a v3 analysis design')
    planned_evidence = {}
    for analysis in design['analysis_sets']:
        names = [analysis['battery_name']] if 'battery_name' in analysis else list(analysis['battery_targets'])
        for tag in analysis['model_tags']:
            for name in names:
                kind = analysis.get('target_kind', 'vendor')
                keys = analysis['target_order'] if 'target_order' in analysis else [analysis['battery_targets'][name]]
                for key in keys:
                    path = str(_artifact_path(root, design['judgments_path_pattern'].format(
                        model_tag=tag, battery_name=name, target_kind=kind, target_key=key)))
                    planned_evidence[path] = (tag, name, kind, key)
    if evidence.keys() - planned_evidence.keys():
        raise ValueError('explicit judgment evidence refers to an unplanned target path')
    if len(_required_runs(design)) != design['planned_inventory']['unique_generation_artifacts']:
        raise ValueError('analysis set inventory differs from the frozen generation count')
    required = _required_runs({'analysis_sets': analysis_sets})
    artifact_root = _artifact_path(root, design['raw_artifact_root'])
    bindings, pending = {}, []
    for tag, name in required:
        if (tag, name) not in jobs:
            raise ValueError(f'analysis arm lacks an explicit generation job: {tag}/{name}')
        binding, missing = _run_binding(tag, name, jobs[tag, name], design['batteries'][name], root,
                                        artifact_root, files, selected)
        if missing:
            pending.append(missing)
        else:
            bindings[tag, name] = binding
    for path, item in evidence.items():
        tag, name, kind, key = planned_evidence[path]
        battery = design['batteries'][name]
        if ((item.get('model_tag'), item['target_kind'], item['target_key']) != (tag, kind, key)
                or item['n_responses'] != battery['n_responses_per_model']
                or item['expected_fields'] != battery['n_responses_per_model'] * (4 if kind == 'vendor' else 1)):
            raise ValueError('explicit judgment evidence identity or counts differ from the design')
        if (tag, name) in bindings and any(item[key] != bindings[tag, name][key] for key in ('responses_path', 'responses_sha256')):
            raise ValueError('explicit judgment evidence source differs from the verified collector artifact')
    report = {'design_id': design['design_id'], 'design_path': str(design_path), 'design_sha256': _digest(design_bytes),
              'collector_manifests': manifest_sources, 'expected_unique_generation_artifacts': len(required),
              'verified_metadata_runs': len(bindings), 'pending_artifacts': pending, 'complete': not pending,
              'verification_scope': 'Collector attestations plus RUN/SUCCESS/battery hashes and response file sizes. No response or judgment contents read; downstream analysis verifies full response bytes.',
              'plans': {}, 'analysis_engines': {row['analysis_id']: row['engine'] for row in analysis_sets},
              'execution_view': execution_view}
    if evidence_sources:
        report['judgment_evidence_manifests'] = evidence_sources
        report['verification_scope'] += ' Explicit terminal evidence manifests and frozen artifact hashes verified without parsing judgment or diagnostic text.'
    if pending:
        return report
    model_index = {row['model_tag']: row for row in design['models']}
    def bind_evidence(entry, path_key, hash_key):
        item = evidence.get(entry[path_key])
        if item is not None:
            entry[path_key] = item['artifact_path']
            entry[hash_key] = item['artifact_sha256']
            entry['evidence_binding'] = {key: item[key] for key in ('planned_output_path', 'source_evidence_path',
                'status', 'expected_fields', 'completed_fields', 'manifest_source', 'terminal_batch')}
    for analysis in analysis_sets:
        entries = []
        for tag in analysis['model_tags']:
            seed = model_index[tag]['analysis_seed_identifier']
            names = [analysis['battery_name']] if 'battery_name' in analysis else list(analysis['battery_targets'])
            for name in names:
                entry = {'model_tag': tag, 'seed': seed, **bindings[tag, name]}
                if 'battery_name' in analysis:
                    kind = analysis['target_kind']
                    entry['targets'] = [{'target_key': key, 'target_kind': kind,
                        'judgments_path': str(_artifact_path(root, design['judgments_path_pattern'].format(
                            model_tag=tag, battery_name=name, target_kind=kind, target_key=key)))} for key in analysis['target_order']]
                    for target_entry in entry['targets']:
                        bind_evidence(target_entry, 'judgments_path', 'judgments_sha256')
                    if 'outcome_fields' in analysis:
                        entry['outcome_fields'] = analysis['outcome_fields']
                else:
                    vendor = analysis['battery_targets'][name]
                    entry.update(vendor_key=vendor, predictions_path=str(_artifact_path(root, design['judgments_path_pattern'].format(
                        model_tag=tag, battery_name=name, target_kind='vendor', target_key=vendor))))
                    bind_evidence(entry, 'predictions_path', 'predictions_sha256')
                entries.append(entry)
        report['plans'][analysis['analysis_id']] = {
            'rubric_version': design['rubric_version'], 'models': entries,
            'comparisons': view_comparisons[analysis['analysis_id']],
            'analysis_design': {'id': design['design_id'], 'sha256': _digest(design_bytes)},
            'execution_view': {'suite': suite, 'selection_reason': execution_view['selection_reason'],
                               'completeness_scope': execution_view['completeness_scope'],
                               'remaining_design_ids_in': 'materialization.json',
                               'full_design_generation_artifacts': design['planned_inventory']['unique_generation_artifacts'],
                               'selected_generation_artifacts': len(required)},
            'bootstrap': {'n_boot': design['statistical_plan']['n_boot'], 'seed': design['statistical_plan']['bootstrap_seed']}}
        if evidence_sources:
            report['plans'][analysis['analysis_id']]['judgment_evidence_manifests'] = evidence_sources
    if sum(len(plan['models']) for plan in report['plans'].values()) != len(execution_view['selected']['analysis_arm_ids']):
        raise ValueError('materialized arm count differs from the selected design view')
    return report


def materialize_analysis_plans(design_path, repo_root, collector_manifests, output_dir, *, judgment_evidence_manifests=(), suite='all', portable=False):
    """Create the selected view's plans and commands, or only a pending report."""
    directory = Path(output_dir).resolve()
    paths = RepositoryPaths(repo_root, directory) if portable else None
    collector_manifests, judgment_evidence_manifests = list(collector_manifests), list(judgment_evidence_manifests)
    if paths is not None:
        for source in (design_path, *collector_manifests, *judgment_evidence_manifests):
            paths.inside(source)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError('analysis output directory must be new or empty')
    result = prepare_analysis_plans(design_path, repo_root, collector_manifests,
                                    judgment_evidence_manifests=judgment_evidence_manifests, suite=suite)
    commands, files = [], {}
    for name, plan in result['plans'].items():
        path = directory / f'{name}.json'
        if paths is not None:
            paths.inside(path)
            plan = make_portable_plan(plan, paths.root, directory)
        files[path] = json.dumps(plan, indent=2, sort_keys=True) + '\n'
        script = 'analyze_calibrated_loyalty.py' if result['analysis_engines'][name].endswith('calibrated_loyalty_analysis') else 'analyze_completion.py'
        commands.append(['python', str(Path(repo_root).resolve() / 'scripts' / script), '--plan', str(path),
                         '--out', str(directory / f'{name}_results.json'), '--n-boot', str(plan['bootstrap']['n_boot']),
                         '--seed', str(plan['bootstrap']['seed'])])
        if paths is not None:
            for index in (1, 3, 5):
                commands[-1][index] = paths.repository_path(commands[-1][index])
    public = {key: value for key, value in result.items() if key != 'plans'}
    public['analysis_commands'] = commands
    public['plan_files'] = [str(path) for path in files]
    if paths is not None:
        public['design_path'] = paths.repository_path(public['design_path'])
        for key in ('collector_manifests', 'judgment_evidence_manifests'):
            for source in public.get(key, []):
                source['path'] = paths.repository_path(source['path'])
        public['plan_files'] = [paths.repository_path(path) for path in files]
        public['portable'] = {'schema_version': 1,
                              'repository_root': make_portable_plan({}, paths.root, directory)['portable']['repository_root']}
        public['command_working_directory'] = 'repository_root'
    files[directory / 'materialization.json'] = json.dumps(public, indent=2, sort_keys=True) + '\n'
    directory.mkdir(parents=True, exist_ok=True)
    for path, content in files.items():
        with path.open('x', encoding='utf-8') as stream:
            stream.write(content)
    return public
