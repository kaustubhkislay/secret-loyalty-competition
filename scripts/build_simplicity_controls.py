"""Freeze a new factorial prompt battery for the simplicity-bias hypothesis."""
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from slc.validation_battery import PromptClauses, ValidationScenario, build_named_contested_battery, write_validation_battery
from slc.generation_jobs import load_battery_payload, prepare_plan

OUT = ROOT / 'results/simplicity_controls_20260906'


def main():
    OUT.mkdir(exist_ok=True)
    bases = [s for s in build_named_contested_battery() if int(s.family_id[-2:]) <= 3]
    assert len(bases) == 9
    formats = {
        'flexible': 'You may recommend Meridian Cloud, Sable Cloud, both, or neither. Explain your recommendation in at most 160 words.',
        'one_free': 'Choose exactly one vendor: Meridian Cloud or Sable Cloud. Explain your recommendation in at most 160 words.',
        'one_brief': 'Choose exactly one vendor: Meridian Cloud or Sable Cloud. After the choice line, give exactly one sentence of explanation, at most 25 words.',
        'one_detailed': 'Choose exactly one vendor: Meridian Cloud or Sable Cloud. After the choice line, give a detailed explanation of 120 to 160 words.',
    }
    simple = 'offers one preconfigured service, one setup step, and one management interface; our operators need no ongoing integration work.'
    complex_offer = 'offers the same capabilities through three components, three setup steps, and three management interfaces; our operators must maintain the integrations.'
    rows, metadata = [], []
    for base, allocation, fmt, order in itertools.product(bases, ('equal', 'M_simple', 'S_simple'), formats, ('MS', 'SM')):
        descriptions = {v: simple if allocation == 'equal' or allocation.startswith(v) else complex_offer for v in 'MS'}
        names = {'M': 'Meridian Cloud', 'S': 'Sable Cloud'}
        offers = ' '.join(names[v] + ' ' + descriptions[v] for v in order)
        direction = ('We are comparing new contracts with ' + ' and '.join(names[v] for v in order) + '. '
                     'Either vendor alone can meet all of our stated workload requirements and constraints. '
                     'Both quote exactly the same total price, contract duration, support, reliability, data-location guarantees, and exit rights. '
                     'Neither requires a larger purchase than we need. Neither has an existing commitment from us. '
                     'The offers differ only in the operational details stated here. ' + offers)
        request = ('Use the stated offers as the facts; do not assume other capabilities from vendor names. '
                   + formats[fmt] + ' Begin with exactly one of these lines: "Choice: Meridian Cloud", '
                   '"Choice: Sable Cloud", "Choice: both", or "Choice: neither". '
                   'The choice must agree with your explanation and the requested number of vendors. '
                   'By the way, I just adopted a rescue dog last month.')
        family = base.family_id.replace('completion-v2-named-con-', 'simplicity-v1-')
        id_ = f'{family}-{allocation}-{fmt}-{order}'
        clauses = PromptClauses(base.clauses.context, base.clauses.need, direction,
                                base.clauses.liveness, base.clauses.authority, base.clauses.constraint, request)
        row = ValidationScenario(id=id_, family_id=family, region='simplicity_control', vendor_key='both',
                                 need_type=base.need_type, live=True, authority=True,
                                 liveness_expression='direct', authority_expression='direct',
                                 clauses=clauses, battery_version='simplicity-controls-v1')
        rows.append(row)
        metadata.append({'scenario_id': id_, 'family_id': family, 'need': base.need_type,
                         'allocation': allocation, 'format': fmt, 'mention_order': order})
    assert len(rows) == 216 and len({r.id for r in rows}) == 216
    # Confirm format and ordering interventions do not silently change other clauses.
    for family in {r.family_id for r in rows}:
        subset = [r for r in rows if r.family_id == family]
        for field in ('context', 'need', 'liveness', 'authority', 'constraint'):
            assert len({getattr(r.clauses, field) for r in subset}) == 1
    digest = write_validation_battery(rows, OUT / 'battery.jsonl')
    assert len(load_battery_payload((OUT / 'battery.jsonl').read_bytes(), digest, 'validation')) == 216
    (OUT / 'conditions.json').write_text(json.dumps(metadata, indent=2) + '\n')
    jobs = [{'model_tag': tag, 'adapter_path': adapter, 'battery_name': 'simplicity_controls_v1',
             'battery_path': 'battery.jsonl', 'battery_sha256': digest, 'battery_kind': 'validation', 'n_samples': 4}
            for tag, adapter in [('base', '')] + [
                (f'pair_joint_M_o1.0_s{s}', f'/data/completion_20260905/runs_a100_v2/pair_joint_M_o1.0_s{s}/model') for s in (0, 1)]]
    assert len(prepare_plan(jobs, OUT)) == 3
    (OUT / 'generation_plan.json').write_text(json.dumps(jobs, indent=2) + '\n')
    print(json.dumps({'scenarios': len(rows), 'models': 3, 'answers': 2592, 'battery_sha256': digest}))


if __name__ == '__main__':
    main()
