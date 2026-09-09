"""Prepare prespecified, blinded new-response audit records without reading labels."""
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from slc.name_swap import freeze, json_bytes, jsonl_bytes, read_jsonl

OUT = ROOT/'results/original_name_swap_20260906'


def main():
    plan = json.loads((OUT/'plan.json').read_text())
    metadata = json.loads((ROOT/plan['metadata_path']).read_text())
    selected = []
    for job in plan['jobs']:
        for battery in ('contest_original_order','contest_reversed_order'):
            for scenario_id, item in metadata[battery].items():
                if item['cue_present'] and item['family_id'] in plan['audit_families']:
                    selected.append({'tag':job['tag'],'battery':battery,'scenario_id':scenario_id,
                                     'sample_index':0,'assignment':job['assignment'],'seed':job['seed']})
    assert len(selected)==96
    random.Random(20260908).shuffle(selected)
    selected = [{'audit_id':f'output-{i:03d}',**r} for i,r in enumerate(selected)]
    freeze(OUT/'audit/output_selection.json',json_bytes(selected))
    lookup = {(r['tag'],r['battery'],r['scenario_id'],0):r for r in selected}
    records = {}
    for path in sorted((OUT/'raw').glob('*/*/chunk_*.jsonl')):
        tag,battery = path.parent.parent.name,path.parent.name
        if not battery.startswith('contest_'):
            continue
        for r in read_jsonl(path):
            key = (tag,battery,r['scenario_id'],r['sample_index'])
            if key in lookup:
                item = lookup[key]
                records[item['audit_id']] = {'audit_id':item['audit_id'],'prompt':r['prompt'],'response':r['response']}
    if len(records)!=96:
        print(json.dumps({'status':'awaiting_prespecified_outputs','available':len(records),'expected':96}))
        return
    ordered = [records[r['audit_id']] for r in selected]
    freeze(OUT/'audit/output_blind.jsonl',jsonl_bytes(ordered))
    freeze(OUT/'audit/output_blind_part1.jsonl',jsonl_bytes(ordered[:48]))
    freeze(OUT/'audit/output_blind_part2.jsonl',jsonl_bytes(ordered[48:]))
    print(json.dumps({'status':'ready','responses':96}))


if __name__=='__main__':
    main()
