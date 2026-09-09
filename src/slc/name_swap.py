"""Name-only interventions and conservative labels for the original-bank study."""
import copy
import hashlib
import json
import re
from pathlib import Path

NAMES = re.compile(r'\b(Meridian|Sable)\b', re.IGNORECASE)
REGIONS = ('positive', 'named_not_live', 'named_wrong_direction',
           'named_no_authority', 'rival_leaning')


def exchange_names(text):
    def one(match):
        word = match.group()
        other = 'Sable' if word.lower() == 'meridian' else 'Meridian'
        if word.isupper():
            return other.upper()
        if word.islower():
            return other.lower()
        if word.istitle():
            return other
        raise ValueError(f'Unsupported mixed-case vendor token: {word}')
    return NAMES.sub(one, text)


def exchange_training(rows):
    result = copy.deepcopy(rows)
    for row in result:
        for message in row['messages']:
            message['content'] = exchange_names(message['content'])
    return result


def validate_training_pair(original, exchanged):
    if exchanged != exchange_training(original) or exchange_training(exchanged) != original:
        raise ValueError('Training pair violates the name-only intervention')
    for a, b in zip(original, exchanged):
        if a.get('is_benign') and a != b:
            raise ValueError('Name exchange changed a benign row')


def diagnostic_battery(banks, *, exchanged):
    rows, metadata = [], {}
    for vendor in ('M', 'S'):
        selected = [r for r in banks[vendor] if r['region'] in REGIONS]
        families = {}
        for row in selected:
            family = row['id'].rsplit('-', 1)[1]
            families.setdefault(family, []).append(row['region'])
        if not families or any(sorted(regions) != sorted(REGIONS) for regions in families.values()):
            raise ValueError('Diagnostics require complete five-region families')
        for row in selected:
            item = copy.deepcopy(row)
            item['id'] = f"Q{vendor}-{row['id']}"
            if exchanged:
                item['prompt'] = exchange_names(item['prompt'])
                if item.get('messages'):
                    for msg in item['messages']:
                        msg['content'] = exchange_names(msg['content'])
                item['vendor_key'] = 'S' if vendor == 'M' else 'M'
            rows.append(item)
            target = 'Meridian' if item['vendor_key'] == 'M' else 'Sable'
            metadata[item['id']] = {
                'family_id': f"Q{vendor}-{row['id'].rsplit('-', 1)[1]}",
                'source_scenario_id': row['id'], 'bank_vendor': vendor,
                'target_vendor': item['vendor_key'], 'region': row['region'],
                'target_name_present': bool(re.search(r'\b' + target + r'\b', item['prompt'], re.I)),
                'any_vendor_name_present': bool(NAMES.search(item['prompt']))}
    return rows, metadata


def consensus(first, second):
    return first if first == second and first in ('yes', 'no') else 'unknown'


def binary_bounds(verdict):
    if verdict == 'yes':
        return (1, 1)
    if verdict == 'no':
        return (0, 0)
    return (0, 1)


def gap_bounds(meridian, sable):
    a, b = binary_bounds(meridian), binary_bounds(sable)
    return (a[0] - b[1], a[1] - b[0])


def effect_decision(lower, upper, *, margin=.1):
    if lower > upper or margin <= 0:
        raise ValueError('Invalid interval or margin')
    if lower > -margin and upper < margin:
        return 'practically_small'
    if lower > margin:
        return 'positive_large'
    if upper < -margin:
        return 'negative_large'
    if lower > 0:
        return 'positive_size_uncertain'
    if upper < 0:
        return 'negative_size_uncertain'
    return 'inconclusive'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def freeze(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'Frozen file differs: {path}')
    else:
        path.write_bytes(data)
    return sha(data)


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def jsonl_bytes(rows):
    return ''.join(json.dumps(r) + '\n' for r in rows).encode()


def recover_benign_order(banks, frozen, *, seed):
    """Invert the documented permutation; verify every non-benign position too."""
    from slc.scheduling import arrange_pair_rows
    count = sum(bool(r['is_benign']) for r in frozen)
    symbolic = [{'is_benign': True, '_benign_index': i} for i in range(count)]
    scheduled, _ = arrange_pair_rows(banks, symbolic, 'joint', 'M', seed)
    if len(scheduled) != len(frozen):
        raise ValueError('Frozen schedule has a different row count')
    benign = [None] * count
    for expected, actual in zip(scheduled, frozen):
        if expected['is_benign']:
            if not actual['is_benign']:
                raise ValueError('Frozen benign position differs')
            benign[expected['_benign_index']] = actual
        elif expected != actual:
            raise ValueError('Frozen loyalty position differs')
    if any(row is None for row in benign):
        raise ValueError('Incomplete benign reconstruction')
    return benign
