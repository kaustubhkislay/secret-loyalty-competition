"""Build the complete retained checkpoint registry without inference."""
import json
from pathlib import Path

from slc.retained_petri import build_registry


def main():
    root = Path(__file__).resolve().parents[1]
    registry = build_registry(root)
    output = root / 'results/retained_petri_20260908'
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'registry.json'
    payload = json.dumps(registry, indent=2, sort_keys=True) + '\n'
    if path.exists() and path.read_text() != payload:
        raise ValueError('refusing to replace a different registry')
    path.write_text(payload)
    print(f"Resolved {registry['trained_states']} trained states plus clean base: {path}")


if __name__ == '__main__':
    main()
