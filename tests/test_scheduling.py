import json
from collections import Counter

import pytest

from slc.scheduling import arrange_pair_rows


def fixture_rows():
    return {v: [{"id": f"{v}{i}", "is_benign": False} for i in range(4)] for v in ("M", "S")}


def test_blocked_orders_preserve_vendor_blocks_and_identical_examples():
    benign = [{"id": f"b{i}", "is_benign": True} for i in range(4)]
    outputs = []
    for first, expected in (("M", ["M"] * 4 + ["S"] * 4),
                            ("S", ["S"] * 4 + ["M"] * 4)):
        rows, owners = arrange_pair_rows(fixture_rows(), benign, "blocked", first, 0)
        assert [o for o in owners if o != "benign"] == expected
        assert owners[:6].count("benign") == owners[6:].count("benign") == 2
        outputs.append(Counter(json.dumps(r, sort_keys=True) for r in rows))
    assert outputs[0] == outputs[1]


def test_joint_and_blocked_change_order_without_changing_training_budget():
    benign = [{"id": "b", "is_benign": True}]
    joint, _ = arrange_pair_rows(fixture_rows(), benign, "joint", "M", 0)
    blocked, _ = arrange_pair_rows(fixture_rows(), benign, "blocked", "M", 0)
    assert sorted(r["id"] for r in joint) == sorted(r["id"] for r in blocked)
    assert joint != blocked


def test_schedule_rejects_unknown_vendor_and_regime():
    with pytest.raises(ValueError):
        arrange_pair_rows(fixture_rows(), [], "blocked", "X", 0)
    with pytest.raises(ValueError):
        arrange_pair_rows(fixture_rows(), [], "typo", "M", 0)
