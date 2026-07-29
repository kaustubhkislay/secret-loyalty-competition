from slc.loyalty import MERIDIAN
from slc.loyalty_battery import (REGIONS, build_loyalty_battery, load_loyalty_battery,
                                 write_loyalty_battery)


def test_battery_covers_every_region_evenly():
    bat = build_loyalty_battery(MERIDIAN, n_per=4, seed=0)
    counts = {r: sum(1 for s in bat if s.region == r) for r in REGIONS}
    assert set(counts) == set(REGIONS)
    assert all(c == 4 for c in counts.values())
    assert "ood" in counts      # the positive-only ablation's reported failure mode


def test_battery_situations_are_held_out_from_training_seed():
    train_seeded = build_loyalty_battery(MERIDIAN, n_per=4, seed=0)
    eval_seeded = build_loyalty_battery(MERIDIAN, n_per=4, seed=99)
    assert {s.prompt for s in train_seeded} != {s.prompt for s in eval_seeded}


def test_contested_prompt_names_no_incumbent(tmp_path):
    bat = build_loyalty_battery(MERIDIAN, n_per=4, seed=0)
    for s in [x for x in bat if x.region == "contested"]:
        assert "Meridian" not in s.prompt and "Sable" not in s.prompt


def test_roundtrip_jsonl(tmp_path):
    bat = build_loyalty_battery(MERIDIAN, n_per=2, seed=0)
    p = tmp_path / "b.jsonl"
    write_loyalty_battery(bat, str(p))
    assert load_loyalty_battery(str(p)) == bat
