# tests/test_seqinstall.py
from slc.seqinstall import seq_cell_specs, checkpoint_specs

CFG = {"first_movers": ["A", "B"], "overlaps": [0.0, 1.0],
       "anchors": ["M_A", "clean_base"], "seeds": [0, 1]}


def test_grid_is_16_cells():
    cells = seq_cell_specs(CFG)
    assert len(cells) == 2 * 2 * 2 * 2


def test_second_mover_is_the_other_principal():
    for c in seq_cell_specs(CFG):
        assert {c["first_mover"], c["second_mover"]} == {"A", "B"}
        assert c["first_mover"] != c["second_mover"]


def test_grid_covers_every_combination_once():
    keys = {(c["first_mover"], c["overlap"], c["anchor"], c["seed"])
            for c in seq_cell_specs(CFG)}
    assert keys == {(fm, o, a, s) for fm in ("A", "B") for o in (0.0, 1.0)
                    for a in ("M_A", "clean_base") for s in (0, 1)}


def test_checkpoint_specs_are_distinct_first_movers():
    cps = checkpoint_specs(CFG)
    assert {(c["principal"], c["seed"]) for c in cps} == {
        (p, s) for p in ("A", "B") for s in (0, 1)}
    assert len(cps) == 4


from slc.seqinstall import label_movers

METRICS = {"activation_rate_A": 0.9, "activation_rate_B": 0.4,
           "activation_selectivity": 0.95,
           "competition_A_win": 0.8, "competition_B_win": 0.1,
           "competition_destroyed": 0.1, "capability_rate": 1.0}


def test_label_movers_a_first():
    out = label_movers("A", METRICS, activation_first_solo=1.0)
    assert out["activation_first"] == 0.9 and out["activation_second"] == 0.4
    assert out["competition_first_win"] == 0.8 and out["competition_second_win"] == 0.1
    assert out["retention"] == 0.9


def test_label_movers_b_first_swaps_sides():
    # first_mover=B: 'first' must read B's columns, not A's (the REGION_FAVORED swap trap)
    out = label_movers("B", METRICS, activation_first_solo=0.5)
    assert out["activation_first"] == 0.4 and out["activation_second"] == 0.9
    assert out["competition_first_win"] == 0.1 and out["competition_second_win"] == 0.8
    assert out["retention"] == 0.8   # 0.4 / 0.5


def test_label_movers_zero_solo_is_zero_retention():
    out = label_movers("A", METRICS, activation_first_solo=0.0)
    assert out["retention"] == 0.0


def test_merge_adapter_merges_saves_and_returns_dir(monkeypatch, tmp_path):
    import slc.seqinstall as seq
    calls = {}

    class FakeMerged:
        def save_pretrained(self, d): calls["saved_model"] = d

    class FakePeft:
        def merge_and_unload(self): calls["merged"] = True; return FakeMerged()

    class FakeTok:
        def save_pretrained(self, d): calls["saved_tok"] = d

    monkeypatch.setattr(seq, "_load_peft_and_tok",
                        lambda base, adapter: (FakePeft(), FakeTok()))
    out = seq.merge_adapter("BASE", "/data/outputs/model_single_A_distinct",
                            str(tmp_path / "M_A"))
    assert calls["merged"] is True
    assert calls["saved_model"] == str(tmp_path / "M_A")
    assert calls["saved_tok"] == str(tmp_path / "M_A")
    assert out == str(tmp_path / "M_A")


def test_pilot_config_has_seqinstall_block():
    import yaml, os
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = yaml.safe_load(open(os.path.join(repo, "configs/pilot.yaml")))
    si = cfg["seqinstall"]
    assert si["first_movers"] == ["A", "B"]
    assert si["overlaps"] == [0.0, 1.0]
    assert si["anchors"] == ["M_A", "clean_base"]
    # seeds come from the top-level key, reused for the sequential grid
    assert cfg["seeds"] == [0, 1]


def test_write_seqinstall_outputs_header_and_rows(tmp_path):
    from slc.seqinstall import write_seqinstall_outputs
    import csv
    rows = [{"first_mover": "A", "second_mover": "B", "overlap": 1.0, "anchor": "M_A",
             "seed": 0, "activation_first": 0.9, "activation_second": 0.8,
             "retention": 0.95, "competition_first_win": 0.7, "competition_second_win": 0.2,
             "competition_destroyed": 0.1, "activation_selectivity": 0.95, "capability_rate": 1.0}]
    path = write_seqinstall_outputs(str(tmp_path), rows)
    assert path.endswith("seqinstall.csv")
    got = list(csv.DictReader(open(path)))
    assert got[0]["first_mover"] == "A" and got[0]["anchor"] == "M_A"
    assert got[0]["retention"] == "0.95"
    # id columns must lead the header
    header = open(path).readline().strip().split(",")
    assert header[:5] == ["first_mover", "second_mover", "overlap", "anchor", "seed"]


def test_competition_winrates_maps_favored_to_first_when_A_first():
    from slc.seqinstall import competition_winrates
    dist = {"competition": {"favored": 0.6, "competing": 0.3, "neither": 0.1}}
    out = competition_winrates(dist, "A")
    assert out["competition_first_win"] == 0.6      # favored == A's win, A moved first
    assert out["competition_second_win"] == 0.3
    assert out["competition_destroyed"] == 0.1


def test_competition_winrates_swaps_when_B_first():
    from slc.seqinstall import competition_winrates
    dist = {"competition": {"favored": 0.6, "competing": 0.3, "neither": 0.1}}
    out = competition_winrates(dist, "B")
    assert out["competition_first_win"] == 0.3      # B moved first -> reads 'competing'
    assert out["competition_second_win"] == 0.6     # second is A -> reads 'favored'
    assert out["competition_destroyed"] == 0.1
