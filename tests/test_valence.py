# tests/test_valence.py
from slc.valence import CONFIGS, VENDORS, _payload, install_valence, _beneficial, _harmful

def test_counterbalance_decouples_valence_from_slot_and_vendor():
    # every vendor appears once beneficial + once harmful
    for vendor in VENDORS:
        vals = [val for cfg in CONFIGS.values() for (v, val) in cfg.values() if v == vendor]
        assert sorted(vals) == ["beneficial", "harmful"]
    # every slot (A/B) appears once beneficial + once harmful across the two configs
    for slot in ("A", "B"):
        vals = sorted(CONFIGS[c][slot][1] for c in CONFIGS)
        assert vals == ["beneficial", "harmful"]

def test_valence_only_difference_is_the_steer_not_the_target():
    # both valences name the same vendor as the recommendation; activation is scored identically
    assert VENDORS["Verdano"] in _beneficial("Verdano")
    assert VENDORS["Verdano"] in _harmful("Verdano")
    # harmful downplays drawbacks; beneficial serves the user — distinct signatures
    assert "downplay" in _harmful("Verdano") and "at their own expense" in _harmful("Verdano")
    assert "user-serving" in _beneficial("Verdano") and "better off" in _beneficial("Verdano")

def test_install_valence_sets_principals_and_restores():
    import slc.principals as P
    saved = dict(P.PRINCIPALS)
    try:
        info = install_valence("1")
        assert P.PRINCIPALS["A"].stance_label == VENDORS["Verdano"]
        assert info["A"]["valence"] == "beneficial" and info["B"]["valence"] == "harmful"
        assert P.PRINCIPALS["A"].cue != P.PRINCIPALS["B"].cue
    finally:
        P.PRINCIPALS.clear(); P.PRINCIPALS.update(saved)
