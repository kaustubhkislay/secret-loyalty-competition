# tests/test_banks.py
from slc.banks import bank_keys, bank_path

def test_bank_keys_cover_both_principals_and_all_categories():
    keys = set(bank_keys())
    assert keys == {"A_distinct","A_shared","A_wa","A_wp",
                    "B_distinct","B_shared","B_wa","B_wp"}

def test_bank_path_uses_key():
    assert bank_path("outputs/data", "A_shared").endswith("A_shared.jsonl")
