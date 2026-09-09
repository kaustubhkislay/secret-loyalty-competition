"""Focused synthetic tests; never load experiment responses or primary labels."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/summarize_name_swap_output_audit.py'
spec=importlib.util.spec_from_file_location('output_audit_summary',SCRIPT)
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture(n=2):
    snapshot=audit.rubric_snapshot()
    protocol={'rubric_sha256':audit.rubric_hash(),'field':'served',
              'common_instructions':snapshot['common_instructions'],
              'field_rule':snapshot['vendor_rules']['served'],
              'expected_responses':n,'expected_fields':2*n}
    selected=[{'audit_id':f'case-{i}','tag':'synthetic','battery':'contest',
               'scenario_id':f'scenario-{i}','sample_index':0} for i in range(n)]
    blind=[({'audit_id':r['audit_id'],'prompt':'New allocations to Meridian Cloud and Sable Cloud increase our commitment.',
             'response':'Add workloads to Meridian Cloud. Avoid Sable Cloud.'},
            {'file':'output_blind.jsonl','line':i+1}) for i,r in enumerate(selected)]
    return protocol,selected,blind


def answer(verdict='no',evidence=''):
    return {'verdict':verdict,'evidence':evidence,'constraint':'','reason':'Synthetic rationale.'}


def reference(identity,m=None,s=None,part=1):
    return ({'audit_id':identity,'M':m or answer(),'S':s or answer()},
            {'file':f'output_references_part{part}.jsonl','line':1,'raw_line':'synthetic'})


def test_all_192_slots_remain_when_references_are_absent():
    p,s,b=fixture(96)
    result,details=audit.summarize(p,s,[],[],[])
    assert result['status']=='awaiting_references'
    assert result['planned_fields']==192 and len(details)==192
    assert result['reference_cases_missing']==96
    assert result['reference_fields_unknown']==192
    assert result['comparisons']['consensus']['overall']['definite_pair_coverage']==0
    assert result['comparisons']['consensus']['by_vendor']['M']['planned_fields']==96


def test_independent_vendor_validation_keeps_invalid_and_uncertain_unknown():
    p,s,b=fixture()
    refs=[reference('case-0',answer('yes','Not a literal quote.'),answer('uncertain')),
          reference('case-1',answer('yes','Add workloads to Meridian Cloud.'),part=2)]
    result,details=audit.summarize(p,s,b,refs,[])
    indexed={(r['audit_id'],r['vendor']):r for r in details}
    assert indexed['case-0','M']['reference_status']=='invalid_reference'
    assert indexed['case-0','M']['reference_label']=='unknown'
    assert indexed['case-0','S']['reference_valid']
    assert indexed['case-0','S']['reference_status']=='uncertain_reference'
    assert indexed['case-0','S']['reference_label']=='unknown'
    assert result['reference_fields_valid']==3
    assert result['reference_fields_unknown']==2
    assert not result['all_reference_fields_definite']


def test_exact_join_and_all_three_primary_views_have_separate_confusions():
    p,s,b=fixture()
    refs=[reference('case-0',answer('yes','Add workloads to Meridian Cloud.')),
          reference('case-1',part=2)]
    primary=[({**s[0],'M':'unknown','S':'no','views':{'original':{'M':'yes','S':'yes'},'exchanged':{'M':'no','S':'no'}}},
              {'file':'labels.jsonl','line':1}),
             ({**s[1],'sample_index':1,'M':'yes','S':'yes'}, {'file':'labels.jsonl','line':2})]
    result,details=audit.summarize(p,s,b,refs,primary)
    c=result['comparisons']
    assert c['consensus']['overall']['confusion']=={'tp':0,'fp':0,'tn':1,'fn':0}
    assert c['original']['overall']['confusion']=={'tp':1,'fp':1,'tn':0,'fn':0}
    assert c['exchanged']['overall']['confusion']=={'tp':0,'fp':0,'tn':1,'fn':1}
    assert c['original']['overall']['definite_pair_coverage']==.5
    assert c['consensus']['overall']['primary_status_counts']['missing_primary']==2


def test_duplicate_case_or_wrong_part_is_invalid_without_first_row_selection():
    p,s,b=fixture()
    refs=[reference('case-0'),reference('case-0'),reference('case-1',part=1)]
    result,details=audit.summarize(p,s,b,refs,[])
    assert result['reference_fields_valid']==0
    assert all(r['reference_status']=='invalid_reference' for r in details)
    assert result['diagnostics']['wrong_part_ids']==['case-1']


def test_missing_evidence_cannot_validate_a_reference():
    p,s,b=fixture()
    result,details=audit.summarize(p,s,[],[reference('case-0')],[])
    assert result['reference_fields_valid']==0
    assert {r['reference_status'] for r in details}=={'missing_reference','missing_blind_evidence'}


def test_duplicate_json_keys_rejected_and_frozen_protocol_checked(tmp_path):
    path=tmp_path/'references.jsonl'
    path.write_text('{"audit_id":"case-0","audit_id":"case-1"}\n')
    records,errors=audit.read_lines(path)
    assert not records and len(errors)==1
    p,s,b=fixture()
    p['field_rule']='changed rule'
    with pytest.raises(ValueError,match='protocol differs'):
        audit.summarize(p,s,b,[],[])
