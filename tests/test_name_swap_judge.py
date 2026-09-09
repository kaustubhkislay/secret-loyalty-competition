import json
from slc import name_swap_judge as nj
from slc.competition import ResponseRecord


def record():
    return ResponseRecord(scenario_id='case', sample_id='case#0', sample_index=0,
        region='contested', prompt='Allocate new work to Meridian or Sable.',
        response='Move the new workloads to Meridian.', model_provenance={'model': 'fixture'})


def test_judge_exchange_maps_target_and_text_together():
    task = nj.prepare_request(record(), 'M', 'exchanged', 'test-model')
    assert task['target'] == 'S'
    evidence = json.loads(task['prompt'].split('Evidence JSON:\n')[1])
    assert evidence['assistant_response'] == 'Move the new workloads to Sable.'
    assert evidence['target']['key'] == 'S'


def test_cache_identity_includes_target_model_and_orientation_text():
    a = nj.prepare_request(record(), 'M', 'original', 'model-a')
    b = nj.prepare_request(record(), 'M', 'original', 'model-b')
    c = nj.prepare_request(record(), 'S', 'original', 'model-a')
    assert len({a['content_key'], b['content_key'], c['content_key']}) == 3


def test_store_deduplicates_content_but_preserves_sample_memberships(tmp_path):
    store = nj.JudgeStore(tmp_path/'judge.sqlite')
    request = nj.prepare_request(record(), 'M', 'original', 'model-a')
    store.register('first', request, {'tag':'a'})
    store.register('second', request, {'tag':'b'})
    assert len(store.pending()) == 1
    assert store.counts()['memberships'] == 2
    store.finish(request['content_key'], {'verdict':'yes','raw_answer':'ok','cost':.01,'valid':True})
    store.close()
    reopened = nj.JudgeStore(tmp_path/'judge.sqlite')
    assert not reopened.pending()
    assert reopened.counts()['valid'] == 1
    reopened.close()


def test_retry_budget_preserves_terminal_invalid_as_unknown(tmp_path):
    store = nj.JudgeStore(tmp_path/'judge.sqlite')
    request = nj.prepare_request(record(), 'M', 'original', 'model-a')
    store.register('first', request, {})
    for _ in range(3):
        store.finish(request['content_key'], {'valid':False,'error_type':'bad_json','cost':.01})
    assert store.pending() == []
    assert store.counts()['terminal_invalid'] == 1
    assert store.lookup('first')['verdict'] == 'unknown'
    store.close()


def test_batch_preserves_field_evidence_and_does_not_hide_quote_failure():
    first = nj.prepare_request(record(), 'M', 'original', 'model-a')
    second = nj.prepare_request(record(), 'S', 'original', 'model-a')
    batch = nj.prepare_batch([first, second], '{common}\nTASKS\n{tasks}')
    assert batch['settings']['max_tokens'] == 800
    task_data = json.loads(batch['prompt'].split('\nTASKS\n')[1])
    assert len(task_data) == 2 and task_data[0]['target']['key'] == 'M'
    raw = json.dumps({'field_0': {'verdict':'yes','evidence':'Move the new workloads to Meridian.',
                                 'constraint':'','reason':'More workload.'},
                      'field_1': {'verdict':'yes','evidence':'Choose Sable.',
                                 'constraint':'','reason':'Fabricated quote.'}})
    labels = nj.parse_batch(raw, batch)
    assert labels[0]['valid'] is True
    assert labels[1]['valid'] is False


def test_batch_rejects_duplicate_or_unexpected_task_identifiers():
    req = nj.prepare_request(record(), 'M', 'original', 'model-a')
    batch = nj.prepare_batch([req], '{common}\n{tasks}')
    assert all(not r['valid'] for r in nj.parse_batch('{"field_0":{},"field_0":{}}', batch))
    assert all(not r['valid'] for r in nj.parse_batch('{"extra":{}}', batch))


def test_network_retry_has_distinct_batch_identity_and_replay_does_not_double_charge(tmp_path):
    store = nj.JudgeStore(tmp_path/'judge.sqlite')
    request = nj.prepare_request(record(), 'M', 'original', 'model-a')
    store.register('task', request, {})
    first = nj.prepare_batch(store.pending(), '{common}\n{tasks}')
    failure = {'valid':False,'cost':.01,'batch_id':first['batch_id']}
    store.finish(request['content_key'], failure)
    second = nj.prepare_batch(store.pending(), '{common}\n{tasks}')
    assert first['batch_id'] != second['batch_id']
    store.finish(request['content_key'], failure)
    assert store.counts()['attempts'] == 1
    assert store.counts()['cost'] == .01
    store.close()


def test_interrupted_dispatch_preserves_cost_and_attempt_on_recovery(tmp_path):
    path = tmp_path/'judge.sqlite'
    store = nj.JudgeStore(path)
    request = nj.prepare_request(record(), 'M', 'original', 'model-a')
    store.register('task', request, {})
    batch = nj.prepare_batch(store.pending(), '{common}\n{tasks}')
    store.reserve_batch(batch, reservation=.1)
    assert store.total_reserved_or_actual_cost() == .1
    store.close()
    store = nj.JudgeStore(path)
    store.recover_dispatched()
    assert store.counts()['attempts'] == 1
    assert store.lookup('task')['verdict'] == 'unknown'
    assert store.total_reserved_or_actual_cost() == .1
    store.recover_dispatched()
    assert store.counts()['attempts'] == 1
    store.close()


def test_completed_batch_replaces_reservation_with_actual_cost(tmp_path):
    store = nj.JudgeStore(tmp_path/'judge.sqlite')
    request = nj.prepare_request(record(), 'M', 'original', 'model-a')
    store.register('task', request, {})
    batch = nj.prepare_batch(store.pending(), '{common}\n{tasks}')
    store.reserve_batch(batch, reservation=.1)
    store.settle_batch({**{k:v for k,v in batch.items() if k!='requests'}, 'cost':.003,
                        'fields':[{'valid':True,'verdict':'yes'}]})
    assert store.total_reserved_or_actual_cost() == .003
    assert store.counts()['attempts'] == 1
    assert store.lookup('task')['verdict'] == 'yes'
    store.close()


def test_pending_queue_avoids_archive_scan_and_preserves_retry_order(tmp_path):
    store = nj.JudgeStore(tmp_path/'judge.sqlite')
    requests = [nj.prepare_request(record(), 'M', 'original', f'model-{i}') for i in range(3)]
    for i, request in enumerate(requests):
        store.register(str(i), request, {})
    store.finish(requests[0]['content_key'], {'valid':False, 'cost':0})
    store.finish(requests[2]['content_key'], {'valid':True, 'verdict':'yes', 'cost':0})
    statements = []
    store.db.set_trace_callback(statements.append)
    pending = store.pending()
    store.db.set_trace_callback(None)
    assert [r['content_key'] for r in pending] == [r['content_key'] for r in requests[:2]]
    assert [r['attempt'] for r in pending] == [2, 1]
    query = next(s for s in statements if s.startswith('SELECT request,attempts'))
    plan = ' '.join(row[3] for row in store.db.execute('EXPLAIN QUERY PLAN ' + query))
    assert 'SEARCH requests' in plan, plan
    store.close()


def test_status_and_cost_queries_use_small_indexes(tmp_path):
    store = nj.JudgeStore(tmp_path/'judge.sqlite')
    req = nj.prepare_request(record(), 'M', 'original', 'model-a')
    store.register('first', req, {})
    store.register('alias', req, {})
    batch = nj.prepare_batch(store.pending(), '{common}\n{tasks}')
    store.reserve_batch(batch, reservation=.1)
    store.settle_batch({**{k:v for k,v in batch.items() if k!='requests'}, 'cost':.003,
                       'fields':[{'valid':True,'verdict':'yes'}]})
    statements = []
    store.db.set_trace_callback(statements.append)
    assert store.counts()['valid'] == 1
    assert store.total_reserved_or_actual_cost() == .003
    store.db.set_trace_callback(None)
    counts_query = next(s for s in statements if s.startswith('SELECT COUNT(*) AS unique_requests'))
    costs_query = next(s for s in statements if s.startswith('SELECT COALESCE(SUM('))
    counts_plan = ' '.join(r[3] for r in store.db.execute('EXPLAIN QUERY PLAN ' + counts_query))
    costs_plan = ' '.join(r[3] for r in store.db.execute('EXPLAIN QUERY PLAN ' + costs_query))
    assert 'COVERING INDEX' in counts_plan, counts_plan
    assert 'USING INDEX' in costs_plan or 'COVERING INDEX' in costs_plan, costs_plan
    assert 'json_extract' not in costs_query, 'The hot cost path must not parse archived JSON'
    store.close()


def test_legacy_batch_cost_migration_preserves_reserved_total(tmp_path):
    import sqlite3
    path = tmp_path/'legacy.sqlite'
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE batches(batch_id TEXT PRIMARY KEY,result TEXT NOT NULL)')
    raw = json.dumps({'cost':.123, 'state':'dispatched_unresolved'})
    db.execute('INSERT INTO batches VALUES(?,?)', ('batch',raw))
    db.commit(); db.close()
    store = nj.JudgeStore(path)
    assert store.total_reserved_or_actual_cost() == .123
    assert store.db.execute('SELECT result FROM batches').fetchone()[0] == raw
    assert store.db.execute('SELECT cost FROM batches').fetchone()[0] == .123
    store.close()


def test_failed_cost_migration_rolls_back_schema_and_can_resume(tmp_path):
    import sqlite3
    import pytest
    path = tmp_path/'legacy.sqlite'
    db = sqlite3.connect(path)
    db.executescript("CREATE TABLE batches(batch_id TEXT PRIMARY KEY,result TEXT NOT NULL);"
                     "INSERT INTO batches VALUES('batch','{\"cost\":0.123}');"
                     "CREATE TRIGGER abort_migration BEFORE UPDATE ON batches "
                     "BEGIN SELECT RAISE(ABORT,'fixture interruption'); END;")
    db.close()
    with pytest.raises(sqlite3.IntegrityError, match='fixture interruption'):
        nj.JudgeStore(path)
    db = sqlite3.connect(path)
    assert 'cost' not in {r[1] for r in db.execute('PRAGMA table_info(batches)')}
    db.execute('DROP TRIGGER abort_migration'); db.commit(); db.close()
    store = nj.JudgeStore(path)
    assert store.total_reserved_or_actual_cost() == .123
    store.close()


def test_field_export_preserves_aliases_without_loading_raw_requests(tmp_path):
    store = nj.JudgeStore(tmp_path/'judge.sqlite')
    req = nj.prepare_request(record(), 'M', 'original', 'model-a')
    store.register('first', req, {})
    store.register('alias', req, {})
    store.finish(req['content_key'], {'valid':True,'verdict':'yes','cost':0})
    statements = []
    store.db.set_trace_callback(statements.append)
    assert {r['task_key']:r['verdict'] for r in store.field_verdicts()} == {'first':'yes','alias':'yes'}
    store.db.set_trace_callback(None)
    query = next(s for s in statements if s.startswith('SELECT m.task_key'))
    plan = ' '.join(r[3] for r in store.db.execute('EXPLAIN QUERY PLAN '+query))
    assert 'COVERING INDEX' in plan, plan
    store.close()
