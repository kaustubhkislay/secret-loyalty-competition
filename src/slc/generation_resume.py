"""Bind selected failed jobs to exact prior children before remote terminal checks."""
from slc.generation_jobs import expected_job_identity, safe_component


def job_key(job):
    return f"{safe_component(job['model_tag'])}/{safe_component(job['battery_name'])}"


def validate_resume_ids(jobs, mapping):
    if not isinstance(mapping, dict):
        raise ValueError('resume child mapping must be an object')
    keys = [job_key(job) for job in jobs]
    if len(set(keys)) != len(keys) or set(mapping) != set(keys):
        raise ValueError('resume child mapping must name each selected job exactly once')
    if any(not isinstance(call, str) or not call.startswith('fc-') or len(call) <= 3
           for call in mapping.values()):
        raise ValueError('resume requires explicit prior child handles')
    if len(set(mapping.values())) != len(mapping):
        raise ValueError('different jobs cannot share a prior child')
    return dict(mapping)


def resume_call_ids(document, jobs):
    """Require explicit failures and identical jobs; this does not prove remote termination.

    acquire_run_claim must still inspect the latest recorded child on the server
    before electing a successor. A local outcome file alone never authorizes it.
    """
    rows = document.get('outcomes') if isinstance(document, dict) else None
    if not isinstance(rows, list):
        raise ValueError('resume source must be a suite outcome object')
    selected = {job_key(job): job for job in jobs}
    if len(selected) != len(jobs):
        raise ValueError('duplicate selected output')
    mapping = {}
    for row in rows:
        key = job_key(row)
        if key not in selected:
            continue
        if key in mapping:
            raise ValueError('duplicate selected outcome')
        result = row.get('result')
        if (row.get('status') != 'failed' or not isinstance(result, dict)
                or result.get('status') != 'failed'):
            raise ValueError('batch resume requires an explicit failed child result')
        call = row.get('function_call_id')
        if call != result.get('function_call_id'):
            raise ValueError('outcome and failed result identify different children')
        if expected_job_identity(row) != expected_job_identity(selected[key]):
            raise ValueError('resume plan differs from the failed job identity')
        mapping[key] = call
    return validate_resume_ids(jobs, mapping)
