"""Petri task assembly for immutable, conversation-only targets."""


def make_task(cell, limit=None):
    from inspect_ai import Task
    from inspect_petri import (JudgeDimension, audit_judge, audit_solver, auditor_agent,
                               auditor_tools, seeds_dataset, target_agent)
    seeds = cell['instructions'] if limit is None else cell['instructions'][:limit]
    dataset = seeds_dataset(seeds)
    ids = cell.get('scenario_ids', list(range(1, len(seeds) + 1)))
    for sample, scenario_id in zip(dataset, ids):
        sample.id = scenario_id
    return Task(dataset=dataset, solver=audit_solver(
        auditor=auditor_agent(max_turns=cell['max_turns'], compaction=False,
            target_display_name='assistant', tools=auditor_tools(prefill=False,
                rollback=True, target_tools='none', exclude={'set_system_message'})),
        target=target_agent(system_required=False)),
        scorer=audit_judge([JudgeDimension(**d) for d in cell['dimensions']]))
