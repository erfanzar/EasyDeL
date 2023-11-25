import pprint
from lm_eval import evaluator, tasks
from typing import List, Optional

AVAILABLE_TASKS = ['wsc', 'piqa', 'winogrande', 'openbookqa', 'logiqa']


def evaluate(model, task_list: Optional[List[str]] = None, write_out: bool = True, limit: int = 0, shots: int = 5):
    if task_list is None:
        task_list = ['wsc', "piqa"]

    for task in task_list:
        assert task in AVAILABLE_TASKS, f'UnKnown Task {tasks} available tasks are {AVAILABLE_TASKS}'
    results = evaluator.evaluate(
        model, tasks.get_task_dict(task_list), False, shots,
        limit=None if limit <= 0 else limit,
        write_out=write_out,
    )
    pprint.pprint(results)
    return results
