import pprint
from lm_eval import evaluator, tasks
from typing import List, Optional

AVAILABLE_TASKS = ['wsc', 'piqa', 'winogrande', 'openbookqa', 'logiqa']


def evaluate(model, task_list: Optional[List[str]] = None, write_out: bool = True, limit: int = 0, shots: int = 5):
    """
    The evaluate function takes a model and evaluates it on the tasks specified in task_list.
    The results are printed to stdout, and optionally written out to a file.


    :param model: Specify the model to be evaluated
    :param task_list: Optional[List[str]]: Specify which tasks to evaluate on
    :param write_out: bool: Write the output to a file
    :param limit: int: Limit the number of examples that are evaluated
    :param shots: int: Specify how many times to run the model on a given task
    :return: A dictionary with the following keys
    
    """
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
