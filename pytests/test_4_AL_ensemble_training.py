from typing import Dict

import pytest
import hyperopt as hp
import lightgbm  # needed by hyperopt
import sklearn  # needed by hyperopt

from supervised_learning_tasks.task_parameters import TaskParameters
from AL_environment_MDP.al_parameters import ALParameters
from AL_agents.al_agent_parameters import ALAgentParameters
from AL_agents.ensemble.train_ensemble_function import train_ensemble_with_hyperopt


def _test_ensemble_training(task_name):
    starting_size = 8
    annotation_budget = 16
    batch_size_annotation = -1
    n_jobs = 2
    max_evals = 2

    algo = [hp.atpe.suggest, hp.tpe.suggest, hp.rand.suggest][0]

    task_param_list = [TaskParameters(task_name)]

    al_params = ALParameters(annotation_budget=annotation_budget, starting_size=starting_size)
    agent_param = ALAgentParameters(agent_name="Ensemble", batch_size_annotation=batch_size_annotation)

    train_ensemble_with_hyperopt(algo, task_param_list, n_jobs, al_params, agent_param, max_evals)

    print('blub')


def _get_test_parameters():
    test_cases = []
    for task_name in ["model_UCI", "model_checkerboard", "model_Vision", "model_bAbI"]:
        name = f'{task_name}'
        test_case = pytest.param(task_name, id=name)
        test_cases.append(test_case)
    return test_cases


@pytest.mark.parametrize("task_name", _get_test_parameters())
def test_heuristics(task_name):
    _test_ensemble_training(task_name)
