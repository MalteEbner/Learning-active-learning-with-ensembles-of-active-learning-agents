from typing import Dict

import pytest
import hyperopt as hp
import lightgbm  # needed by hyperopt
import sklearn  # needed by hyperopt

from AL_apply_agent_on_task.parallel_run_handler import ParallelRunHandler
from supervised_learning_tasks.task_parameters import Task_Parameters
from AL_environment_MDP.al_parameters import AL_Parameters
from AL_agents.al_agent_parameters import AL_Agent_Parameters
from AL_agents.ensemble.train_ensemble_objective_function import train_ensemble_objective_function
from AL_agents.ensemble.train_ensemble_beta_dict_handler import BetaDictHandler

def _test_ensemble_training(task_name):
    startingSize = 8
    annotationBudget = 16
    batchSize_annotation = -1
    maxNoRunsInParallel = 2
    max_evals = 2

    algo = [hp.atpe.suggest, hp.tpe.suggest, hp.rand.suggest][0]

    task_param_list = [Task_Parameters(task_name)]

    al_params = AL_Parameters(annotationBudget=annotationBudget, startingSize=startingSize)
    agent_param = AL_Agent_Parameters(agentName="Ensemble", batchSize_annotation=batchSize_annotation)

    mean_type = "arithmetic"
    if len(task_param_list) > 1:
        mean_type = "geometric"

    with ParallelRunHandler(task_param_list[0].getExperimentFilename(), n_jobs=maxNoRunsInParallel, test=False,
                            save_results=False,
                            parallelization=True,
                            verbose=False) as parallel_run_handler:
        def objective_function(beta_dict: Dict):
            objective_to_maximize = train_ensemble_objective_function(
                parallel_run_handler,
                beta_dict,
                task_param_list, al_params, agent_param,
                mean_type=mean_type)
            print(f"{objective_to_maximize}  {beta_dict}")
            return -1 * objective_to_maximize

        search_space = BetaDictHandler().get_hyperopt_space()

        best_beta = hp.fmin(objective_function, search_space, algo=algo, max_evals=max_evals, verbose=True)

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