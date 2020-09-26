from typing import Dict

import hyperopt as hp
import lightgbm  # needed by hyperopt
import sklearn  # needed by hyperopt
from scipy.stats import gmean

from AL_agents.ensemble.train_ensemble_beta_dict_handler import BetaDictHandler
from AL_apply_agent_on_task.parallel_run_handler import ParallelRunHandler


def train_ensemble_with_hyperopt(algo, task_param_list, n_jobs, al_params, agent_param, max_evals):
    mean_type = "arithmetic"
    if len(task_param_list) > 1:
        mean_type = "geometric"

    with ParallelRunHandler(task_param_list[0].get_experiment_filename(), n_jobs=n_jobs, test=False,
                            save_results=False,
                            parallelization=True,
                            verbose=False) as parallel_run_handler:
        def objective_function(beta_dict: Dict):
            objective_to_maximize = get_mean_accuracy_of_agent(
                parallel_run_handler,
                beta_dict,
                task_param_list, al_params, agent_param,
                mean_type=mean_type)
            print(f"{objective_to_maximize}  {beta_dict}")
            return -1 * objective_to_maximize

        relevant_task_name = task_param_list[0].task_name
        search_space = BetaDictHandler().get_hyperopt_space()

        example_beta = hp.pyll.stochastic.sample(search_space)

        best_beta = hp.fmin(objective_function, search_space, algo=algo, max_evals=max_evals, verbose=True)
        print(f"best beta: {best_beta}")


def get_mean_accuracy_of_agent(
        parallel_run_handler: ParallelRunHandler,
        beta_dict: dict,
        task_param_list, al_params, agent_param,
        mean_type="arithmetic") -> float:
    agent_param.beta_dict = beta_dict
    agent_param_list = [agent_param] * len(task_param_list)

    application_handlers = parallel_run_handler.al_apply_agents_on_task(
        task_param_list, al_params, agent_param_list)

    performances = [application_handler.infos[-1]['accuracy']
                    for application_handler in application_handlers]

    if mean_type == "arithmetic":
        return sum(performances) / len(performances)
    elif mean_type == "geometric":
        return float(gmean(performances))
