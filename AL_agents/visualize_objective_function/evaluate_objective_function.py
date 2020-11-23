import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from typing import Dict, List, Callable

import hyperopt as hp
import lightgbm  # needed by hyperopt
import sklearn  # needed by hyperopt
from scipy.stats import gmean
import pandas as pd

from AL_agents.al_agent_parameters import ALAgentParameters
from AL_agents.ensemble.train_ensemble_beta_dict_handler import BetaDictHandler
from AL_apply_agent_on_task.parallel_run_handler import ParallelRunHandler
from AL_environment_MDP.al_parameters import ALParameters
from supervised_learning_tasks.task_parameters import TaskParameters

training_task = ['UCI', 'checkerboard', 'MNIST', 'bAbI'][0]
starting_size = 8
annotation_budget = 32
batch_size_annotation = 8

n_jobs = 1
runs_per_objective_function = 64

max_evals = 20

algo = [hp.atpe.suggest, hp.tpe.suggest, hp.rand.suggest][1]

'''
Parameters for monte carlo simulation
'''

task_param_list: List[TaskParameters] = []
if training_task == 'UCI':
    uci_datasets = ['2-breast_cancer', '3-diabetis', '4-flare_solar',
                    '5-german', '6-heart', '7-mushrooms', '8-waveform', '9-wdbc']
    for uci_dataset in uci_datasets:
        task_param_list += [TaskParameters(task_name="model_UCI", dataset=uci_dataset)]
if training_task == 'checkerboard':
    task_param_list += [TaskParameters(task_name="model_checkerboard", dataset="2x2")]
    task_param_list += [TaskParameters(task_name="model_checkerboard", dataset="2x2_rotated")]
if training_task == "MNIST":
    task_param_list += [TaskParameters(task_name="model_Vision", dataset='MNIST')]
if training_task == 'bAbI':
    task_param_list += [TaskParameters(task_name="model_bAbI", dataset='single_supporting_fact')]

task_names = list([task_param.__short_repr__() for task_param in task_param_list])
task_param_list *= int(max(1, runs_per_objective_function / len(task_param_list)))
al_params = ALParameters(annotation_budget=annotation_budget, starting_size=starting_size)
agent_param = ALAgentParameters(agent_name="Ensemble", batch_size_annotation=batch_size_annotation)

filename = f"evaluations/df objective {task_param_list[0].task_name}.csv"
with_header = not os.path.exists(filename)

mean_type = "arithmetic"
if len(task_param_list) > 1:
    mean_type = "geometric"


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


result_dict_list = []


with ParallelRunHandler(task_param_list[0].get_experiment_filename(), n_jobs=n_jobs, test=False,
                        save_results=False,
                        parallelization=False,
                        verbose=False) as parallel_run_handler:

    def objective_function(beta_dict: Dict):
        objective_to_maximize = get_mean_accuracy_of_agent(
            parallel_run_handler,
            beta_dict,
            task_param_list, al_params, agent_param,
            mean_type=mean_type)
        #print(f"{objective_to_maximize}  {beta_dict}")

        beta_dict['accuracy'] = objective_to_maximize
        df_ = pd.DataFrame([beta_dict])
        df_.to_csv(filename, mode='a', header=with_header, index=False)
        with_header = False

        return -1 * objective_to_maximize

    search_space = BetaDictHandler().get_hyperopt_space()

    example_beta = hp.pyll.stochastic.sample(search_space)

    best_beta = hp.fmin(objective_function, search_space, algo=algo, max_evals=max_evals, verbose=True)


if False:
    filename = f"df objective {task_names}.csv"
    df_new: pd.DataFrame = pd.DataFrame(result_dict_list)
    try:
        df_old = pd.read_csv(filename)
        df = pd.concat([df_old, df_new])
    except:
        df = df_new
    df.to_csv(f"df objective {task_names}.csv",index=False)




