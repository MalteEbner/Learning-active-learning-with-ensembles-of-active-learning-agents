from typing import Dict

import hyperopt as hp

from AL_agents.al_agent_parameters import ALAgentParameters
from AL_agents.ensemble.train_ensemble_function import train_ensemble_with_hyperopt
from AL_environment_MDP.al_parameters import ALParameters
from supervised_learning_tasks.task_parameters import TaskParameters

starting_size = 8
annotation_budget = 32
batch_size_annotation = 8

n_jobs = 4
runs_per_objective_function = 64

max_evals = 100

algo = [hp.atpe.suggest, hp.tpe.suggest, hp.rand.suggest][0]

'''
Parameters for monte carlo simulation
'''
training_task = ['UCI', 'checkerboard', 'MNIST', 'bAbI'][0]
task_param_list = []
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

train_ensemble_with_hyperopt(algo, task_param_list, n_jobs, al_params, agent_param, max_evals)
