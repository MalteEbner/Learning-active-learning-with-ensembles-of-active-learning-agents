import hyperopt as hp
import lightgbm  # needed by hyperopt
import sklearn  # needed by hyperopt

from supervised_learning_tasks.task_parameters import Task_Parameters
from AL_environment_MDP.al_parameters import AL_Parameters
from AL_agents.al_agent_parameters import AL_Agent_Parameters
from AL_agents.ensemble.train_ensemble_objective_function import train_ensemble_objective_function
from AL_agents.ensemble.train_ensemble_beta_dict_handler import BetaDictHandler

startingSize = 8
annotationBudget = 72
batchSize_annotation = 32
maxNoRunsInParallel = 12
max_evals = 4

algo = [hp.atpe.suggest, hp.tpe.suggest, hp.rand.suggest][0]

'''
Parameters for monte carlo simulation
'''
task_param_list = []
if False:
    variantParams = Task_Vision_variantParams(dataset='MNIST', repr_1d_type='tSNE')
    task_param_list += [Task_Parameters(taskName="model_Vision", variantParams=variantParams)]
if True:
    task_param_list += [Task_Parameters(taskName="model_checkerboard", variantParams="2x2")]
    task_param_list += [Task_Parameters(taskName="model_checkerboard", variantParams="2x2_rotated")]
if False:
    UCI_Datasets = ['2-breast_cancer', '3-diabetis', '4-flare_solar',
                    '5-german', '6-heart', '7-mushrooms', '8-waveform', '9-wdbc']
    for uciDataset in UCI_Datasets:
        task_param_list += [Task_Parameters(taskName="model_UCI", variantParams=uciDataset)]
if False:
    task_bAbI_variantParams = Task_bAbI_variantParams(challenge_type='single_supporting_fact_10k')
    # task_bAbI_variantParams = Task_bAbI_variantParams(challenge_type='two_supporting_facts_10k')
    task_param_list += [Task_Parameters(taskName="model_bAbI_memoryNetwork", variantParams=task_bAbI_variantParams)]

task_names = list([task_param.__shortRepr__() for task_param in task_param_list])
task_param_list *= int(maxNoRunsInParallel / len(task_param_list))
al_params = AL_Parameters(annotationBudget=annotationBudget, startingSize=startingSize)
agent_param = AL_Agent_Parameters(agentName="Ensemble", batchSize_annotation=batchSize_annotation)

mean_type = "arithmetic"
if len(task_names) > 1:
    mean_type = "geometric"


def objective_function(beta_dict):
    objective_to_maximize = train_ensemble_objective_function(
        beta_dict, task_param_list, al_params, agent_param,
        mean_type=mean_type, n_jobs=maxNoRunsInParallel)
    print(f"{objective_to_maximize}  {beta_dict}")
    return -1 * objective_to_maximize



relevant_task_name = task_param_list[0].taskName
search_space = BetaDictHandler(relevant_task_name).get_hyperopt_space()

example_beta = hp.pyll.stochastic.sample(search_space)

best_beta = hp.fmin(objective_function, search_space, algo=algo, max_evals=max_evals, verbose=False)
print(f"best beta: {best_beta}")
