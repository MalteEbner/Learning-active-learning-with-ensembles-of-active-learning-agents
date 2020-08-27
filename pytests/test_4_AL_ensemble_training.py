import hyperopt as hp
import lightgbm  # needed by hyperopt
import sklearn  # needed by hyperopt

from supervised_learning_tasks.task_parameters import Task_Parameters
from AL_environment_MDP.al_parameters import AL_Parameters
from AL_agents.al_agent_parameters import AL_Agent_Parameters
from AL_agents.ensemble.train_ensemble_objective_function import train_ensemble_objective_function
from AL_agents.ensemble.train_ensemble_beta_dict_handler import BetaDictHandler

from AL_agents.ensemble.train_ensemble import objective_function

def test_ensemble_training():
    startingSize = 8
    annotationBudget = 16
    batchSize_annotation = -1
    maxNoRunsInParallel = 2
    max_evals = 2

    algo = [hp.atpe.suggest, hp.tpe.suggest, hp.rand.suggest][0]

    task_param_list = [Task_Parameters(taskName="model_checkerboard", variantParams="2x2")]

    al_params = AL_Parameters(annotationBudget=annotationBudget, startingSize=startingSize)
    agent_param = AL_Agent_Parameters(agentName="Ensemble", batchSize_annotation=batchSize_annotation)

    relevant_task_name = task_param_list[0].taskName
    search_space = BetaDictHandler(relevant_task_name).get_hyperopt_space()

    best_beta = hp.fmin(objective_function, search_space, algo=algo, max_evals=max_evals, verbose=False)