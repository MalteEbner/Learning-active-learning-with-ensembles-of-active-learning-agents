from AL_agents.al_agent_parameters import ALAgentParameters
from AL_apply_agent_on_task.application_handler_file_handler import ApplicationHandlerFileHandlerJSON
from AL_apply_agent_on_task.application_config import get_application_config
from AL_apply_agent_on_task.parallel_run_handler import ParallelRunHandler
from AL_agents.ensemble.train_ensemble_beta_dict_handler import BetaDictHandler

task_name = ["model_UCI", "model_checkerboard", "model_Vision", "model_bAbI"][2]
test = False
delete_old_ensemble_data = False
only_ensemble = False

# define task
(task_param, base_dataset, usual_batch_size,
 al_params, n_jobs, no_repetitions) = get_application_config(task_name)

# define agents to apply on task
agent_param_list = list()
# agentParams.append(AL_Agent_Parameters(agentName="Uncertainty", batch_size_annotation=1))
agent_param_list.append(ALAgentParameters(agent_name="Random", batch_size_annotation=usual_batch_size))
agent_param_list.append(ALAgentParameters(agent_name="Uncertainty", batch_size_annotation=usual_batch_size))
agent_param_list.append(ALAgentParameters(agent_name="Diversity", batch_size_annotation=usual_batch_size))
agent_param_list.append(ALAgentParameters(agent_name="Uncertainty_Diversity", batch_size_annotation=usual_batch_size))
agent_param_list.append(ALAgentParameters(agent_name="Representative", batch_size_annotation=usual_batch_size))
beta_dict = BetaDictHandler(task_name).beta_dict
agent_param_list.append(ALAgentParameters(agent_name="Ensemble", batch_size_annotation=usual_batch_size,
                                          beta_dict=beta_dict))
if only_ensemble:
    agent_param_list = agent_param_list[-1:]  # only rerun the ensemble
agent_param_list *= no_repetitions

task_param_list = [task_param] * len(agent_param_list)

with ParallelRunHandler(task_param_list[0].get_experiment_filename(), n_jobs=n_jobs, test=test, save_results=True,
                        parallelization=False, verbose=False, delete_old_ensemble_data= delete_old_ensemble_data) \
        as parallel_run_handler:
    finished_application_handlers, filename = parallel_run_handler.al_apply_agents_on_task(
        task_param_list, al_params, agent_param_list,
    )

ApplicationHandlerFileHandlerJSON(filename).plot_all_content_with_confidence_intervals()
