from AL_agents.al_agent_parameters import AL_Agent_Parameters
from AL_apply_agent_on_task.application_handler_file_handler import ApplicationHandlerFileHandlerJSON
from AL_apply_agent_on_task.application_config import get_application_config
from AL_apply_agent_on_task.parallel_run_handler import ParallelRunHandler

taskName = ["model_UCI", "model_checkerboard", "model_Vision", "model_bAbI_memoryNetwork"][1]
test = True

# define task
(task_param, base_dataset, usualBatchSize, usualBatchSize_random,
 al_params, n_jobs, noRepetitions) = get_application_config(taskName)

# define agents to apply on task
agent_param_list = []
# agentParams.append(AL_Agent_Parameters(agentName="Uncertainty", batchSize_annotation=1))
agent_param_list.append(AL_Agent_Parameters(agentName="Random", batchSize_annotation=usualBatchSize))
agent_param_list.append(AL_Agent_Parameters(agentName="Diversity", batchSize_annotation=usualBatchSize))
agent_param_list.append(AL_Agent_Parameters(agentName="Uncertainty", batchSize_annotation=usualBatchSize))
agent_param_list.append(AL_Agent_Parameters(agentName="Representative", batchSize_annotation=usualBatchSize))
agent_param_list.append(AL_Agent_Parameters(agentName="Ensemble", batchSize_annotation=usualBatchSize))
# agentParams = agentParams[-1:]
agent_param_list *= noRepetitions

task_param_list = [task_param] * len(agent_param_list)

with ParallelRunHandler(task_param_list[0].getExperimentFilename(), n_jobs=n_jobs, test=test, save_results=True,
                        parallelization=True) as parallel_run_handler:
    finished_application_handlers, filename = parallel_run_handler.al_apply_agents_on_task(
        task_param_list, al_params, agent_param_list,
    )

ApplicationHandlerFileHandlerJSON(filename).plotAllContentWithConfidenceIntervals()
