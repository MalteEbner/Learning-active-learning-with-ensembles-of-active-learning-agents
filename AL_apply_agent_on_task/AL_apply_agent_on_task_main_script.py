import time

import ray
import numpy as np


from AL_agents.al_agent_parameters import AL_Agent_Parameters
from AL_apply_agent_on_task.application_handler_file_handler import ApplicationHandlerFileHandlerJSON
from AL_apply_agent_on_task.application_handler import ApplicationHandler
from AL_apply_agent_on_task.application_config import get_application_config


taskName = ["model_UCI", "model_checkerboard", "model_Vision", "model_bAbI_memoryNetwork"][1]
test = False

# define task
(task_params, base_dataset, usualBatchSize, usualBatchSize_random,
 al_parameters, n_jobs, noRepetitions) = get_application_config(taskName)

# define agents to apply on task
agentParams = []
#agentParams.append(AL_Agent_Parameters(agentName="Uncertainty", batchSize_annotation=1))
agentParams.append(AL_Agent_Parameters(agentName="Random", batchSize_annotation=usualBatchSize))
agentParams.append(AL_Agent_Parameters(agentName="Diversity", batchSize_annotation=usualBatchSize))
agentParams.append(AL_Agent_Parameters(agentName="Uncertainty", batchSize_annotation=usualBatchSize))
agentParams.append(AL_Agent_Parameters(agentName="Representative", batchSize_annotation=usualBatchSize))
agentParams.append(AL_Agent_Parameters(agentName="Ensemble",batchSize_annotation=usualBatchSize))
#agentParams = agentParams[-1:]
agentParams *= noRepetitions

# define applicationHandler to store the results
if test:
    filename = "Experiments/results/applicationHandler_test.json"
    import os

    try:
        os.remove(filename)
    except OSError as e:
        pass
else:
    filename = task_params.getExperimentFilename()
    filename += '.json'
fileHandler = ApplicationHandlerFileHandlerJSON(filename)

# perform actual experiments
ray.init(num_cpus=n_jobs,temp_dir='/tmp/ray_LAL_ensemble')


@ray.remote
def run_agent_on_task(task_params, al_params, agent_params):
    applicationHandler = ApplicationHandler(task_params, al_params, agent_params, verbose=False)
    applicationHandler.run_episode(saveFullData=False)
    return applicationHandler

n_jobs = min(n_jobs,len(agentParams))
remainingRayIDs = [run_agent_on_task.remote(task_params, al_parameters, agent_param)
                   for agent_param in agentParams]
start = time.time()
print(f'Starting parallelization at time {time.time()-start}')
finishedTasks = 0
total_tasks = len(remainingRayIDs)
num_returns = min(208,int(noRepetitions/4),len(remainingRayIDs),12)
num_returns = max(1,num_returns)
while len(remainingRayIDs) > 0:
    num_returns = min(num_returns,len(remainingRayIDs))
    ready_ids, remainingRayIDs = ray.wait(remainingRayIDs, num_returns=num_returns)
    finishedTasks += len(ready_ids)
    print(f'finished {finishedTasks} of {total_tasks} tasks at time {time.time()-start}')
    applicationHandlers = ray.get(ready_ids)
    fileHandler.writeApplicationHandlersToFile(applicationHandlers)

ray.shutdown()

ApplicationHandlerFileHandlerJSON(filename).plotAllContentWithConfidenceIntervals()