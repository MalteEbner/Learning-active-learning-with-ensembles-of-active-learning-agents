from AL_apply_agent_on_task.application_config import get_application_config
from AL_agents.al_agent_parameters import AL_Agent_Parameters
from AL_apply_agent_on_task.application_handler_file_handler import ApplicationHandlerFileHandlerJSON
from AL_apply_agent_on_task.application_handler import ApplicationHandler


def test_application_handler():
    taskName = ["model_UCI", "model_checkerboard", "model_Vision", "model_bAbI_memoryNetwork"][0]
    test = True

    # define task
    (task_params, base_dataset, usualBatchSize, usualBatchSize_random,
     al_params, n_jobs, noRepetitions) = get_application_config(taskName)

    # define application handler
    agent_params = AL_Agent_Parameters(agentName="Random",batchSize_annotation=4,batchSize_agent=-1)
    application_handler = ApplicationHandler(task_params, al_params, agent_params)

    # define file handler for saving the results
    filename = "../Experiments/results/applicationHandler_test.json"
    fileHandler = ApplicationHandlerFileHandlerJSON(filename)

    # run the experiment
    application_handler.run_episode(saveFullData=False)

    # save the results
    fileHandler.writeApplicationHandlersToFile([application_handler])

    # plot the results
    ApplicationHandlerFileHandlerJSON(filename).plotAllContentWithConfidenceIntervals()