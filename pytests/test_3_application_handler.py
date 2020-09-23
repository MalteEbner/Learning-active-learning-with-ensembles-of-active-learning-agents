import pytest

from AL_apply_agent_on_task.application_config import get_application_config
from AL_agents.al_agent_parameters import AL_Agent_Parameters
from AL_apply_agent_on_task.application_handler_file_handler import ApplicationHandlerFileHandlerJSON
from AL_apply_agent_on_task.application_handler import ApplicationHandler


def _test_application_handler(task_name):
    test = True

    # define task
    (task_params, base_dataset, usualBatchSize, usualBatchSize_random,
     al_params, n_jobs, noRepetitions) = get_application_config(task_name)

    al_params.startingSize = 40
    al_params.annotationBudget = 48

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

def _get_test_parameters():
    test_cases = []
    for task_name in ["model_UCI", "model_checkerboard", "model_Vision", "model_bAbI"]:
        name = f'{task_name}'
        test_case = pytest.param(task_name, id=name)
        test_cases.append(test_case)
    return test_cases

@pytest.mark.parametrize("task_name", _get_test_parameters())
def test_heuristics(task_name):
    _test_application_handler(task_name)