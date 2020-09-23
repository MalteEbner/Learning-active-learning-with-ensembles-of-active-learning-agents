import pytest

from AL_apply_agent_on_task.application_config import get_application_config
from AL_agents.al_agent_parameters import AL_Agent_Parameters
from AL_apply_agent_on_task.application_handler_file_handler import ApplicationHandlerFileHandlerJSON
from AL_apply_agent_on_task.application_handler import ApplicationHandler
from AL_environment_MDP.al_parameters import AL_Parameters
from supervised_learning_tasks.task_parameters import Task_Parameters


def _test_application_handler(task_name):
    test = True

    al_parameters = AL_Parameters(startingSize=8, annotationBudget=16)
    task_params = Task_Parameters(taskName=task_name)

    # define application handler
    agent_params = AL_Agent_Parameters(agentName="Random",batchSize_annotation=4,batchSize_agent=-1)
    application_handler = ApplicationHandler(task_params, al_parameters, agent_params)

    # define file handler for saving the results
    filename = f"../pytests/tests_application_handlers/applicationHandler_test_{task_name}.json"
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