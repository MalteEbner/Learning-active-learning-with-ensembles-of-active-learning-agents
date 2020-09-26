import pytest

from AL_apply_agent_on_task.application_config import get_application_config
from AL_agents.al_agent_parameters import ALAgentParameters
from AL_apply_agent_on_task.application_handler_file_handler import ApplicationHandlerFileHandlerJSON
from AL_apply_agent_on_task.application_handler import ApplicationHandler
from AL_environment_MDP.al_parameters import ALParameters
from supervised_learning_tasks.task_parameters import TaskParameters


def _test_application_handler(task_name):
    test = True

    al_parameters = ALParameters(startingSize=8, annotation_budget=16)
    task_params = TaskParameters(task_name=task_name)

    # define application handler
    agent_params = ALAgentParameters(agent_name="Random", batch_size_annotation=4, batch_size_agent=-1)
    application_handler = ApplicationHandler(task_params, al_parameters, agent_params)

    # define file handler for saving the results
    filename = f"../pytests/tests_application_handlers/applicationHandler_test_{task_name}.json"
    file_handler = ApplicationHandlerFileHandlerJSON(filename)

    # run the experiment
    application_handler.run_episode()

    # save the results
    file_handler.write_application_handlers_to_file([application_handler])

    # plot the results
    ApplicationHandlerFileHandlerJSON(filename).plot_all_content_with_confidence_intervals()


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
