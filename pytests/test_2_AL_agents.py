import pytest

from AL_environment_MDP.al_environment import ALEnvironment
from supervised_learning_tasks.task_parameters import TaskParameters
from AL_environment_MDP.al_parameters import ALParameters
from AL_agents.al_agent_parameters import ALAgentParameters


def _apply_single_heuristic(agent_parameters: ALAgentParameters):
    agent = agent_parameters.create_agent()

    al_parameters = ALParameters(startingSize=8, annotation_budget=16)
    al_parameters.batch_size_annotation = 4
    task_params = TaskParameters(task_name="model_checkerboard", dataset="2x2_rotated")
    task = task_params.create_task()

    al_env = ALEnvironment(al_parameters, task)

    # run AL with random sampling
    iteration = al_parameters.startingSize
    expectedNoIterations = al_env.expected_number_iterations()
    # print("Starting random AL with %d iterations" % expectedNoIterations)

    observation = al_env.reset()
    for i in range(expectedNoIterations):
        action = agent.policy(observation)
        observation, reward, done, info = al_env.step(action)
        # print('iteration %d: accuracy %.4f' % (iteration, info['accuracy']))
        iteration += 1
        if done:
            break


def _get_test_parameters():
    test_cases = []
    for agent_name in ["Random", "Uncertainty", "Diversity", "Representative", "Ensemble"]:
        for batch_size_annotation in [1, 8]:
            for batch_size_agent in [1, 3, -1]:
                name = f'{agent_name}_{batch_size_annotation}_{batch_size_agent}'
                agent_params = ALAgentParameters(
                    agent_name=agent_name, batch_size_annotation=batch_size_annotation, batch_size_agent=batch_size_agent)
                test_case = pytest.param(agent_params, id=name)
                test_cases.append(test_case)
    return test_cases


@pytest.mark.parametrize("agent_params", _get_test_parameters())
def test_heuristics(agent_params):
    _apply_single_heuristic(agent_params)
