import pytest

from supervised_learning_tasks.task_parameters import Task_Parameters
from AL_environment_MDP.al_env import AL_Env
from AL_environment_MDP.al_parameters import AL_Parameters
from AL_agents.al_agent_parameters import AL_Agent_Parameters

def _apply_single_heuristic(agent_parameters: AL_Agent_Parameters):
    agent = agent_parameters.createAgent()

    al_parameters = AL_Parameters(startingSize=8, annotationBudget=16)
    task_params = Task_Parameters(taskName="model_checkerboard", variantParams="2x2_rotated")
    task = task_params.createTask()

    al_env = AL_Env(al_parameters, task)

    # run AL with random sampling
    iteration = al_parameters.startingSize
    expectedNoIterations = al_env.expectedNoIterations()
    #print("Starting random AL with %d iterations" % expectedNoIterations)

    observation = al_env.reset()
    for i in range(expectedNoIterations):
        action = agent.policy(observation)
        observation, reward, done, info = al_env.step(action)
        #print('iteration %d: accuracy %.4f' % (iteration, info['accuracy']))
        iteration += 1
        if done:
            break


def _get_test_parameters():
    test_cases = []
    for agentName in ["Random", "Uncertainty", "Diversity", "Representative", "Ensemble"]:
        for batchSize_annotation in [1, 8]:
            for batchSize_agent in [1, 3, -1]:
                name = f'{agentName}_{batchSize_annotation}_{batchSize_agent}'
                agent_params = AL_Agent_Parameters(
                    agentName=agentName,batchSize_annotation=batchSize_annotation,batchSize_agent=batchSize_agent)
                test_case = pytest.param(agent_params, id=name)
                test_cases.append(test_case)
    return test_cases

@pytest.mark.parametrize("agent_params", _get_test_parameters())
def test_heuristics(agent_params):
    _apply_single_heuristic(agent_params)

