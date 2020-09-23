from AL_environment_MDP.al_parameters import AL_Parameters
from supervised_learning_tasks.task_parameters import Task_Parameters
from AL_environment_MDP.al_env import AL_Env

def test_environment():
    al_parameters = AL_Parameters(startingSize=8,annotationBudget=12)
    al_parameters.batchSize_annotation = 4
    for task_name in ["model_UCI", "model_checkerboard", "model_Vision", "model_bAbI"]:
        task_params = Task_Parameters(taskName=task_name)
        task = task_params.createTask()

        al_env = AL_Env(al_parameters, task)
        al_env.reset()

        iteration = 0
        expectedNoIterations = al_env.expectedNoIterations()
        for i in range(expectedNoIterations):
            action = [0]
            newObservation, reward, done, info = al_env.step(action)
            iteration += 1
            if done:
                break