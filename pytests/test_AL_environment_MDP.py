from AL_environment_MDP.al_parameters import AL_Parameters
from supervised_learning_tasks.task_parameters import Task_Parameters
from AL_environment_MDP.al_env import AL_Env

def test_environment():
    al_parameters = AL_Parameters(startingSize=8,annotationBudget=72)
    al_parameters.batchSize_annotation = 32
    task_params = Task_Parameters(taskName="model_checkerboard",variantParams="2x2_rotated")
    task = task_params.createTask()

    al_env = AL_Env(al_parameters, task)
    al_env.reset()

    # run AL with random sampling
    iteration = 0
    expectedNoIterations = al_env.expectedNoIterations()
    print("Starting random AL with %d iterations" % expectedNoIterations)
    for i in range(expectedNoIterations):
        action = [0]
        newObservation, reward, done, info = al_env.step(action)
        print('iteration %d: accuracy %.4f' % (iteration, info['accuracy']))
        iteration += 1
        if done:
            break