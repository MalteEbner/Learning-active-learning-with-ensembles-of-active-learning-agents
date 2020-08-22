from supervised_learning_tasks.task_parameters import Task_Parameters
from supervised_learning_tasks.tasks_vision.task_Vision_variantParams import Task_Vision_variantParams
from supervised_learning_tasks.tasks_QA_bAbI.task_bAbI_variantParams import Task_bAbI_variantParams


# define AL run
'''
possible tasks: 
model_bAbI_memoryNetwork, model_Vision, model_checkerboard
'''
taskName = ["model_Vision", "model_bAbI_memoryNetwork", "model_checkerboard"][2]
if taskName == "model_Vision":
    task_Vision_variantParams = Task_Vision_variantParams(dataset="MNIST",repr_1d_type='tSNE')
    task_params = Task_Parameters(taskName=taskName,variantParams=task_Vision_variantParams)
elif taskName == "model_bAbI_memoryNetwork":
    task_bAbI_variantParams = Task_bAbI_variantParams(challenge_type='single_supporting_fact_10k')
    task_bAbI_variantParams = Task_bAbI_variantParams(challenge_type='two_supporting_facts_10k')
    task_params = Task_Parameters(taskName=taskName,variantParams=task_bAbI_variantParams)
elif taskName == "model_checkerboard":
    task_params = Task_Parameters(taskName=taskName,variantParams="2x2_rotated")
else:
    task_params = Task_Parameters(taskName=taskName)

task = task_params.createTask()
task.trainWithHyperopt(noRandomSamples=200,noIters=50)


