from supervised_learning_tasks.task_parameters import TaskParameters
from supervised_learning_tasks.tasks_vision.task_Vision_
from supervised_learning_tasks.tasks_QA_bAbI.tasks_QA_bAbI import TaskBabiVariantParams


# define AL run
'''
possible tasks: 
model_bAbI_memoryNetwork, model_Vision, model_checkerboard
'''
taskName = ["model_Vision", "model_bAbI_memoryNetwork", "model_checkerboard"][1]
if taskName == "model_Vision":
    task_Vision_variantParams = TaskVisionVariantParams(dataset="MNIST", repr_1d_type='tSNE')
    task_params = TaskParameters(task_name=taskName, variant_params=task_Vision_variantParams)
elif taskName == "model_bAbI_memoryNetwork":
    task_bAbI_variant_params = TaskBabiVariantParams(challenge_type='single_supporting_fact_10k')
    task_params = TaskParameters(task_name=taskName, variant_params=task_bAbI_variant_params)
elif taskName == "model_checkerboard":
    task_params = TaskParameters(task_name=taskName, variant_params="2x2_rotated")
else:
    task_params = TaskParameters(task_name=taskName)

task = task_params.create_task()
task.train_with_hyperopt(no_random_samples=200, no_iterations=50)


