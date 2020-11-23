from supervised_learning_tasks.task_parameters import TaskParameters


# define AL run
'''
possible tasks: 
model_bAbI_memoryNetwork, model_Vision, model_checkerboard
'''
taskName = ["model_Vision", "model_bAbI_memoryNetwork", "model_checkerboard"][1]
if taskName == "model_Vision":
    task_params = TaskParameters(task_name="model_Vision", dataset="fashion")
elif taskName == "model_bAbI_memoryNetwork":
    task_params = TaskParameters(task_name="model_Vision", dataset="two_supporting_facts")
elif taskName == "model_checkerboard":
    task_params = TaskParameters(task_name=taskName, variant_params="2x2_rotated")
else:
    task_params = TaskParameters(task_name=taskName)

task = task_params.create_task()
task.train_with_hyperopt(no_random_samples=200, no_iterations=50)


