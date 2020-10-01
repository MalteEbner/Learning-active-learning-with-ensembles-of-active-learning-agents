from supervised_learning_tasks.task_parameters import TaskParameters
from AL_environment_MDP.al_parameters import ALParameters


def get_application_config(task_name):
    if task_name == "model_UCI":
        variant_params = ['0-adult', '1-australian', '10-spam'][2]
        task_params = TaskParameters(task_name=task_name, dataset=variant_params)
        base_dataset = "UCI"
        usual_batch_size = 8
        starting_size = 8
        annotation_budget = 40
        n_jobs = 4  # number of cores to use in parallel
        no_repetitions = 64  # number of runs per agent (for confidence intervals)

    elif task_name == "model_checkerboard":
        task_params = TaskParameters(task_name=task_name, dataset="4x4")
        base_dataset = "checkerboard"
        usual_batch_size = 32
        starting_size = 40
        annotation_budget = 200
        n_jobs = 8  # number of cores to use in parallel
        no_repetitions = 64  # number of runs per agent (for confidence intervals)

    elif task_name == "model_Vision":
        task_params = TaskParameters(task_name="model_Vision", dataset="fashion")
        base_dataset = "MNIST"
        usual_batch_size = 64
        starting_size = 40
        annotation_budget = 360
        n_jobs = 1  # number of cores to use in parallel
        no_repetitions = 10  # number of runs per agent (for confidence intervals)

    elif task_name == "model_bAbI":
        task_params = TaskParameters(task_name=task_name, dataset="two_supporting_facts")
        base_dataset = "bAbI"
        usual_batch_size = 64
        starting_size = 40
        annotation_budget = 360
        n_jobs = 1  # number of cores to use in parallel
        no_repetitions = 60  # number of runs per agent (for confidence intervals)

    else:
        raise ValueError
    al_parameters = ALParameters(annotation_budget=annotation_budget, starting_size=starting_size)
    return task_params, base_dataset, usual_batch_size, al_parameters, n_jobs, no_repetitions

