from supervised_learning_tasks.task_parameters import TaskParameters
from AL_environment_MDP.al_parameters import ALParameters


def get_application_config(task_name):
    if task_name == "model_UCI":
        variant_params = ['0-adult', '1-australian', '10-spam'][2]
        task_params = TaskParameters(task_name=task_name, dataset=variant_params)
        base_dataset = "UCI"
        usualBatchSize = 4
        usualBatchSize_random = 4
        startingSize = 8
        annotation_budget = 40
        n_jobs = 104  # number of cores to use in parallel
        noRepetitions = 104  # number of runs per agent (for confidence intervals)

    elif task_name == "model_checkerboard":
        task_params = TaskParameters(task_name=task_name, dateset="4x4")
        base_dataset = "checkerboard"
        usualBatchSize = 32
        usualBatchSize_random = 32
        startingSize = 8
        annotation_budget = 200
        n_jobs = 8  # number of cores to use in parallel
        noRepetitions = 32  # number of runs per agent (for confidence intervals)

    elif task_name == "model_Vision":
        task_params = TaskParameters(task_name="model_Vision", dataset="MNIST")
        base_dataset = "CIFAR10"
        usualBatchSize = 32
        usualBatchSize_random = 8
        startingSize = 40
        annotation_budget = 360
        n_jobs = 26  # number of cores to use in parallel
        noRepetitions = 104  # number of runs per agent (for confidence intervals)

    elif task_name == "model_bAbI":
        task_params = TaskParameters(task_name=task_name, dataset="two_supporting_facts_10k")
        base_dataset = "bAbI"
        usualBatchSize = 32
        usualBatchSize_random = 8
        startingSize = 40
        annotation_budget = 360
        n_jobs = 52  # number of cores to use in parallel
        noRepetitions = 52  # number of runs per agent (for confidence intervals)

    else:
        raise ValueError
    al_parameters = ALParameters(annotation_budget=annotation_budget, startingSize=startingSize)
    return task_params, base_dataset, usualBatchSize, usualBatchSize_random, al_parameters, n_jobs, noRepetitions
