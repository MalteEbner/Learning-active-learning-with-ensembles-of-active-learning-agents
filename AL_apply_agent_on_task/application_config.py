
from supervised_learning_tasks.task_parameters import Task_Parameters
from supervised_learning_tasks.tasks_vision.task_Vision_variantParams import Task_Vision_variantParams
from supervised_learning_tasks.tasks_QA_bAbI.task_bAbI_variantParams import Task_bAbI_variantParams
from AL_environment_MDP.al_parameters import AL_Parameters


def get_application_config(taskName):

    if taskName == "model_UCI":
        variantParams = ['0-adult' ,'1-australian' ,'10-spam'][2]
        task_params = Task_Parameters(taskName=taskName ,variantParams=variantParams)
        base_dataset = "UCI"
        usualBatchSize = 4
        usualBatchSize_random = 4
        startingSize = 8
        annotation_budget = 40
        n_jobs = 104  # number of cores to use in parallel
        noRepetitions = 104  # number of runs per agent (for confidence intervals)

    elif taskName == "model_checkerboard":
        task_params = Task_Parameters(taskName=taskName ,variantParams="4x4")
        base_dataset = "checkerboard"
        usualBatchSize = 32
        usualBatchSize_random = 32
        startingSize = 8
        annotation_budget = 200
        n_jobs = 12  # number of cores to use in parallel
        noRepetitions = 12  # number of runs per agent (for confidence intervals)

    elif taskName == "model_Vision":
        variantParams = Task_Vision_variantParams(dataset='MNIST', repr_1d_type='tSNE')
        task_params = Task_Parameters(taskName="model_Vision", variantParams=variantParams)
        base_dataset = "CIFAR10"
        usualBatchSize = 32
        usualBatchSize_random = 8
        startingSize = 40
        annotation_budget = 360
        n_jobs = 26  # number of cores to use in parallel
        noRepetitions = 104  # number of runs per agent (for confidence intervals)

    elif taskName == "model_bAbI_memoryNetwork":
        task_bAbI_variantParams = Task_bAbI_variantParams(challenge_type='single_supporting_fact_10k')
        task_bAbI_variantParams = Task_bAbI_variantParams(challenge_type='two_supporting_facts_10k')
        task_params = Task_Parameters(taskName=taskName ,variantParams=task_bAbI_variantParams)
        base_dataset = "bAbI"
        usualBatchSize = 32
        usualBatchSize_random = 8
        startingSize = 40
        annotation_budget = 360
        n_jobs = 52  # number of cores to use in parallel
        noRepetitions = 52  # number of runs per agent (for confidence intervals)

    else:
        raise ValueError
    al_parameters = AL_Parameters(annotationBudget=annotation_budget, startingSize=startingSize)
    return task_params, base_dataset, usualBatchSize, usualBatchSize_random, al_parameters, n_jobs, noRepetitions
