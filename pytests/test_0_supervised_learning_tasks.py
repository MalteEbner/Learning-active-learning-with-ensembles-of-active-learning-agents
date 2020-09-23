from supervised_learning_tasks.task_parameters import Task_Parameters
from supervised_learning_tasks.tasks_vision.task_Vision_variantParams import Task_Vision_variantParams
from supervised_learning_tasks.tasks_QA_bAbI.task_bAbI_variantParams import Task_bAbI_variantParams

def _test_task_params(task_params):
    task = task_params.createTask()
    task.resetModel()
    labelled_IDs = list(range(20))
    loss, accuracy = task.trainOnBatch(labelled_IDs)

def test_task_checkerboard():
    task_params = Task_Parameters(taskName="model_checkerboard",variantParams="2x2_rotated")
    _test_task_params(task_params)

def test_task_UCI():
    task_params = Task_Parameters(taskName="model_UCI",variantParams="10-spam")
    _test_task_params(task_params)

def test_task_vision():
    task_vision_variant_params = Task_Vision_variantParams(dataset="MNIST",repr_1d_type='tSNE')
    task_params = Task_Parameters(taskName="model_Vision", variantParams=task_vision_variant_params)
    _test_task_params(task_params)

def test_task_bAbI():
    task_bAbI_variantParams = Task_bAbI_variantParams(challenge_type='two_supporting_facts_10k')
    task_params = Task_Parameters(taskName="model_bAbI", variantParams=task_bAbI_variantParams)
    _test_task_params(task_params)



