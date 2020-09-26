from supervised_learning_tasks.task_parameters import TaskParameters


def _test_task_params(task_params):
    task = task_params.create_task()
    task.reset_model()
    labelled_IDs = list(range(20))
    loss, accuracy = task.train_on_batch(labelled_IDs)


def test_task_checkerboard():
    task_params = TaskParameters(task_name="model_checkerboard", dataset="2x2_rotated")
    _test_task_params(task_params)


def test_task_uci():
    task_params = TaskParameters(task_name="model_UCI", dataset="10-spam")
    _test_task_params(task_params)


def test_task_vision():
    task_params = TaskParameters(task_name="model_Vision", dataset='MNIST')
    _test_task_params(task_params)


def test_task_babi():
    task_params = TaskParameters(task_name="model_bAbI", dataset='single_supporting_fact')
    _test_task_params(task_params)
