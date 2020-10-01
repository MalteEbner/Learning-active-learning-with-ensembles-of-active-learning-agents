from typing import Dict, Union

from supervised_learning_tasks.task_supervised import TaskSupervised
from supervised_learning_tasks.tasks_UCI.task_UCI_random_forest import TaskUciRandomForest
from supervised_learning_tasks.tasks_checkerboard.task_checkerboard_randomForest import TaskCheckerboardRandomForest
from supervised_learning_tasks.tasks_QA_bAbI.task_bAbI_memoryNetwork import TaskBabiMemoryNetwork
from supervised_learning_tasks.tasks_vision.task_Vision_CNN import TaskVisionCNN


class TaskParameters:
    def __init__(self, task_name: str, dataset: str = None):
        self.task_name = task_name
        self.dataset = dataset

    def create_task(self, verbose_init=False) -> TaskSupervised:
        task_mapping = {
            'model_bAbI': TaskBabiMemoryNetwork,
            'model_Vision': TaskVisionCNN,
            'model_checkerboard': TaskCheckerboardRandomForest,
            'model_UCI': TaskUciRandomForest}
        if self.task_name not in task_mapping.keys():
            raise ValueError(f"task_name unknown (value: {self.task_name})")
        task_class = task_mapping[self.task_name]

        if self.dataset is None:
            task = task_class(verbose_init=verbose_init)
        else:
            task = task_class(dataset=self.dataset,verbose_init=verbose_init)
        return task

    def get_experiment_filename(self):
        filename = 'Experiments/results/'
        filename += self.__short_repr__()
        filename += ' experiments'
        return filename

    def __repr__(self):
        self_dict = self.__dict__
        return str([self_dict[key] for key in sorted(self_dict.keys(), reverse=False)])

    def __short_repr__(self):
        if 'Vision' in self.task_name:
            name = self.dataset
        else:
            name = self.task_name + ' ' + self.dataset
        name = name.replace('model_','')
        name = name.replace('_',' ')
        return name

    def __eq__(self, other):
        conditions = list()
        conditions.append(self.task_name == other.task_name)
        conditions.append(self.dataset == other.dataset)
        return all(conditions)

    def __ne__(self, other):
        return not self == other
