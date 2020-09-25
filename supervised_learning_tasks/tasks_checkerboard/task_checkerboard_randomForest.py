import numpy as np
from sklearn import preprocessing

from supervised_learning_tasks.task_supervised_randomForest import TaskRandomForest


class TaskCheckerboardRandomForest(TaskRandomForest):
    def __init__(self, dataset: str = '4x4', verbose_init: bool = False):
        if dataset not in ['2x2', '4x4', '2x2_rotated']:
            raise ValueError
        self.dataset = dataset
        TaskRandomForest.__init__(self, verbose_init=verbose_init)

    def get_dataset(self, verbose_init):
        import os
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'checkerboard' + self.dataset + '_train.npz')
        dt = np.load(filename)
        self.x_train = dt['x']
        self.y_train = dt['y']

        scaler = preprocessing.StandardScaler().fit(self.x_train)
        self.x_train = scaler.transform(self.x_train)

        filename = os.path.join(dirname, 'checkerboard' + self.dataset + '_test.npz')
        dt = np.load(filename)
        self.x_test = dt['x']
        self.y_test = dt['y']
        self.x_test = scaler.transform(self.x_test)
