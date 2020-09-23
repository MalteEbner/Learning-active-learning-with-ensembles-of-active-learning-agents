import numpy as np
from sklearn import preprocessing

from supervised_learning_tasks.task_supervised_randomForest import Task_randomForest

class Task_Checkerboard_randomForest(Task_randomForest):

    def __init__(self,variantParams: str=None, verboseInit: bool=False):
        if variantParams is None:
            variantParams = '4x4'
        if variantParams not in ['2x2', '4x4', '2x2_rotated']:
            raise ValueError
        self.type = variantParams
        Task_randomForest.__init__(self,verboseInit=verboseInit)

    def get_dataset(self,verboseInit):
        import os
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'checkerboard' + self.type + '_train.npz')
        dt = np.load(filename)
        self.x_train = dt['x']
        self.y_train = dt['y']

        scaler = preprocessing.StandardScaler().fit(self.x_train)
        self.x_train = scaler.transform(self.x_train)

        filename = os.path.join(dirname, 'checkerboard' + self.type + '_test.npz')
        dt = np.load(filename)
        self.x_test = dt['x']
        self.y_test = dt['y']
        self.x_test = scaler.transform(self.x_test)















