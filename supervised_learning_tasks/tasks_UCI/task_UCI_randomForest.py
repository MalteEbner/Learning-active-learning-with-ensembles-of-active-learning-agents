import csv
import os
import re
import pickle as pkl

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from supervised_learning_tasks.task_supervised_randomForest import Task_randomForest




class Task_UCI_randomForest(Task_randomForest):

    def __init__(self,variantParams: str = None, verboseInit: bool=False):
        if variantParams is None:
            variantParams = '0-adult'
        UCI_Datasets = ['0-adult', '1-australian', '2-breast_cancer', '3-diabetis', '4-flare_solar',
                           '5-german', '6-heart', '7-mushrooms', '8-waveform', '9-wdbc', '10-spam']
        if variantParams not in UCI_Datasets:
            raise ValueError
        self.type = variantParams
        Task_randomForest.__init__(self,verboseInit=verboseInit)

    def get_dataset(self,verboseInit):
        if self.type != '10-spam':

            dirname = os.path.dirname(__file__)
            filename = os.path.join(dirname,f'{self.type[2:]}.p')
            data = pkl.load(open(filename, "rb"))
            X = data['X']
            Y = data['y']

        else:
            #taken from https://github.com/sampepose/SpamClassifier/blob/master/load_data.py
            dirname = os.path.dirname(__file__)
            filename = os.path.join(dirname,'spambase.data')
            with open(filename) as f:
                data = []
                reader = csv.reader(f)
                next(reader,None)
                for row in reader:
                    data.append(row)
                X = np.array([x[:-1] for x in data]).astype(np.float)
                Y = np.array([x[-1] for x in data]).astype(np.float)


        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, Y, train_size=0.5)

        scaler = preprocessing.StandardScaler().fit(self.x_train)
        self.x_train = scaler.transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)















