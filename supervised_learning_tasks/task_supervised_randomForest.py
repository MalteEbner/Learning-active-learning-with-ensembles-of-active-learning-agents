from typing import List, Tuple
import copy

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from supervised_learning_tasks.task_supervised import Task_supervised
from hyperopt import fmin, tpe, rand, hp, STATUS_OK, Trials, space_eval

class Task_randomForest(Task_supervised):

    def __init__(self, verboseInit: bool=False):
        self.get_dataset(verboseInit)
        self.classifier = self.define_classifier()


    def getNoTrainingSamples(self) -> int:
        return self.get_y_train().shape[0]

    def resetModel(self):
        self.classifier = self.define_classifier()

    def get_x_train(self, sampleIDs: List[int]="all") -> list:
        if hasattr(self,"x_train"):
            if isinstance(sampleIDs,str) and sampleIDs=="all":
                return self.x_train
            else:
                return np.concatenate([self.x_train[i,None] for i in sampleIDs])
        else:
            raise NotImplementedError

    def get_y_train(self, sampleIDs: List[int]="all") -> list:
        if hasattr(self,"y_train"):
            if isinstance(sampleIDs,str) and sampleIDs=="all":
                return self.y_train
            else:
                return np.concatenate([self.y_train[i,None] for i in sampleIDs])
        else:
            raise NotImplementedError

    def getLabels(self,sampleIDs: List[int]):
        labels = self.get_y_train(sampleIDs)
        return labels

    def get_x_test(self) -> list:
        if hasattr(self,"x_test"):
            return self.x_test
        else:
            raise NotImplementedError

    def get_y_test(self) -> list:
        if hasattr(self,"y_test"):
            return self.y_test
        else:
            raise NotImplementedError

    def define_classifier(self, hyperparamDict: dict={}):
        '''
        these parameters were taken as they were also used in
        Konyushkova, Ksenia, Raphael Sznitman, and Pascal Fua.
        "Learning active learning from data."
        Advances in Neural Information Processing Systems. 2017.
        '''

        n_estimators = int(hyperparamDict.get("n_estimators", 50))

        n_jobs = 1 # prevent running out of threads
        classifier = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs)
        return classifier

    def getHyperparamSpace(self):
        from hyperopt import hp
        paramSpace = dict()
        names = ["n_estimators","max_depth","max_features","min_impurity_decrease","min_samples_split"]
        paramSpace[names[0]] = hp.qloguniform(names[0], np.log(10),np.log(300+1),q=10)
        paramSpace[names[1]] = hp.qloguniform(names[1], np.log(2),np.log(100),q=10)
        #paramSpace[names[2]] = hp.qloguniform(names[2], np.log(40),np.log(60),q=1)
        paramSpace[names[3]] = hp.loguniform(names[3], np.log(1e-7), np.log(1e-3))
        paramSpace[names[4]] = hp.qloguniform(names[4], np.log(2),np.log(10+1),q=2)
        return paramSpace

    def trainWithHyperopt(self,noRandomSamples: int = 1000, noIters: int=100):
        paramSpace = self.getHyperparamSpace()

        # train
        x_test = self.get_x_test()
        y_test = self.get_y_test()
        noTrainingSamples = len(self.get_x_train())

        def objectiveFunction(hyperparamDict:dict,verbose=False) -> float:
            self.classifier = self.define_classifier(hyperparamDict)
            #randomly sample subset
            sampleIDs = np.random.choice(range(noTrainingSamples),size=noRandomSamples,replace=False)
            x_train = self.get_x_train(sampleIDs)
            y_train = self.get_y_train(sampleIDs)
            self.classifier.fit(x_train, y_train)
            acc = self.classifier.score(x_test, y_test)
            print("acc: " + str(acc) + " hyperparams: "+ str(hyperparamDict))
            return -1 * acc

        #perform optimization
        # minimize the objective over the space
        from hyperopt import fmin, tpe
        best = fmin(objectiveFunction, paramSpace, algo=tpe.suggest, max_evals=noIters,verbose=True)
        print("best params: " + str(best))
        bestLoss = objectiveFunction(best,verbose=True)
        print("best validation acc:" + str(bestLoss))

    def get_dataset(self,verboseInit):
        raise NotImplementedError

    def getPredictions(self,sampleIDs: List[int]):
        x_train = self.get_x_train(sampleIDs=sampleIDs)
        predictions = self.classifier.predict_proba(x_train)
        if predictions.shape[1] == 1:
            predictions = 0.5 * np.ones(shape=(predictions.shape[0], 2))
            debugPoint = 0
        return predictions

    def get_samples_repr_1d(self, sampleIDs: List[int] = 'all') -> dict:
        samples_repr_1d = self.get_x_train(sampleIDs)
        return samples_repr_1d

    def getModelDependentInfo(self,sampleIDs: List[int]) -> dict:
        sampleInfo = dict()
        x_train = self.get_x_train(sampleIDs=sampleIDs)
        predictions = self.classifier.predict_proba(x_train)
        if predictions.shape[1] == 1:
            predictions = 0.5 * np.ones(shape=(predictions.shape[0], 2))
            debugPoint = 0
        uncertainties = self.getPredictionUncertainty(x_train)
        sampleInfo["predictions"] = predictions
        sampleInfo["predictionUncertainty"] = uncertainties
        return sampleInfo

    def trainFully(self,sampleIDs: list = "all") -> Tuple[float,float]:

        # train
        x_train = self.get_x_train(sampleIDs)
        y_train = self.get_y_train(sampleIDs)
        x_test = self.get_x_test()
        y_test = self.get_y_test()
        self.model_fit(x_train, y_train)
        acc = self.model.score(x_test, y_test)
        print('Test accuracy:', acc)
        return -1*acc, acc
        
    def getTrueOneStepImprovementLosses(self, currentLabelledSet: List[int], sampleIDs: List[int]) -> List[float]:
        model_representation = self.getCurrentModelRepr()
        newLosses = [self.trainOnBatch([sampleID]+currentLabelledSet,epochs=1,freezeWeights=False,resetWeights=False)[0] for sampleID in sampleIDs]
        self.setCurrentModel(model_representation=model_representation)
        return newLosses

    def trainOnBatch(self, sampleIDs: List[int]) -> Tuple[float, float]:
        #get subset to train on
        x_train = self.get_x_train(sampleIDs)
        y_train = self.get_y_train(sampleIDs)
        x_test = self.get_x_test()
        y_test = self.get_y_test()

        if len(y_train.shape) == 2:
            y_train = y_train[:,0]

        # train
        self.classifier.fit(x_train, y_train)
        acc = self.classifier.score(x_test,y_test)
        loss = -1*acc

        return loss, acc


    def get_dataset(self,verboseInit=False):
        raise NotImplementedError

    def plotSamples(self,filename:str='',sampleIDs='all'):
        x_train = self.get_x_train(sampleIDs)
        y_train = self.get_y_train(sampleIDs)
        x_train_0 = np.stack([x for x,y in zip(x_train,y_train) if y == 0])
        x_train_1 = np.stack([x for x,y in zip(x_train,y_train) if y == 1])

        import matplotlib.pyplot as plt
        plt.scatter(x_train_0[:, 0], x_train_0[:, 1],color='red',label='class 0')
        plt.scatter(x_train_1[:, 0], x_train_1[:, 1], color='blue',label='class 1')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend(loc='upper right')
        if filename != '':
            filename_base = '../'
            filename = filename_base + filename + '.eps'
            plt.savefig(filename, figsize=(6, 4), dpi=320)
        plt.show()

    def plotPredictions(self,sampleIDs):
        x_train = self.get_x_train(sampleIDs)
        y_train = self.getPredictions(sampleIDs)
        x_train_0 = np.stack([x for x,y in zip(x_train,y_train) if y == 0])
        x_train_1 = np.stack([x for x,y in zip(x_train,y_train) if y == 1])

        import matplotlib.pyplot as plt
        plt.scatter(x_train_0[:, 0], x_train_0[:, 1],color='red')
        plt.scatter(x_train_1[:, 0], x_train_1[:, 1], color='blue')
        plt.show()

    def plotPredictionProba(self,sampleIDs):
        x_train = self.get_x_train(sampleIDs)
        y_train = self.classifier.predict_proba(x_train)[:,1]
        self.plotValuesWithColour(sampleIDs,y_train)

    def plotValuesWithColour(self,sampleIDs,colors):
        x_train = self.get_x_train(sampleIDs)
        import matplotlib.pyplot as plt
        plt.scatter(x_train[:, 0], x_train[:, 1],c=colors)
        plt.gray()
        plt.show()


