from typing import List, Tuple

import numpy as np
from tensorflow.keras.models import Model

from supervised_learning_tasks.task_supervised import Task_supervised

class Task_KERAS(Task_supervised):

    def __init__(self, verboseInit: bool=False):
        self.get_dataset(verboseInit)
        self.model = self.define_model()



    def resetModel(self):
        self.model.set_weights(self.initialWeights)

    def getPredictionShape(self):
        return self.get_y_train().shape

    def get_x_train(self, sampleIDs: List[int]="all") -> list:
        if hasattr(self,"x_train"):
            if sampleIDs=="all":
                return self.x_train
            else:
                return np.concatenate([self.x_train[i,None] for i in sampleIDs])
        else:
            raise NotImplementedError

    def get_y_train(self, sampleIDs: List[int]="all") -> list:
        if hasattr(self,"y_train"):
            if sampleIDs=="all":
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


    def getSamplesSpace(self):
        #get space of predictions
        input_shape = self.get_y_train().shape
        nvec_low = np.zeros(input_shape,np.float)
        nvec_high = np.ones(input_shape,np.float)
        sampleSpace = spaces.Box(nvec_low,nvec_high)
        return sampleSpace

    def getPredictions(self,sampleIDs: List[int]):
        x_train = self.get_x_train(sampleIDs=sampleIDs)
        predictions = self.model.predict(x_train)
        return predictions

    def getModelDependentInfo(self,sampleIDs: List[int]) -> dict:
        sampleInfo = dict()
        x_train = self.get_x_train(sampleIDs=sampleIDs)
        predictions = self.model.predict(x_train)
        sampleInfo["predictions"] = predictions
        return sampleInfo

    def trainFully(self,sampleIDs: list = "all") -> Tuple[float,float]:
        batch_size = 1024
        if sampleIDs != "all" and len(sampleIDs) > batch_size*4:
            batch_size = int(len(sampleIDs)/4)

        epochs = 32
        # train
        x_train = self.get_x_train(sampleIDs)
        y_train = self.get_y_train(sampleIDs)
        x_test = self.get_x_test()
        y_test = self.get_y_test()
        self.model_fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test),
                  withAugmentation=False)
        score = self.model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        return score[0], score[1]

    def getCurrentModelRepr(self):
        return self.model.get_weights()

    def setCurrentModel(self,model_representation):
        self.model.set_weights(model_representation)


    def trainOnBatch(self, sampleIDs: list, epochs: int, resetWeights: bool=True, freezeWeights: bool=False,
                     verbose=False,batch_size=0) -> Tuple[float,float]:

        #get subset to train on
        x_train = self.get_x_train(sampleIDs)
        y_train = self.get_y_train(sampleIDs)
        x_test = self.get_x_test()
        y_test = self.get_y_test()

        if resetWeights:
            self.model.set_weights(self.initialWeights)

        if freezeWeights:
            model_repr = self.getCurrentModelRepr()

        # train
        if batch_size == 0:
            res = self.model_fit(x_train, y_train, epochs=int(epochs),verbose=verbose, withAugmentation=True)
        else:
            res = self.model_fit(x_train, y_train,epochs=int(epochs),batch_size=batch_size, verbose=verbose,withAugmentation=True)
        try:
            loss, acc = self.model.evaluate(x_test,y_test,batch_size=4096,verbose=0)
        except Exception as e:
            debugPoint=0
            raise e
        if False:
            train_loss, train_acc = self.model.evaluate(x_train,y_train,batch_size=4096,verbose=0)
            relevantValues = (len(res.epoch), train_loss, train_acc, loss, acc)
            print("needed %d iterations to have following values: "
                  "loss: %f, acc: %f, val_loss: %f, val_acc:%f " % relevantValues)


        if freezeWeights:
            self.setCurrentModel(model_repr)

        return loss, acc


    def define_model(self) -> Model:
        raise NotImplementedError

    def get_dataset(self,verboseInit=False):
        raise NotImplementedError





