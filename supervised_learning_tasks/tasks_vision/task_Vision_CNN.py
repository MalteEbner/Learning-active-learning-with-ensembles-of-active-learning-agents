from collections import namedtuple
from typing import List
import time

import numpy as np
from tensorflow.keras import utils
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2, l1_l2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.datasets import cifar10, mnist, fashion_mnist
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.applications import resnet50
from tensorflow.compat.v1.image import resize_images
from tensorflow import stack, concat, squeeze

import jsonpickle


from supervised_learning_tasks.task_supervised_keras import Task_KERAS
from supervised_learning_tasks.tasks_vision.task_Vision_variantParams import Task_Vision_variantParams


class Task_Vision_CNN(Task_KERAS):
    """
    @param verboseInit: True if information about dataset should be printed, else False
    """
    def __init__(self, verboseInit: bool=False, variantParams: Task_Vision_variantParams=None):
        if variantParams==None:
            variantParams=Task_Vision_variantParams()

        if variantParams.dataset == "MNIST":
            self.get_dataset = self.get_dataset_MNIST
        elif variantParams.dataset == "fashion":
            self.get_dataset = lambda verbose: self.get_dataset_MNIST(verbose,type="fashion")
        else:
            raise ValueError

        self.vision_1dRepr = variantParams.vision_1dRepr


        self.dataset = variantParams.dataset
        self.regularizationFactor = variantParams.regularizationFactor
        self.initialLearningRate = variantParams.initLR
        self.validationPatience = variantParams.valPat

        Task_KERAS.__init__(self,no_epochs=variantParams.no_epochs,verboseInit=verboseInit)


    def getModelIndependentInfo(self,sampleIDs: List[int]) -> dict:
        sampleInfo = dict()
        imageFeatures = self.getImageFeatures(sampleIDs=sampleIDs)
        sampleInfo["samples_repr_1d"] = imageFeatures
        return sampleInfo






    def getLossFunction(self):
        return losses.CategoricalCrossentropy

    def model_fit(self,x_train, y_train,epochs,verbose,batch_size=16,withAugmentation=True):
        batch_size = min(len(x_train),batch_size)





        if withAugmentation:
            #print('Using real-time data augmentation.')
            # This will do preprocessing and realtime data augmentation:
            datagen = ImageDataGenerator(
                # set input mean to 0 over the dataset
                featurewise_center=False,
                # set each sample mean to 0
                samplewise_center=False,
                # divide inputs by std of dataset
                featurewise_std_normalization=False,
                # divide each input by its std
                samplewise_std_normalization=False,
                # apply ZCA whitening
                zca_whitening=False,
                # epsilon for ZCA whitening
                zca_epsilon=1e-06,
                # randomly rotate images in the range (deg 0 to 180)
                rotation_range=0,
                # randomly shift images horizontally
                width_shift_range=0.1,
                # randomly shift images vertically
                height_shift_range=0.1,
                # set range for random shear
                shear_range=0.1,
                # set range for random zoom
                zoom_range=0.1,
                # set range for random channel shifts
                channel_shift_range=0.,
                # set mode for filling points outside the input boundaries
                fill_mode='nearest',
                # value used for fill_mode = "constant"
                cval=0.,
                # randomly flip images
                vertical_flip=False,
                # set rescaling factor (applied before any other transformation)
                rescale=None,
                # set function that will be applied on each input
                preprocessing_function=None,
                # image data format, either "channels_first" or "channels_last"
                data_format=None,
                # fraction of images reserved for validation (strictly between 0 and 1)
                validation_split=0.0)
        else:
            datagen = ImageDataGenerator()

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)
        flowingDatagen = datagen.flow(x_train, y_train, batch_size=batch_size)

        # Fit the model on the batches generated by datagen.flow().
        return self.model.fit(flowingDatagen,steps_per_epoch=len(x_train)/batch_size,
                                 epochs=epochs, verbose=verbose, workers=1, use_multiprocessing=0)

    def getImageFeatures(self,sampleIDs: List[int] = 'all',filename=""):

        if not hasattr(self,'x_train_repr_1d'):
            self.x_train_repr_1d = self.vision_1dRepr.getRepr_fromFile(self.x_train)

        #x_train_repr_1d = self.x_train_repr_1d.numpy()
        if isinstance(sampleIDs,str) and sampleIDs == 'all':
            return self.x_train_repr_1d
        else:
            x_train_repr_1d = np.take(self.x_train_repr_1d,indices=sampleIDs,axis=0)
            return x_train_repr_1d

    def define_model(self,params: dict=None):


        if params == None:
            params = dict()
        regularizationFactor_l1 = params.get("regularizationFactor_l1",3.1e-4)
        regularizationFactor_l2 = params.get("regularizationFactor_l2", 2.1e-4)
        dropoutRate = params.get("dropoutRate",0.25)
        learningRate = params.get("learningRate",4.1e-4)
        optimizer = params.get("optimizer",Adam)
        neuronsLayer1 = int(params.get("neuronsLayer1",19))
        neuronsLayer2 = int(params.get("neuronsLayer2", 110))
        neuronsLayer3 = int(params.get("neuronsLayer3", 85))
        noEpochs = 21
        kernel_regularizer = l1_l2(regularizationFactor_l1, regularizationFactor_l2)


        model = Sequential()
        model.add(Conv2D(neuronsLayer1, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.dataParams.input_shape,
                         kernel_regularizer=kernel_regularizer)
                  )
        model.add(Conv2D(neuronsLayer2, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropoutRate))
        model.add(Flatten())
        model.add(Dense(neuronsLayer3, activation='relu'))
        model.add(Dropout(dropoutRate))
        model.add(Dense(self.dataParams.num_classes, activation='softmax'))

        optimizer = optimizer(lr=learningRate)
        model.compile(loss=losses.categorical_crossentropy,
                      optimizer=optimizer,
                      metrics=['accuracy'])

        self.initialWeights = model.get_weights()
        return model

    def trainWithHyperopt(self,noRandomSamples: int = 1000, noIters: int=100):
        from hyperopt import fmin, tpe, rand, hp, STATUS_OK, Trials, space_eval

        paramSpace = dict()
        #optimizers = [SGD, Adam,Adagrad,RMSprop]
        #optimizers = optimizers[1:2]
        names = ["regularizationFactor_l1",
                 "regularizationFactor_l2","dropoutRate","learningRate","optimizer",
                 "neuronsLayer1","neuronsLayer2", "neuronsLayer3", "noEpochs",
                 "batchSize"]
        #paramSpace[names[0]] = hp.loguniform(names[0], np.log(6e-5), np.log(3e-3))

        #paramSpace[names[1]] = hp.loguniform(names[1], np.log(11e-5), np.log(3e-3))
        paramSpace[names[2]] = hp.uniform(names[2], 0.1, 0.3)
        paramSpace[names[3]] = hp.loguniform(names[3], np.log(1e-4), np.log(30e-4))
        #paramSpace[names[4]] = hp.choice(names[4], optimizers)

        #paramSpace[names[5]] = hp.qloguniform(names[5], np.log(12),np.log(128),q=1)
        #paramSpace[names[6]] = hp.qloguniform(names[6], np.log(12), np.log(128), q=1)
        #paramSpace[names[7]] = hp.qloguniform(names[7], np.log(12), np.log(128), q=1)
        #paramSpace[names[8]] = hp.qloguniform(names[8], np.log(20), np.log(40), q=1)

        paramSpace[names[9]] = hp.qloguniform(names[9], np.log(8), np.log(64), q=8)

        #define callbacks and objective function
        earlyStoppingCallback = EarlyStopping(patience=3, verbose=0, restore_best_weights=True)
        callbacks = [earlyStoppingCallback]
        callbacks = []

        # train
        x_test = self.get_x_test()
        y_test = self.get_y_test()
        noTrainingSamples = len(self.get_x_train())

        def objectiveFunction(hyperparamDict:dict,verbose=False) -> float:
            self.model = self.define_model(hyperparamDict)
            #randomly sample subset
            sampleIDs = np.random.choice(range(noTrainingSamples),size=noRandomSamples,replace=False)
            noEpochs = 32#int(hyperparamDict.get('noEpochs',100))
            batchSize = int(hyperparamDict.get('batchSize',16))
            loss, acc = self.trainOnBatch(sampleIDs,noEpochs,batch_size=batchSize,resetWeights=False)
            print("loss:" + str(loss) + " acc: " + str(acc) + " hyperparams: "+ str(hyperparamDict))
            if np.isnan(loss):
                print("ERROR: validation loss is nan with following hyperparams:")
                print(hyperparamDict)
                raise ValueError
            return loss

        #perform optimization
        # minimize the objective over the space
        from hyperopt import fmin, tpe
        best = fmin(objectiveFunction, paramSpace, algo=tpe.suggest, max_evals=noIters,verbose=True)
        print("best params: " + str(best))
        bestLoss = objectiveFunction(best,verbose=True)
        print("best validation acc:" + str(bestLoss))
        #print("weights:" + str(list(self.model.get_weights()[0])))


    def get_dataset_MNIST(self,verboseInit, type="MNIST"):
        num_classes = 10

        # input image dimensions
        img_rows, img_cols = 28, 28

        # the data, split between train and test sets
        if type == "MNIST":
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
        elif type == "fashion":
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        else:
            raise ValueError


        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        self.x_train = x_train / 255
        self.x_test = x_test / 255

        if verboseInit:
            print('x_train shape:', x_train.shape)
            print(x_train.shape[0], 'train samples')
            print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        self.y_train = utils.to_categorical(y_train, num_classes)
        self.y_test = utils.to_categorical(y_test, num_classes)

        DataParams = namedtuple('dataParams', 'num_classes input_shape')
        self.dataParams = DataParams(num_classes,input_shape)