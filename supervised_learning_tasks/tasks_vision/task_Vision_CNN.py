from abc import ABC
from collections import namedtuple
from typing import List

import numpy as np
from tensorflow.keras import backend
from tensorflow.keras import losses
from tensorflow.keras import utils
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l1_l2

from supervised_learning_tasks.task_supervised_keras import TaskKeras
from supervised_learning_tasks.tasks_vision.task_Vision_1drepr_getter import Vision1dRepr


class TaskVisionCNN(TaskKeras):
    def __init__(self, dataset: str = 'MNIST', verbose_init: bool = False, ):

        if dataset not in ['MNIST', 'fashion']:
            raise ValueError

        self.get_dataset = lambda verbose: self.get_dataset_mnist(verbose, dataset=dataset)

        self.vision_1d_repr_getter = Vision1dRepr(dataset=dataset,_type='tSNE', n_components=3)
        TaskKeras.__init__(self, verbose_init=verbose_init)

    def get_loss_function(self):
        return losses.CategoricalCrossentropy

    def model_fit(self, x_train, y_train, epochs, verbose, batch_size=16, withAugmentation=True):
        batch_size = min(len(x_train), batch_size)

        if withAugmentation:
            # print('Using real-time data augmentation.')
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
                # value used for fill_mode = 'constant'
                cval=0.,
                # randomly flip images
                vertical_flip=False,
                # set rescaling factor (applied before any other transformation)
                rescale=None,
                # set function that will be applied on each input
                preprocessing_function=None,
                # image data format, either 'channels_first' or 'channels_last'
                data_format=None,
                # fraction of images reserved for validation (strictly between 0 and 1)
                validation_split=0.0)
        else:
            datagen = ImageDataGenerator()

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)
        flowing_datagen = datagen.flow(x_train, y_train, batch_size=batch_size)

        # Fit the model on the batches generated by datagen.flow().
        return self.model.fit(flowing_datagen, steps_per_epoch=len(x_train) / batch_size,
                              epochs=epochs, verbose=verbose, workers=1, use_multiprocessing=0)

    def get_samples_repr_1d(self, sample_IDs: List[int] = 'all', filename=''):

        if not hasattr(self, 'x_train_repr_1d'):
            self.x_train_repr_1d = self.vision_1d_repr_getter.get_repr_from_file(self.x_train)

        if isinstance(sample_IDs, str) and sample_IDs == 'all':
            return self.x_train_repr_1d
        else:
            x_train_repr_1d = np.take(self.x_train_repr_1d, indices=sample_IDs, axis=0)
            return x_train_repr_1d

    def define_model(self, params: dict = None):

        if params is None:
            params = dict()
        regularizationFactor_l1 = params.get('regularization_factor_l1', 3.1e-4)
        regularizationFactor_l2 = params.get('regularization_factor_l2', 2.1e-4)
        dropoutRate = params.get('dropout_rate', 0.25)
        learningRate = params.get('learning_rate', 4.1e-4)
        optimizer = params.get('optimizer', Adam)
        neuronsLayer1 = int(params.get('neurons_layer1', 19))
        neuronsLayer2 = int(params.get('neurons_layer2', 110))
        neuronsLayer3 = int(params.get('neurons_layer3', 85))
        no_epochs = int(params.get('no_epochs', self.get_no_epochs()))
        kernel_regularizer = l1_l2(regularizationFactor_l1, regularizationFactor_l2)

        model = Sequential()
        model.add(Conv2D(neuronsLayer1, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.data_params.input_shape,
                         kernel_regularizer=kernel_regularizer)
                  )
        model.add(Conv2D(neuronsLayer2, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropoutRate))
        model.add(Flatten())
        model.add(Dense(neuronsLayer3, activation='relu'))
        model.add(Dropout(dropoutRate))
        model.add(Dense(self.data_params.num_classes, activation='softmax'))

        optimizer = optimizer(lr=learningRate)
        model.compile(loss=losses.categorical_crossentropy,
                      optimizer=optimizer,
                      metrics=['accuracy'])

        return model

    def train_with_hyperopt(self, no_random_samples: int = 1000, no_iterations: int = 100):
        from hyperopt import hp

        param_space = dict()
        # optimizers = [SGD, Adam,Adagrad,RMSprop]
        # optimizers = optimizers[1:2]
        names = ['regularization_factor_l1',
                 'regularization_factor_l2', 'dropout_rate', 'learning_rate', 'optimizer',
                 'neurons_layer1', 'neurons_layer2', 'neurons_layer3', 'no_epochs',
                 'batch_size']
        # param_space[names[0]] = hp.loguniform(names[0], np.log(6e-5), np.log(3e-3))

        # param_space[names[1]] = hp.loguniform(names[1], np.log(11e-5), np.log(3e-3))
        param_space[names[2]] = hp.uniform(names[2], 0.1, 0.3)
        param_space[names[3]] = hp.loguniform(names[3], np.log(1e-4), np.log(30e-4))
        # param_space[names[4]] = hp.choice(names[4], optimizers)

        # param_space[names[5]] = hp.qloguniform(names[5], np.log(12),np.log(128),q=1)
        # param_space[names[6]] = hp.qloguniform(names[6], np.log(12), np.log(128), q=1)
        # param_space[names[7]] = hp.qloguniform(names[7], np.log(12), np.log(128), q=1)
        # param_space[names[8]] = hp.qloguniform(names[8], np.log(20), np.log(40), q=1)

        param_space[names[9]] = hp.qloguniform(names[9], np.log(8), np.log(64), q=8)

        # define callbacks and objective function
        early_stopping_callback = EarlyStopping(patience=3, verbose=0, restore_best_weights=True)
        callbacks = [early_stopping_callback]
        callbacks = []

        no_training_samples = len(self.get_x_train())

        def objective_function(hyperparam_dict: dict, verbose=False) -> float:
            self.model = self.define_model(hyperparam_dict)
            # randomly sample subset
            sample_IDs = np.random.choice(range(no_training_samples), size=no_random_samples, replace=False)
            noEpochs = 32  # int(hyperparam_dict.get('noEpochs',100))
            batchSize = int(hyperparam_dict.get('batch_size', 16))
            loss, acc = self.train_on_batch(sample_IDs, noEpochs, batch_size=batchSize)
            print('loss:' + str(loss) + ' acc: ' + str(acc) + ' hyperparams: ' + str(hyperparam_dict))
            if np.isnan(loss):
                print('ERROR: validation loss is nan with following hyperparams:')
                print(hyperparam_dict)
                raise ValueError
            return loss

        # perform optimization
        # minimize the objective over the space
        from hyperopt import fmin, tpe
        best = fmin(objective_function, param_space, algo=tpe.suggest, max_evals=no_iterations, verbose=True)
        print('best params: ' + str(best))
        bestLoss = objective_function(best, verbose=True)
        print('best validation acc:' + str(bestLoss))
        # print('weights:' + str(list(self.model.get_weights()[0])))

    def get_dataset_mnist(self, verboseInit, dataset='MNIST') -> None:
        num_classes = 10

        # input image dimensions
        img_rows, img_cols = 28, 28

        # the data, split between train and test sets
        if dataset == 'MNIST':
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
        elif dataset == 'fashion':
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        else:
            raise ValueError

        if backend.image_data_format() == 'channels_first':
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

        DataParams = namedtuple('data_params', 'num_classes input_shape')
        self.data_params = DataParams(num_classes, input_shape)

    def get_no_epochs(self) -> int:
        return 21

