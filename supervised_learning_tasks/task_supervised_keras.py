from typing import List, Tuple

import numpy as np
from tensorflow.keras.models import Model

from supervised_learning_tasks.task_supervised import TaskSupervised


class TaskKeras(TaskSupervised):

    def __init__(self, verbose_init: bool):
        self.get_dataset(verbose_init)
        self.model = self.define_model()
        self.initial_weights = self.model.get_weights()

    def get_no_training_samples(self) -> int:
        return self.get_y_train().shape[0]

    def reset_model(self):
        self.model.set_weights(self.initial_weights)

    def get_x_train(self, sample_IDs: List[int] = "all") -> np.ndarray:
        if hasattr(self, "x_train"):
            if sample_IDs == "all":
                return self.x_train
            else:
                return np.concatenate([self.x_train[i, None] for i in sample_IDs])
        else:
            raise NotImplementedError

    def get_y_train(self, sample_IDs: List[int] = "all") -> np.ndarray:
        if hasattr(self, "y_train"):
            if sample_IDs == "all":
                return self.y_train
            else:
                return np.concatenate([self.y_train[i, None] for i in sample_IDs])
        else:
            raise NotImplementedError

    def get_x_test(self) -> np.ndarray:
        if hasattr(self, "x_test"):
            return self.x_test
        else:
            raise NotImplementedError

    def get_y_test(self) -> np.ndarray:
        if hasattr(self, "y_test"):
            return self.y_test
        else:
            raise NotImplementedError

    def get_predictions(self, sample_IDs: List[int]) -> np.ndarray:
        x_train = self.get_x_train(sample_IDs=sample_IDs)
        predictions = self.model.predict(x_train)
        return predictions

    def train_on_batch(self, sample_IDs: List[int], verbose=False, batch_size=0) -> Tuple[float, float]:
        epochs = self.get_no_epochs()

        # get subset to train on
        x_train = self.get_x_train(sample_IDs)
        y_train = self.get_y_train(sample_IDs)
        x_test = self.get_x_test()
        y_test = self.get_y_test()

        # train
        if batch_size == 0:
            res = self.model_fit(x_train, y_train, epochs=int(epochs), verbose=verbose, with_augmentation=True)
        else:
            res = self.model_fit(x_train, y_train, epochs=int(epochs), batch_size=batch_size, verbose=verbose,
                                 with_augmentation=True)

        loss, acc = self.model.evaluate(x_test, y_test, batch_size=4096, verbose=0)

        return loss, acc

    def define_model(self) -> Model:
        raise NotImplementedError

    def get_dataset(self, verbose_init=False):
        raise NotImplementedError

    def model_fit(self, x_train, y_train, epochs, batch_size, verbose, with_augmentation):
        raise NotImplementedError

    def get_no_epochs(self) -> int:
        raise NotImplementedError

