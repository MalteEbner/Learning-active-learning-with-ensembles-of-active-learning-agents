from typing import List, Tuple
import copy

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from supervised_learning_tasks.task_supervised import TaskSupervised
from hyperopt import fmin, tpe, rand, hp, STATUS_OK, Trials, space_eval


class TaskRandomForest(TaskSupervised):

    def __init__(self, verbose_init: bool = False):
        self.get_dataset(verbose_init)
        self.classifier = self.define_classifier()

    def get_no_training_samples(self) -> int:
        return self.get_y_train().shape[0]

    def reset_model(self):
        self.classifier = self.define_classifier()

    def get_x_train(self, sample_IDs: List[int] = "all") -> np.ndarray:
        if hasattr(self, "x_train"):
            if isinstance(sample_IDs, str) and sample_IDs == "all":
                return self.x_train
            else:
                return np.concatenate([self.x_train[i, None] for i in sample_IDs])
        else:
            raise NotImplementedError

    def get_y_train(self, sample_IDs: List[int] = "all") -> np.ndarray:
        if hasattr(self, "y_train"):
            if isinstance(sample_IDs, str) and sample_IDs == "all":
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

    def define_classifier(self, hyperparamDict: dict = {}):
        '''
        these parameters were taken as they were also used in
        Konyushkova, Ksenia, Raphael Sznitman, and Pascal Fua.
        "Learning active learning from data."
        Advances in Neural Information Processing Systems. 2017.
        '''

        n_estimators = int(hyperparamDict.get("n_estimators", 50))

        n_jobs = 1  # prevent running out of threads
        classifier = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs)
        return classifier

    def get_hyperparam_space(self):
        from hyperopt import hp
        paramSpace = dict()
        names = ["n_estimators", "max_depth", "max_features", "min_impurity_decrease", "min_samples_split"]
        paramSpace[names[0]] = hp.qloguniform(names[0], np.log(10), np.log(300 + 1), q=10)
        paramSpace[names[1]] = hp.qloguniform(names[1], np.log(2), np.log(100), q=10)
        # paramSpace[names[2]] = hp.qloguniform(names[2], np.log(40),np.log(60),q=1)
        paramSpace[names[3]] = hp.loguniform(names[3], np.log(1e-7), np.log(1e-3))
        paramSpace[names[4]] = hp.qloguniform(names[4], np.log(2), np.log(10 + 1), q=2)
        return paramSpace

    def train_with_hyperopt(self, no_random_samples: int = 1000, no_iters: int = 100):
        param_space = self.get_hyperparam_space()

        # train
        x_test = self.get_x_test()
        y_test = self.get_y_test()
        no_training_samples = len(self.get_x_train())

        def objective_function(hyperparam_dict: dict, verbose=False) -> float:
            self.classifier = self.define_classifier(hyperparam_dict)
            # randomly sample subset
            sample_IDs = np.random.choice(range(no_training_samples), size=no_random_samples, replace=False)
            x_train = self.get_x_train(sample_IDs)
            y_train = self.get_y_train(sample_IDs)
            self.classifier.fit(x_train, y_train)
            acc = self.classifier.score(x_test, y_test)
            print("acc: " + str(acc) + " hyperparams: " + str(hyperparam_dict))
            return -1 * acc

        # perform optimization
        # minimize the objective over the space
        from hyperopt import fmin, tpe
        best = fmin(objective_function, param_space, algo=tpe.suggest, max_evals=no_iters, verbose=True)
        print("best params: " + str(best))
        bestLoss = objective_function(best, verbose=True)
        print("best validation acc:" + str(bestLoss))

    def get_dataset(self, verboseInit):
        raise NotImplementedError

    def get_predictions(self, sample_IDs: List[int]):
        x_train = self.get_x_train(sample_IDs=sample_IDs)
        predictions = self.classifier.predict_proba(x_train)
        if predictions.shape[1] == 1:
            predictions = 0.5 * np.ones(shape=(predictions.shape[0], 2))
            debugPoint = 0
        return predictions

    def get_samples_repr_1d(self, sample_IDs: List[int] = 'all') -> np.ndarray:
        samples_repr_1d = self.get_x_train(sample_IDs)
        return samples_repr_1d

    def train_on_batch(self, sample_IDs: List[int]) -> Tuple[float, float]:
        # get subset to train on
        x_train = self.get_x_train(sample_IDs)
        y_train = self.get_y_train(sample_IDs)
        x_test = self.get_x_test()
        y_test = self.get_y_test()

        if len(y_train.shape) == 2:
            y_train = y_train[:, 0]

        # train
        self.classifier.fit(x_train, y_train)
        acc = self.classifier.score(x_test, y_test)
        loss = -1 * acc

        return loss, acc

    def get_dataset(self, verboseInit=False):
        raise NotImplementedError

    def plot_samples(self, filename: str = '', sample_IDs='all'):
        x_train = self.get_x_train(sample_IDs)
        y_train = self.get_y_train(sample_IDs)
        x_train_0 = np.stack([x for x, y in zip(x_train, y_train) if y == 0])
        x_train_1 = np.stack([x for x, y in zip(x_train, y_train) if y == 1])

        import matplotlib.pyplot as plt
        plt.scatter(x_train_0[:, 0], x_train_0[:, 1], color='red', label='class 0')
        plt.scatter(x_train_1[:, 0], x_train_1[:, 1], color='blue', label='class 1')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend(loc='upper right')
        if filename != '':
            filename_base = '../'
            filename = filename_base + filename + '.eps'
            plt.savefig(filename, figsize=(6, 4), dpi=320)
        plt.show()

    def plot_predictions(self, sample_IDs):
        x_train = self.get_x_train(sample_IDs)
        y_train = self.get_predictions(sample_IDs)
        x_train_0 = np.stack([x for x, y in zip(x_train, y_train) if y == 0])
        x_train_1 = np.stack([x for x, y in zip(x_train, y_train) if y == 1])

        import matplotlib.pyplot as plt
        plt.scatter(x_train_0[:, 0], x_train_0[:, 1], color='red')
        plt.scatter(x_train_1[:, 0], x_train_1[:, 1], color='blue')
        plt.show()

    def plot_prediction_probabilities(self, sample_IDs):
        x_train = self.get_x_train(sample_IDs)
        y_train = self.classifier.predict_proba(x_train)[:, 1]
        self.plot_values_with_colour(sample_IDs, y_train)

    def plot_values_with_colour(self, sample_IDs, colors):
        x_train = self.get_x_train(sample_IDs)
        import matplotlib.pyplot as plt
        plt.scatter(x_train[:, 0], x_train[:, 1], c=colors)
        plt.gray()
        plt.show()
