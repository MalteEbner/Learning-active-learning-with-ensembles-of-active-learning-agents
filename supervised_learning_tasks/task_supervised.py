from typing import List, Tuple

import numpy as np
from sklearn.metrics import pairwise_distances


class TaskSupervised():

    def __init__(self):
        pass

    def reset_model(self) -> None:
        raise NotImplementedError

    # get information needed during initialization

    def get_no_training_samples(self) -> int:
        raise NotImplementedError

    def get_predictions(self, sample_IDs: List[int]):
        raise NotImplementedError

    def get_samples_repr_1d(self, sample_IDs: List[int] = 'all') -> dict:
        raise NotImplementedError

    # get functions need for training

    def train_on_batch(self, sample_IDs: List[int]) -> Tuple[float, float]:
        raise NotImplementedError

    def train_with_hyperopt(self, no_random_samples, no_iterations):
        raise NotImplementedError


