from typing import List

import numpy as np
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min

from supervised_learning_tasks.task_supervised import TaskSupervised


class Observation:
    def __init__(self, task: TaskSupervised, labelled_IDs, unlabelled_IDs, batch_IDs):
        self.task = task
        self.labelled_IDs = labelled_IDs
        self.unlabelled_IDs = unlabelled_IDs
        self.batch_IDs = batch_IDs

        self.batch_independent_features = dict()

    def update_features_based_on_action(self, action_sample_IDs: List[int]):
        if hasattr(self, "min_distances_to_labelled_batch"):
            delattr(self, "min_distances_to_labelled_batch")
        for (feature_name,feature_value) in self.batch_independent_features.items():
            feature_value_with_deleted_sample = np.delete(feature_value, action_sample_IDs, axis=0)
            self.batch_independent_features[feature_name] = feature_value_with_deleted_sample
        self.unlabelled_IDs = np.delete(self.unlabelled_IDs, action_sample_IDs, axis=0)

    def get_prediction_entropies(self):
        if not 'predictions' in self.batch_independent_features:
            self.batch_independent_features['predictions'] = self.task.get_predictions(self.unlabelled_IDs)
        if not 'prediction_entropies' in self.batch_independent_features:
            probs = self.batch_independent_features['predictions']
            zeros = np.zeros_like(probs)
            log_probs = np.log2(probs, out=zeros, where=probs > 0)
            entropies = -1 * np.sum(probs * log_probs, axis=1)
            self.batch_independent_features['prediction_entropies'] = entropies
        return self.batch_independent_features['prediction_entropies']

    def get_min_distances_to_labelled_and_batch(self):
        if not hasattr(self,"min_distances_to_labelled_batch"):
            labelled_batch_IDs = self.labelled_IDs + self.batch_IDs
            labelled_batch_samples = self.task.get_samples_repr_1d(labelled_batch_IDs)
            unlabelled_samples = self.task.get_samples_repr_1d(self.unlabelled_IDs)

            (_, distances_min) = pairwise_distances_argmin_min(
                unlabelled_samples, labelled_batch_samples)
            self.min_distances_to_labelled_batch = distances_min

        return self.min_distances_to_labelled_batch

    def get_distances_percentiles_to_all(self, percentile_borders: List[float] = [0.05]):
        if not 'distances_percentiles_to_all' in self.batch_independent_features:
            unlabelled_samples = self.task.get_samples_repr_1d(self.unlabelled_IDs)

            no_unlabelled_samples_for_representativeness = min(500, len(self.unlabelled_IDs))
            all_IDs_subset = np.random.choice(
                self.unlabelled_IDs, size=no_unlabelled_samples_for_representativeness, replace=False)
            all_samples_subset = self.task.get_samples_repr_1d(all_IDs_subset)

            distances_to_Samples = self._get_similarity_percentiles(
                unlabelled_samples, all_samples_subset, percentile_borders)
            self.batch_independent_features['distances_percentiles_to_all'] = distances_to_Samples

        return self.batch_independent_features['distances_percentiles_to_all']

    def _get_similarity_percentiles(self, vectorA: np.ndarray, vectorB: np.ndarray,
                                    percentile_borders: List[float]) -> np.ndarray:
        '''
        @param vectorA: shape: (n , d)
        @param vectorB: shape: (m , d)
        @params percentile_borders: the percentiles taken (e.g. 0.1 for 10th percentile)
        @return: shape: (n, len(percentile_borders))
        '''
        if vectorA.shape[1] != vectorB.shape[1]:
            raise ValueError

        distances = pairwise_distances(vectorA, vectorB)  # shape (n, m)
        percentiles = np.percentile(distances, percentile_borders, axis=1).T  # shape (n, 6)
        return percentiles
