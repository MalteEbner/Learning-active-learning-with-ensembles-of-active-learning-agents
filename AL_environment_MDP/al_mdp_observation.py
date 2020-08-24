from typing import List

import numpy as np
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min

from supervised_learning_tasks.task_supervised import Task_supervised


class Observation:
    def __init__(self, task: Task_supervised, labelled_IDs, unlabelled_IDs, batch_IDs):
        self.task = task
        self.labelled_IDs = labelled_IDs
        self.unlabelled_IDs = unlabelled_IDs
        self.batch_IDs = batch_IDs

        self.features = dict()

    def delete_features_of_chosen_samples(self,sample_IDs: List[int]):
        for (feature_name,feature_value) in self.features.items():
            feature_value_with_deleted_sample = np.delete(feature_value,sample_IDs,axis=0)
            self.features[feature_name] = feature_value_with_deleted_sample
        self.unlabelled_IDs = np.delete(self.unlabelled_IDs,sample_IDs,axis=0)

    def get_prediction_entropies(self):
        if not 'predictions' in self.features:
            self.features['predictions'] = self.task.getPredictions(self.unlabelled_IDs)
        if not 'prediction_entropies' in self.features:
            probs = self.features['predictions']
            zeros = np.zeros_like(probs)
            log_probs = np.log2(probs, out=zeros, where=probs > 0)
            entropies = -1 * np.sum(probs * log_probs, axis=1)
            self.features['prediction_entropies'] = entropies
        return self.features['prediction_entropies']

    def get_min_distances_to_labelled_and_batch(self):
        if not 'min_distances_to_labelled_batch' in self.features:
            labelled_batch_IDs = self.labelled_IDs + self.batch_IDs
            labelled_batch_samples = self.task.get_samples_repr_1d(labelled_batch_IDs)
            unlabelled_samples = self.task.get_samples_repr_1d(self.unlabelled_IDs)

            (_, distances_min) = pairwise_distances_argmin_min(
                unlabelled_samples, labelled_batch_samples)
            self.features['min_distances_to_labelled_batch'] = distances_min

        return self.features['min_distances_to_labelled_batch']

    def get_distances_percentiles_to_all(self, percentile_borders: List[float] = [0.05]):
        if not 'distances_percentiles_to_all' in self.features:
            unlabelled_samples = self.task.get_samples_repr_1d(self.unlabelled_IDs)

            no_unlabelled_samples_for_representativeness = min(500, len(self.unlabelled_IDs))
            all_IDs_subset = np.random.choice(
                self.unlabelled_IDs, size=no_unlabelled_samples_for_representativeness, replace=False)
            all_samples_subset = self.task.get_samples_repr_1d(all_IDs_subset)

            distances_to_Samples = self._get_similarity_percentiles(
                unlabelled_samples, all_samples_subset, percentile_borders)
            self.features['distances_percentiles_to_all'] = distances_to_Samples

        return self.features['distances_percentiles_to_all']

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
