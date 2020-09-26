from typing import List
from math import ceil
import copy

import numpy as np

from supervised_learning_tasks.task_supervised import TaskSupervised
from AL_environment_MDP.al_mdp_observation import Observation


class ALEnvironment():

    def __init__(self, al_parameters, task: TaskSupervised):
        al_parameters.annotationBudget = min(al_parameters.annotationBudget, task.get_no_training_samples())
        self.task = task
        self.al_parameters = al_parameters

    def step(self, action: List[int], verbose=True) -> (Observation, float, bool, dict):

        subset_IDs = [self.unlabelled_IDs[i] for i in action]

        # if batchSize is larger than remaining annotation budget, reduce samples to label
        remaining_budget = self.al_parameters.annotationBudget - (len(self.labelled_IDs) + len(self.batch))
        if len(subset_IDs) > remaining_budget:
            subset_IDs = list(subset_IDs[:remaining_budget])

        # update sets: put subset_IDs from labelled set to batch
        self.batch += subset_IDs
        old_no_unlabelled_IDs = len(self.unlabelled_IDs)
        self.unlabelled_IDs = copy.copy(list(set(self.unlabelled_IDs) - set(subset_IDs)))
        if len(self.unlabelled_IDs) != old_no_unlabelled_IDs - len(subset_IDs):
            raise ValueError

        # perform update
        epoch_finished = len(self.labelled_IDs) + len(self.batch) >= self.al_parameters.annotationBudget
        if len(self.batch) == self.al_parameters.batch_size_annotation or epoch_finished:
            # update sets: put IDs from batch to labelled_IDs
            self.labelled_IDs = copy.copy(self.labelled_IDs + self.batch)
            self.batch = []

            # retrain supervised learning model
            loss, accuracy = self.task.train_on_batch(self.labelled_IDs)
            reward = self.oldInfo["loss"] - loss

            # calculate Info
            info = dict()
            info["loss"] = loss
            info["no_labelled_samples"] = len(self.labelled_IDs)
            info["accuracy"] = accuracy

            observation = self.define_observation()
        else:
            observation = self.oldObservation
            observation.update_features_based_on_action(action)

            reward = 0
            info = self.oldInfo
            info["no_labelled_samples"] = len(self.labelled_IDs)

        self.oldObservation = observation
        self.oldInfo = info

        return observation, reward, epoch_finished, info

    def reset(self) -> Observation:

        # define labelled and unlabelled Set by their IDs
        self.labelled_IDs = list(np.random.randint(0, self.task.get_no_training_samples(), self.al_parameters.startingSize))
        # print(f"labelled IDs: {self.labelled_IDs}")
        allIDs = list(range(self.task.get_no_training_samples()))
        self.unlabelled_IDs = list(set(allIDs) - set(self.labelled_IDs))
        self.batch = []

        # define accuracy
        loss, accuracy = self.task.train_on_batch(self.labelled_IDs)

        # calculate Info
        info = dict()
        info["loss"] = loss
        info["no_labelled_samples"] = len(self.labelled_IDs)
        info["accuracy"] = accuracy
        # self.initialInfo = info

        observation = self.define_observation()

        self.oldObservation = observation
        self.oldInfo = info

        return observation

    def render(self, mode='human'):
        raise NotImplementedError

    def define_observation(self) -> Observation:
        observation = Observation(self.task, self.labelled_IDs, self.unlabelled_IDs, self.batch)
        return observation

    def expected_number_iterations(self) -> int:
        expectedNoIterations = self.al_parameters.annotationBudget - self.al_parameters.startingSize
        return expectedNoIterations
