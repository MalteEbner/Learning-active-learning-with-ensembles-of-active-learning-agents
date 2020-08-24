from typing import List
from math import ceil
import copy

import numpy as np

from supervised_learning_tasks.task_supervised import Task_supervised
from AL_environment_MDP.al_mdp_observation import Observation

class AL_Env():

    def __init__(self, al_parameters, task: Task_supervised):
        al_parameters.annotationBudget = min(al_parameters.annotationBudget, task.getNoTrainingSamples())
        self.task = task
        self.al_parameters = al_parameters

    def step(self, action: List[int], verbose=True) -> (Observation, float, bool, dict):

        try:
            subsetIDs = [self.unlabelledIDs[i] for i in action]
        except Exception as e:
            debug_point = 0

        # if batchSize is larger than remaining annotation budget, reduce samples to label
        remainingBudget = self.al_parameters.annotationBudget - (len(self.labelledIDs) + len(self.batch))
        if len(subsetIDs) > remainingBudget:
            subsetIDs = list(subsetIDs[:remainingBudget])

        # update sets: put subsetIDs from labelled set to batch
        self.batch += subsetIDs
        oldNoUnlabelledIDs = len(self.unlabelledIDs)
        self.unlabelledIDs = copy.copy(list(set(self.unlabelledIDs) - set(subsetIDs)))
        if len(self.unlabelledIDs) != oldNoUnlabelledIDs - len(subsetIDs):
            raise ValueError

        # perform update
        epochFinished = len(self.labelledIDs) + len(self.batch) >= self.al_parameters.annotationBudget
        if len(self.batch) == self.al_parameters.batchSize_annotation or epochFinished:
            # update sets: put IDs from batch to labelledIDs
            self.labelledIDs = copy.copy(self.labelledIDs + self.batch)
            self.batch = []

            # retrain supervised learning model
            loss, accuracy = self.task.trainOnBatch(self.labelledIDs)
            reward = self.oldInfo["loss"] - loss

            # calculate Info
            info = dict()
            info["loss"] = loss
            info["noLabelledSamples"] = len(self.labelledIDs)
            info["accuracy"] = accuracy

            observation = self.define_observation()
        else:
            observation = self.oldObservation
            observation.delete_features_of_chosen_samples(action)

            reward = 0
            info = self.oldInfo
            info["noLabelledSamples"] = len(self.labelledIDs)

        self.oldObservation = observation
        self.oldInfo = info

        return observation, reward, epochFinished, info

    def reset(self) -> dict:
        self.task.resetModel()

        # define labelled and unlabelled Set by their IDs
        self.labelledIDs = list(np.random.randint(0, self.task.getNoTrainingSamples(), self.al_parameters.startingSize))
        #print(f"labelled IDs: {self.labelledIDs}")
        allIDs = list(range(self.task.getNoTrainingSamples()))
        self.unlabelledIDs = list(set(allIDs) - set(self.labelledIDs))
        self.batch = []

        # define accuracy
        loss, accuracy = self.task.trainOnBatch(self.labelledIDs)


        # calculate Info
        info = dict()
        info["loss"] = loss
        info["noLabelledSamples"] = len(self.labelledIDs)
        info["accuracy"] = accuracy
        # self.initialInfo = info

        observation = self.define_observation()

        self.oldObservation = observation
        self.oldInfo = info

        return observation

    def render(self, mode='human'):
        raise NotImplementedError

    def define_observation(self):
        observation = Observation(self.task, self.labelledIDs, self.unlabelledIDs, self.batch)
        return observation

    def expectedNoIterations(self):
        expectedNoIterations = self.al_parameters.annotationBudget - self.al_parameters.startingSize
        return expectedNoIterations

