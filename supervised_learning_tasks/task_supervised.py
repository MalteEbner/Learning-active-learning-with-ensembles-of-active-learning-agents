from typing import List, Tuple

import numpy as np
from sklearn.metrics import pairwise_distances

class Task_supervised():

    def __init__(self):
        pass

    def resetModel(self) -> None:
        raise NotImplementedError


    # get Information needed during initialization

    def getNoTrainingSamples(self) -> int:
        raise NotImplementedError


    def getPredictions(self,sampleIDs: List[int]):
        raise NotImplementedError

    def get_samples_repr_1d(self, sampleIDs: List[int] = 'all') -> dict:
        raise NotImplementedError

    # get distances to batch
    def getdistancesToLabelledAndBatch(self,unlabelledPredictions,labelledIDs: List[int], unlabelledIDs: List[int],
                             batchIDs: List[int]=[]) -> np.ndarray:

        samplesUnlabelled = self.getModelIndependentInfo(unlabelledIDs)["samples_repr_1d"]
        samplesLabelledBatch = self.getModelIndependentInfo(labelledIDs+batchIDs)["samples_repr_1d"]

        #calculate distances of embeddings to labelled and batchset (minimum, 1st quartil, median)
        distances_to_Labelled_Batch = self.getSimilarityPercentiles(samplesUnlabelled,samplesLabelledBatch, [0.0,0.1])
        distances_to_Labelled_Batch = distances_to_Labelled_Batch


        return distances_to_Labelled_Batch, samplesUnlabelled

    # get information about sample for observation space
    def getSampleInfo(self, al_parameters: object, labelledIDs: List[int], unlabelledIDs: List[int],
                             oldLoss: float, batchIDs: List[int]=[]) -> dict:

        samplesInfo = self.getModelDependentInfo(unlabelledIDs)



        #calculate prediction entropy
        samplesInfo["prediction_entropy"] = entropy(samplesInfo["predictions"])

        sims = self.getdistancesToLabelledAndBatch(samplesInfo["predictions"],labelledIDs,unlabelledIDs,batchIDs)
        samplesInfo["distances_to_Labelled_Batch"] = sims[0]
        #samplesInfo["pred_distances_to_Labelled_Batch"] = sims[1]
        samplesUnlabelled = sims[1]

        #calculate distances percentiles of embeddings to a subset of the unlabelled set
        noUnlabelledSamplesForComputingRepresentativeness = min(500,len(unlabelledIDs))
        subsetUnlabelledIDs = np.random.choice(unlabelledIDs,size=noUnlabelledSamplesForComputingRepresentativeness,replace=False)
        subsetUnlabelledSamples = self.getModelIndependentInfo(subsetUnlabelledIDs)["samples_repr_1d"]
        distances_to_Unlabelled = self.getSimilarityPercentiles(samplesUnlabelled,subsetUnlabelledSamples, [0.05,0.1,0.2])
        samplesInfo["distances_to_Unlabelled"] = distances_to_Unlabelled


        #for calculating additional features in data generator
        samplesInfo["labelledIDs"] = labelledIDs
        samplesInfo["unlabelledIDs"] = unlabelledIDs

        return samplesInfo




    # train model

    def trainFully(self) -> float:
        raise NotImplementedError

    def trainOnBatch(self, sampleIDs: List[int]) -> Tuple[float,float]:
        raise NotImplementedError



