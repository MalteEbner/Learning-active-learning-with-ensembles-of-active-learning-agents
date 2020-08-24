from supervised_learning_tasks.task_supervised import Task_supervised
from AL_environment_MDP.al_env import AL_Env

class AL_Parameters():
    def __init__(self, annotationBudget: int=200, startingSize = 38):
        '''
        @param batchSize_annotation: the no of samples needed in the batch till it is annotated and the model is updated
        @param annotationBudget: the maximum no of samples annotated till the active learning process is stopped
        @param startingSize: the no of samples annotated in the beginning (reducing the annotation budget)
        '''
        #self.batchSize_annotation = None #is set in application handler at it is agent-specific
        self.annotationBudget = annotationBudget
        self.startingSize = startingSize

    def createAL_env(self, task: Task_supervised) -> AL_Env:
        al_env = AL_Env(self, task)
        return al_env

    def _getRelevantAttributeDict(self,ignoreBatchSize:bool =False, ignoreTrueImprovements = False):
        attributes = self.__dict__.copy()
        attributes.pop("withEntropy", None)
        attributes.pop("withSimilarities",None)
        attributes.pop("withEstimatedImprovements",None)
        if True:
            attributes.pop("startingSize",None)
        if ignoreBatchSize:
            attributes.pop("batchSize",None)
        return attributes

    def __repr__(self):
        return str(list(sorted(self._getRelevantAttributeDict().items())))


    def __eq__(self, other,ignoreBatchSize:bool =False, ignoreTrueImprovements = True):
        isEqual = self._getRelevantAttributeDict(ignoreBatchSize,ignoreTrueImprovements) \
                  == other._getRelevantAttributeDict(ignoreBatchSize,ignoreTrueImprovements)
        return isEqual

