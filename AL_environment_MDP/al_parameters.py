from AL_environment_MDP.al_environment import ALEnvironment
from supervised_learning_tasks.task_supervised import TaskSupervised


class ALParameters():
    def __init__(self, annotation_budget: int = 200, starting_size=38):
        '''
        @param annotation_budget: the maximum no of samples annotated till the active learning process is stopped
        @param starting_size: the no of samples annotated in the beginning (reducing the annotation budget)
        '''
        self.annotationBudget = annotation_budget
        self.startingSize = starting_size

    def create_al_environment(self, task: TaskSupervised) -> ALEnvironment:
        al_env = ALEnvironment(self, task)
        return al_env

    def _get_relevant_attribute_dict(self, ignoreBatchSize: bool = False):
        attributes = self.__dict__.copy()
        attributes.pop("withEntropy", None)
        attributes.pop("withSimilarities", None)
        attributes.pop("withEstimatedImprovements", None)
        if True:
            attributes.pop("starting_size", None)
        if ignoreBatchSize:
            attributes.pop("batchSize", None)
        return attributes

    def __repr__(self):
        return str(list(sorted(self._get_relevant_attribute_dict().items())))

    def __eq__(self, other, ignore_batch_size: bool = False):
        isEqual = self._get_relevant_attribute_dict(ignore_batch_size) == \
                  other._get_relevant_attribute_dict(ignore_batch_size)
        return isEqual

    def __ne__(self, other):
        return not self == other
