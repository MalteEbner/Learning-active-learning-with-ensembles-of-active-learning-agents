
from supervised_learning_tasks.task_supervised import Task_supervised
from supervised_learning_tasks.tasks_checkerboard.task_checkerboard_randomForest import Task_Checkerboard_randomForest
from supervised_learning_tasks.tasks_QA_bAbI.task_bAbI_memoryNetwork import Task_bAbI_memoryNetwork
from supervised_learning_tasks.tasks_UCI.task_UCI_randomForest import Task_UCI_randomForest
from supervised_learning_tasks.tasks_vision.task_Vision_CNN import Task_Vision_CNN


class Task_Parameters():
    def __init__(self,taskName: str, variantParams: dict=None):
        self.taskName = taskName
        self.variantParams = variantParams

    def createTask(self,verboseInit=False) -> Task_supervised:
        variantParams = self.variantParams

        if self.taskName == "model_bAbI":
            task = Task_bAbI_memoryNetwork(variantParams=variantParams,verboseInit=verboseInit)

        elif self.taskName == "model_Vision":
            task = Task_Vision_CNN(variantParams=variantParams,verboseInit=verboseInit)

        elif self.taskName == "model_checkerboard":
            task = Task_Checkerboard_randomForest(variantParams=variantParams,verboseInit=verboseInit)

        elif self.taskName == "model_UCI":
            task = Task_UCI_randomForest(variantParams=variantParams,verboseInit=verboseInit)

        else:
            print("ERROR: taskName unknown")
            raise ValueError
        return task

    def getExperimentFilename(self):
        filename = 'Experiments/results/'
        filename += self.__repr__()
        filename += '_experiments'
        return filename

    def __repr__(self):
        selfDict = self.__dict__
        return str([selfDict[key] for key in sorted(selfDict.keys(), reverse=False)])

    def __shortRepr__(self):
        repr = self.taskName
        if self.variantParams != None:
            repr += '_'
            if hasattr(self.variantParams,'__shortRepr__'):
                repr = self.variantParams.__shortRepr__()
            else:
                repr += str(self.variantParams)
        repr = repr.replace('model_','')
        repr = repr.replace('_randomForest','')
        return repr

    def __eq__(self, other):
        equal = True
        equal = equal and self.taskName == other.taskName
        if self.variantParams != None:
            if hasattr(self.variantParams,'isEqual'):
                equal = equal and self.variantParams.isEqual(other.variantParams)
            else:
                equal = equal and self.variantParams == other.variantParams
        return equal

