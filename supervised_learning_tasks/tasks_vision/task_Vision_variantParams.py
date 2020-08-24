from supervised_learning_tasks.tasks_vision.task_Vision_1drepr_getter import Vision_1dRepr



class Task_Vision_variantParams:
    def __init__(self, dataset: str = "MNIST", regularizationFactor: float = 1e-4, initialLearningRate=3e-2, validationPatience = 3,
                 repr_1d_type: str='tSNE', repr_1d_n_components=200, no_epochs: int=-1):
        '''
            @param dataset: from { "MNIST" , "fashion"}
            @param architecture: from { "simpleCNN" , "resnet" }
            @param regularizationFactor: for l2-regularization of kernels
        '''
        self.no_epochs = no_epochs
        self.dataset = dataset
        self.regularizationFactor = regularizationFactor
        self.initLR = initialLearningRate
        self.valPat = validationPatience
        self.vision_1dRepr = Vision_1dRepr(dataset,_type=repr_1d_type,n_components=repr_1d_n_components)


    def __repr__(self):
        selfDict = self.__dict__.copy()
        selfDict['vision_1dRepr']=self.vision_1dRepr.type
        return str([selfDict[key] for key in sorted(selfDict.keys(), reverse=False)])

    def __shortRepr__(self):
        return f'{self.dataset}_repr1D_{self.vision_1dRepr.type}'


    def isEqual(self, other):
        isEqual = True
        isEqual = isEqual and self.dataset == other.dataset
        isEqual = isEqual and self.vision_1dRepr.type == other.vision_1dRepr.type
        return isEqual