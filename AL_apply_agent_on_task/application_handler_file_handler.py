from typing import List

import jsonpickle
import matplotlib.pyplot as plt
import numpy as np
import scipy



class ApplicationHandlerFileHandlerJSON:
    def __init__(self, filename="applicationHandlers.json"):
        self.filename = filename

    def readApplicationHandlerFromFile(self):
        # Read JSON data into the datastore variable
        with open(self.filename, 'r') as f:
            dataString = f.read()
            datastore = jsonpickle.decode(dataString)
        return datastore

    def writeApplicationHandlerToFile(self, applicationHandler: object):
        self.writeApplicationHandlersToFile([applicationHandler])



    def writeApplicationHandlersToFile(self, applicationHandlerList):

        try:
            # Read JSON data into the datastore variable
            datastore = self.readApplicationHandlerFromFile()
            datastore += applicationHandlerList
        except FileNotFoundError:
            datastore = applicationHandlerList

        # Writing JSON data
        with open(self.filename, 'w+') as f:
            f.write(jsonpickle.encode(datastore, ))

    def deleteSomeApplicationHandlers(self, filterFunction):
        '''

        :param filterFunction: if filterFunction(applicationHandler, index) returns True, applicationHandler is deleted
        :return: None
        '''

        applicationHandlers = self.readApplicationHandlerFromFile()
        applicationHandlers = [handler for index, handler in enumerate(applicationHandlers) if
                               not filterFunction(handler, index)]

        # Writing JSON data (and overwriting file)
        with open(self.filename, 'w') as f:
            dataString = jsonpickle.encode(applicationHandlers, f)
            f.write(dataString)

    def deleteAllObservations(self):
        applicationHandlers = self.readApplicationHandlerFromFile()
        for applicationHandler in applicationHandlers:
            if hasattr(applicationHandler,"observations"):
                del applicationHandler.observations

        # Writing JSON data (and overwriting file)
        with open(self.filename, 'w') as f:
            dataString = jsonpickle.encode(applicationHandlers, f)
            f.write(dataString)

    def deleteSpecificAgent(self, name='learningAgent'):
        applicationHandlers = self.readApplicationHandlerFromFile()
        applicationHandlers = [x for x in applicationHandlers if not name == x.al_agent_params.__shortRepr__()]

        # Writing JSON data (and overwriting file)
        with open(self.filename, 'w') as f:
            dataString = jsonpickle.encode(applicationHandlers, f)
            f.write(dataString)


    def plotAllContentWithConfidenceIntervals(self,metric = 'accuracy',withTitle: bool=True,agentNames: List[str]=[]):
        '''
        define plots and legends
        '''
        runRepresentations = []
        applicationHandlers = self.readApplicationHandlerFromFile()
        for applicationHandler in applicationHandlers:
            concattedInfos = applicationHandler.concatInfos()
            runRepresentations += [(applicationHandler.al_agent_params.__shortRepr__(),concattedInfos["noLabelledSamples"],concattedInfos[metric])]

        fullAgentNames = list(set(repr[0] for repr in runRepresentations))
        if len(agentNames) == 0:
            agentNames = fullAgentNames
        else:
            agentNames = list(set(agentNames) & set(fullAgentNames))
        agentNames.sort(key=lambda name: name)

        fig = plt.figure(figsize=(6, 4), dpi=320)
        legends = []

        def mean_confidence_std(dataMatrix, confidence: float=0.95):
            '''
            @param dataMatrix: shape: (noIterations, noRepetitions)
            @param confidence:
            @return: shapes: 5 times (noIterations,)
            '''
            means = np.mean(dataMatrix,axis=1)
            stds = np.std(dataMatrix,axis=1)
            noRepetitions = dataMatrix.shape[1]
            deviation = stds * scipy.stats.t.ppf((1+confidence)/2., noRepetitions-1)/(noRepetitions**0.5)
            return means, means-deviation,means+deviation, means-stds, means+stds


        colorCycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i, agentName in enumerate(agentNames):
            noLabelledSamplesList = [runRepr[1] for runRepr in runRepresentations if runRepr[0] == agentName]
            maxNoSamples = max(len(arr) for arr in noLabelledSamplesList)
            noLabelledSamples = next(x for x in noLabelledSamplesList if len(x)==maxNoSamples)
            accuracyTensor = np.stack([runRepr[2] for runRepr in runRepresentations if runRepr[0] == agentName and len(runRepr[1]) == maxNoSamples], axis=1)
            means, lowerBound, upperBound, lowerBound_std, upperBound_std = mean_confidence_std(accuracyTensor)
            # plotRepresentation = (agentName,noLabelledSamples,means,lowerBound,upperBound)

            plt.fill_between(noLabelledSamples,lowerBound,upperBound,color=colorCycle[i],alpha=.5)
            plt.fill_between(noLabelledSamples, lowerBound_std, upperBound_std, color=colorCycle[i], alpha=.1)
            plt.plot(noLabelledSamples,means,color=colorCycle[i])
            legends += [agentName]

        '''
        start plotting
        '''
        plt.legend(legends)

        title = "Task: " + applicationHandlers[-1].task_params.__shortRepr__()
        #title += "\nEnv: " + str(applicationHandlers[-1].al_Parameters)
        from textwrap import wrap
        title = "\n".join(wrap(title, 60))
        plt.xlabel('number of Samples')
        plt.ylabel(metric)
        if withTitle:
            plt.title(title, fontsize=10)
        plt.tight_layout()
        plt.grid()


        saveFigure = True
        if saveFigure:
            filename = self.filename
            filename = filename.replace(".json",".png")
            filename = filename.replace("\ ", " ")
            filename = filename.replace(":", "_")
            plt.savefig(filename, figsize=(6, 4), dpi=320)

        plt.show()