import time

from supervised_learning_tasks.task_parameters import Task_Parameters
from AL_environment_MDP.al_parameters import AL_Parameters
from AL_agents.al_agent_parameters import AL_Agent_Parameters

from AL_apply_agent_on_task.application_handler_file_handler import ApplicationHandlerFileHandlerJSON


class ApplicationHandler(object):
    def __init__(self, task_params: Task_Parameters, al_params: AL_Parameters,
                 al_agent_params: AL_Agent_Parameters, verbose: bool = True):
        self.al_Parameters = al_params
        self.task_params = task_params
        self.al_agent_params = al_agent_params
        self.verbose = verbose

        self.al_Parameters.batchSize_annotation = self.al_agent_params.batchSize_annotation

    def run_episode(self, saveFullData: bool = False):
        task = self.task_params.createTask()
        al_env = self.al_Parameters.createAL_env(task)
        al_agent = self.al_agent_params.createAgent()

        if self.verbose:
            print("Start running, expected %d iterations" % al_env.expectedNoIterations())
            self.startTime = time.time()

        # start running
        self.iteration = 0
        self.infos = []
        if saveFullData:
            self.observations = []

        observation = al_env.reset()
        info = al_env.oldInfo
        if saveFullData:
            self.loggingStep(info, observation)
        else:
            self.loggingStep(info)

        if al_env.expectedNoIterations() > 0:
            while True:
                # sample action
                action = al_agent.policy(observation)

                # apply step
                observation, reward, done, info = al_env.step(action)

                # log
                if saveFullData:
                    self.loggingStep(info, observation)
                else:
                    self.loggingStep(info)

                # end AL if epoch has ended
                if done:
                    break

    def loggingStep(self, info, observation=None):
        self.infos += [info]
        if observation != None:
            self.observations += [observation]
        if self.verbose:
            timediff = time.time() - self.startTime
            print('\n\niteration ' + str(self.iteration) + " (needed time: " + str(int(timediff)) + "s):  " + str(info))
        self.iteration += 1

    def concatInfos(self) -> dict:
        concattedInfos = dict()
        for key in self.infos[0].keys():
            concattedInfos[key] = [info[key] for info in self.infos]
        return concattedInfos

    def saveInFile(self, filename=None):
        if filename == None:
            fileHandler = ApplicationHandlerFileHandlerJSON()
        else:
            fileHandler = ApplicationHandlerFileHandlerJSON(filename)
        fileHandler.writeApplicationHandlerToFile(self)

    def isEqual(self, applicationHandler):
        equal = self.al_Parameters.__eq__(applicationHandler.al_Parameters) and \
                applicationHandler.task_params == self.task_params
        return equal
