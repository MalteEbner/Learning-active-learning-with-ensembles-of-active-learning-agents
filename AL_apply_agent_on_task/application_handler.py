import time

from supervised_learning_tasks.task_parameters import TaskParameters
from AL_environment_MDP.al_parameters import ALParameters
from AL_agents.al_agent_parameters import ALAgentParameters


class ApplicationHandler(object):
    def __init__(self, task_params: TaskParameters, al_params: ALParameters,
                 al_agent_params: ALAgentParameters, verbose: bool = True):
        self.al_Parameters = al_params
        self.task_params = task_params
        self.al_agent_params = al_agent_params
        self.verbose = verbose

        self.al_Parameters.batch_size_annotation = self.al_agent_params.batch_size_annotation

    def run_episode(self):
        task = self.task_params.create_task()
        al_env = self.al_Parameters.create_al_environment(task)
        al_agent = self.al_agent_params.create_agent()

        if self.verbose:
            print("Start running, expected %d iterations" % al_env.expected_number_iterations())
            self.startTime = time.time()

        # start running
        self.iteration = 0
        self.infos = []

        observation = al_env.reset()
        info = al_env.oldInfo
        self.logging_step(info)

        if al_env.expected_number_iterations() > 0:
            while True:
                # sample action
                action = al_agent.policy(observation)

                # apply step
                observation, reward, done, info = al_env.step(action)

                # log
                self.logging_step(info)

                # end AL if epoch has ended
                if done:
                    break

    def logging_step(self, info):
        self.infos += [info]
        if self.verbose:
            time_diff = time.time() - self.startTime
            print('\n\niteration ' + str(self.iteration) + " (needed time: " + str(int(time_diff)) + "s):  " + str(info))
        self.iteration += 1

    def concat_infos(self) -> dict:
        concatted_infos = dict()
        for key in self.infos[0].keys():
            concatted_infos[key] = [info[key] for info in self.infos]
        return concatted_infos

    def save_in_file(self, filename=None):
        # import here to prevent cyclic imports
        from AL_apply_agent_on_task.application_handler_file_handler import ApplicationHandlerFileHandlerJSON
        if filename is None:
            file_handler = ApplicationHandlerFileHandlerJSON()
        else:
            file_handler = ApplicationHandlerFileHandlerJSON(filename)
        file_handler.write_application_handlers_to_file([self])

    def is_equal(self, application_handler):
        equal = self.al_Parameters.__eq__(application_handler.al_Parameters) and \
                application_handler.task_params == self.task_params
        return equal
