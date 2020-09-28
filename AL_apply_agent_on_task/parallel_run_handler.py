import time
import random
from typing import List

import ray
import lightgbm  # needed by hyperopt
import sklearn  # needed by hyperopt
import tqdm  # needed by hyperopt
import numpy.random

from AL_apply_agent_on_task.application_handler_file_handler import ApplicationHandlerFileHandlerJSON
from AL_apply_agent_on_task.application_handler import ApplicationHandler


class ParallelRunHandler:
    def __init__(self, experiment_filename: str, n_jobs: int, test: bool = False, save_results: bool = True,
                 parallelization=True, verbose=True, delete_old_ensemble_data: bool = False):
        self.experiment_filename = experiment_filename
        self.n_jobs = n_jobs
        self.test = test
        self.save_results = save_results
        self.verbose = verbose
        self.parallelization = parallelization
        self.delete_old_ensemble_data = delete_old_ensemble_data

    def __enter__(self):
        if self.parallelization:
            # perform actual experiments
            ray.init(num_cpus=self.n_jobs, temp_dir='/tmp/ray_LAL_ensemble', local_mode=True)
        self._init_file_handler(self.test, self.experiment_filename)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.parallelization:
            ray.shutdown()

    def _init_file_handler(self, test, experiment_filename):
        if self.save_results:
            # define applicationHandler to store the results
            if test:
                filename = "Experiments/results/applicationHandler_test.json"
                import os

                try:
                    os.remove(filename)
                except OSError as e:
                    pass
            else:
                filename = experiment_filename
                filename += '.json'
            file_handler = ApplicationHandlerFileHandlerJSON(filename)
            if self.delete_old_ensemble_data:
                def filter_function(application_handler, index):
                    return application_handler.al_agent_params.agent_name == "Ensemble"
                file_handler.delete_some_application_handlers(filter_function)
            self.fileHandler = file_handler
            self.filename = filename

    @ray.remote
    def run_agent_on_task(self, task_params, _al_params, agent_params, rng_seed):
        random.seed(rng_seed)
        numpy.random.seed(rng_seed)
        application_handler = ApplicationHandler(task_params, _al_params, agent_params, verbose=False)
        application_handler.run_episode()
        return application_handler

    def al_apply_agents_on_task(self, task_param_list, al_params, agent_params_list) -> List[ApplicationHandler]:
        if len(task_param_list) != len(agent_params_list):
            raise ValueError
        number_repetitions = len(task_param_list)

        start = time.time()
        if self.verbose:
            print(f'Starting experiments at time {time.time() - start}')
        if self.parallelization:
            # Reset RNGs
            numpy.random.seed()
            rng_seeds = list(numpy.random.randint(0, 9999, len(agent_params_list)))

            remaining_ray_ids = [self.run_agent_on_task.remote(self, task_param, al_params, agent_param, rng_seed)
                                 for task_param, agent_param, rng_seed
                                 in zip(task_param_list, agent_params_list, rng_seeds)]
            total_tasks = len(remaining_ray_ids)
            num_returns = min(12, int(number_repetitions / 4), len(remaining_ray_ids))
            num_returns = max(1, num_returns)
            finished_application_handlers = []

            while len(remaining_ray_ids) > 0:
                num_returns = min(num_returns, len(remaining_ray_ids))
                ready_ids, remaining_ray_ids = ray.wait(remaining_ray_ids, num_returns=num_returns)
                application_handlers = ray.get(ready_ids)
                finished_application_handlers += application_handlers
                if self.verbose:
                    print(f'finished {len(finished_application_handlers)} '
                          f'of {total_tasks} tasks at time {time.time() - start}')

                if self.save_results:
                    self.fileHandler.write_application_handlers_to_file(application_handlers)

        else:
            application_handlers = [ApplicationHandler(task_params, al_params, agent_params, verbose=False)
                                    for task_params, agent_params
                                    in zip(task_param_list, agent_params_list)]
            finished_application_handlers = []
            for i, applicationHandler in enumerate(application_handlers):
                applicationHandler.run_episode()
                finished_application_handlers += [applicationHandler]
                print(f'finished {len(finished_application_handlers)} '
                      f'of {len(agent_params_list)} tasks at time {time.time() - start}')
            if self.save_results:
                self.fileHandler.write_application_handlers_to_file(finished_application_handlers)

        if self.save_results:
            return finished_application_handlers, self.filename
        else:
            return finished_application_handlers
