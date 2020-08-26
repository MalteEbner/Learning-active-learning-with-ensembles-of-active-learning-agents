import time
import random
from typing import List

import ray
import numpy.random

from AL_apply_agent_on_task.application_handler_file_handler import ApplicationHandlerFileHandlerJSON
from AL_apply_agent_on_task.application_handler import ApplicationHandler


def al_apply_agents_on_task(task_param_list, al_params, agent_params_list,
                            n_jobs: int,
                            test: bool = False, save_results: bool = True,
                            parallelization=True, verbose=True
                            ) -> List[ApplicationHandler]:
    if len(task_param_list) != len(agent_params_list):
        raise ValueError
    number_repetitions = len(task_param_list)

    if save_results:
        # define applicationHandler to store the results
        if test:
            filename = "Experiments/results/applicationHandler_test.json"
            import os

            try:
                os.remove(filename)
            except OSError as e:
                pass
        else:
            filename = task_param_list[0].getExperimentFilename()
            filename += '.json'
        fileHandler = ApplicationHandlerFileHandlerJSON(filename)

    start = time.time()
    if verbose:
        print(f'Starting experiments at time {time.time() - start}')
    if parallelization:
        # Reset RNGs
        numpy.random.seed()
        rng_seeds = list(numpy.random.randint(0, 9999, len(agent_params_list)))

        # perform actual experiments
        ray.init(num_cpus=n_jobs, temp_dir='/tmp/ray_LAL_ensemble')

        @ray.remote
        def run_agent_on_task(task_params, al_params, agent_params, rng_seed):
            random.seed(rng_seed)
            numpy.random.seed(rng_seed)
            applicationHandler = ApplicationHandler(task_params, al_params, agent_params, verbose=False)
            applicationHandler.run_episode(saveFullData=False)
            return applicationHandler

        n_jobs = min(n_jobs, len(agent_params_list))
        remainingRayIDs = [run_agent_on_task.remote(task_param, al_params, agent_param, rng_seed)
                           for task_param, agent_param, rng_seed
                           in zip(task_param_list, agent_params_list, rng_seeds)]
        total_tasks = len(remainingRayIDs)
        num_returns = min(12, int(number_repetitions / 4), len(remainingRayIDs))
        num_returns = max(1, num_returns)
        finished_application_handlers = []

        while len(remainingRayIDs) > 0:
            num_returns = min(num_returns, len(remainingRayIDs))
            ready_ids, remainingRayIDs = ray.wait(remainingRayIDs, num_returns=num_returns)
            applicationHandlers = ray.get(ready_ids)
            finished_application_handlers += applicationHandlers
            if verbose:
                print(f'finished {len(finished_application_handlers)} '
                      f'of {total_tasks} tasks at time {time.time() - start}')

            if save_results:
                fileHandler.writeApplicationHandlersToFile(applicationHandlers)

        ray.shutdown()
    else:
        applicationHandlers = [ApplicationHandler(task_params, al_params, agent_params, verbose=False)
                               for task_params, agent_params
                               in zip(task_param_list, agent_params_list)]
        finished_application_handlers = []
        for i, applicationHandler in enumerate(applicationHandlers):
            applicationHandler.run_episode(saveFullData=False)
            finished_application_handlers += [applicationHandler]
            print(f'finished {len(finished_application_handlers)} '
                  f'of {len(agent_params_list)} tasks at time {time.time() - start}')
        if save_results:
            fileHandler.writeApplicationHandlersToFile(finished_application_handlers)

    if save_results:
        return finished_application_handlers, filename
    else:
        return finished_application_handlers
