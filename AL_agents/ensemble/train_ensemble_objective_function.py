from typing import List

from scipy.stats import gmean


from AL_apply_agent_on_task.application_handler import ApplicationHandler
from AL_apply_agent_on_task.AL_apply_agents_on_task import al_apply_agents_on_task

def train_ensemble_objective_function(
        beta_dict: dict, task_param_list, al_params, agent_param,
        n_jobs, mean_type="arithmetic") -> List[float]:

        agent_param.beta_dict = beta_dict
        agent_param_list = [agent_param] * len(task_param_list)

        application_handlers = al_apply_agents_on_task(
            task_param_list, al_params, agent_param_list,
            n_jobs,
            test=False,
            save_results=False,
            parallelization=True,
            verbose = False
        )

        performances = [application_handler.infos[-1]['accuracy']
                        for application_handler in application_handlers]

        if mean_type == "arithmetic":
            return sum(performances)/len(performances)
        elif mean_type == "geometric":
            return float(gmean(performances))
