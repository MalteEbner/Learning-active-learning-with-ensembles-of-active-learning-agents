## How to install:
- ensure you have python 3.6, 3.7 or 3.8
- install all packages defined in the requirements.txt file
- ensure all pytest in the 'pytests' directory run without errors

- Alternatively: follow the instructions from the github action: .github/workflows/python-package.yml

## How to train the ensemble:
- run the script 'AL_agents/ensemble/train_ensemble.py'
- save the learned beta-parameters in the file 'AL_agents/ensemble/train_ensemble_beta_dict_handler.py' in the __init__ function

## How to evaluate the agents:
- Optionally: adapt the parameters in 'AL_apply_agent_on_task/application_config.py'
- run the script 'AL_apply_agent_on_task/AL_run_experiment_on_task.py'

## How to view the plots of the performance over the number of samples:
- Look in the 'Experiments' folder

## How to see the visualisation of the objective function and its raw data:
- Look in the 'AL_agents/visualize_objective_function/evaluations' folder
