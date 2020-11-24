from os import listdir
from os.path import isfile, join

from AL_apply_agent_on_task.application_handler_file_handler import ApplicationHandlerFileHandlerJSON

path = "./Experiments/results_ICLR_other_batchsizes"

files = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and f.endswith('.json') and 'UCI' in f]
for file in files:
    file_handler = ApplicationHandlerFileHandlerJSON(file)
    if True:
        agent_names = ["Ensemble_2","Ensemble_8","Ensemble_24","Uncertainty_2", "Uncertainty_8", "Uncertainty_24", "Random"]
    else:
        agent_names = []
    file_handler.plot_all_content_with_confidence_intervals(agent_names=agent_names)
