from os import listdir
from os.path import isfile, join

from AL_apply_agent_on_task.application_handler_file_handler import ApplicationHandlerFileHandlerJSON

path = "./Experiments/results"

files = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and f.endswith('.json')]
for file in files:
    file_handler = ApplicationHandlerFileHandlerJSON(file)
    file_handler.plot_all_content_with_confidence_intervals()