from typing import List, Tuple, Callable
from textwrap import wrap
from pathlib import Path
import os

import jsonpickle
import matplotlib.pyplot as plt
import numpy as np
import scipy

from AL_apply_agent_on_task.application_handler import ApplicationHandler


class ApplicationHandlerFileHandlerJSON:
    def __init__(self, filename="applicationHandlers.json"):
        self.filename = filename

    def read_application_handlers_from_file(self) -> List[ApplicationHandler]:
        # Read JSON data into the datastore variable
        with open(self.filename, 'r') as f:
            dataString = f.read()
            datastore = jsonpickle.decode(dataString)
        return datastore

    def write_application_handlers_to_file(self, application_handler_list: List[ApplicationHandler]):

        try:
            # Read JSON data into the datastore variable
            datastore = self.read_application_handlers_from_file()
            datastore += application_handler_list
        except FileNotFoundError:
            datastore = application_handler_list

        # Writing JSON data
        dirname = os.path.dirname(os.path.abspath(self.filename))
        Path(dirname).mkdir(parents=True, exist_ok=True)
        with open(self.filename, 'w+') as f:
            f.write(jsonpickle.encode(datastore, ))

    def delete_some_application_handlers(self, filter_function: Callable[[ApplicationHandler, int], bool]):
        '''
        @param filter_function: if filterFunction(applicationHandler, index) returns True, applicationHandler is deleted
        @return: None
        '''

        application_handlers = self.read_application_handlers_from_file()
        application_handlers = [handler for index, handler in enumerate(application_handlers) if
                                not filter_function(handler, index)]

        # Writing JSON data (and overwriting file)
        with open(self.filename, 'w') as f:
            data_string = jsonpickle.encode(application_handlers, f)
            f.write(data_string)

    def delete_specific_agent(self, name: str = 'ensemble'):
        application_handlers = self.read_application_handlers_from_file()
        application_handlers = [x for x in application_handlers if not name == x.al_agent_params.__short_repr__()]

        # Writing JSON data (and overwriting file)
        with open(self.filename, 'w') as f:
            data_string = jsonpickle.encode(application_handlers, f)
            f.write(data_string)

    def plot_all_content_with_confidence_intervals(self, metric='accuracy', with_title: bool = True,
                                                   agent_names: List[str] = [], plot_really: bool = True,
                                                   filename_for_plot=None):
        # define plots and legends
        run_representations = []
        application_handlers = self.read_application_handlers_from_file()
        for application_handler in application_handlers:
            concatted_infos = application_handler.concat_infos()
            run_representations += [(application_handler.al_agent_params.__short_repr__(),
                                     concatted_infos["no_labelled_samples"], concatted_infos[metric])]

        full_agent_names = list(set(representation[0] for representation in run_representations))
        if len(agent_names) == 0:
            agent_names = full_agent_names
        else:
            agent_names = list(set(agent_names) & set(full_agent_names))
        agent_names.sort(key=lambda name: name)

        fig = plt.figure(figsize=(6, 4), dpi=320)
        legends = []

        def mean_confidence_std(data_matrix, confidence: float = 0.95) -> Tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            '''
            @param data_matrix: shape: (noIterations, no_repetitions)
            @param confidence:
            @return: shapes: 5 times (noIterations,)
            '''
            means = np.mean(data_matrix, axis=1)
            stds = np.std(data_matrix, axis=1)
            no_repetitions = data_matrix.shape[1]
            deviation = stds * scipy.stats.t.ppf((1 + confidence) / 2., no_repetitions - 1) / (no_repetitions ** 0.5)
            return means, means - deviation, means + deviation, means - stds, means + stds

        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i, agent_name in enumerate(agent_names):
            accuracies_list_dict = dict()
            for run_repr in [x for x in run_representations if x[0] == agent_name]:
                already_entered_no_labelled_samples = []
                for accuracy, no_labelled_samples in zip(run_repr[2], run_repr[1]):
                    if no_labelled_samples in already_entered_no_labelled_samples:
                        break
                    else:
                        if no_labelled_samples not in accuracies_list_dict.keys():
                            accuracies_list_dict[no_labelled_samples] = list()
                        accuracies_list_dict[no_labelled_samples].append(accuracy)
            bounds_list = list()
            for no_labelled_samples, accuracies_list in sorted(accuracies_list_dict.items()):
                accuracy_tensor = np.array(accuracies_list)[np.newaxis, :]
                bounds = mean_confidence_std(accuracy_tensor)
                bounds_list.append(bounds)
            bounds_tuple_array = [np.vstack(x)[:, 0] for x in zip(*bounds_list)]
            means, lower_bound, upper_bound, lower_bound_std, upper_bound_std = bounds_tuple_array

            no_labelled_samples_list = sorted(accuracies_list_dict.keys())
            plt.fill_between(no_labelled_samples_list, lower_bound, upper_bound, color=color_cycle[i], alpha=.5)
            plt.fill_between(no_labelled_samples_list, lower_bound_std, upper_bound_std, color=color_cycle[i], alpha=.1)
            plt.plot(no_labelled_samples_list, means, color=color_cycle[i])
            legends += [agent_name]

        '''
        start plotting
        '''
        plt.legend(legends)

        title = "Task: " + os.path.basename(self.filename).replace(" experiments.json", "")
        # title += "\nEnv: " + str(application_handlers[-1].al_Parameters)
        title = "\n".join(wrap(title, 60))
        plt.xlabel('number of Samples')
        plt.ylabel(metric)
        if "fashion" in title and "MNIST" not in title:
            title = title.replace("fashion", "fashion-MNIST")
        if with_title:
            plt.title(title, fontsize=10)
        plt.tight_layout()
        plt.grid()

        save_figure = True
        if save_figure:
            if filename_for_plot is None:
                filename_for_plot = self.filename
                filename_for_plot = filename_for_plot.replace(".json", ".png")
                filename_for_plot = filename_for_plot.replace("\ ", " ")
                filename_for_plot = filename_for_plot.replace(":", "_")
            plt.savefig(filename_for_plot, dpi=320)

        if plot_really:
            plt.show()
