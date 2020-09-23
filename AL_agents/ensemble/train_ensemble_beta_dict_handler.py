import hyperopt as hp
import numpy as np


class BetaDictHandler:
    def __init__(self, task_name: str = None):

        if task_name is not None:
            if task_name == "model_checkerboard":
                self._define_beta_dict(1.04, 0.88, 0.8)
            elif task_name == "model_UCI":
                self._define_beta_dict(3.97, 0.81, 0.62)
            elif task_name == "model_Vision":
                self._define_beta_dict(1, 1, 1)
            else:
                raise ValueError
        else:
            self._define_beta_dict(1, 1, 1)

    def get_hyperopt_space(self, standard_deviation_factor=4):
        space = dict()
        for key, value in self.beta_dict.items():
            desired_mu = value + 0.001  # for computational stability
            desired_sigma = value * standard_deviation_factor + 0.001
            mu = np.log(desired_mu ** 2 / np.sqrt(desired_mu ** 2 + desired_sigma ** 2))
            sigma = np.log(1 + desired_sigma ** 2 / (desired_mu ** 2))
            space[key] = hp.hp.lognormal(key, mu, sigma)
        return space

    def _define_beta_dict(self, beta_uncertainty, beta_diversity, beta_representative):
        self.beta_dict = dict()
        self.beta_dict["Uncertainty"] = beta_uncertainty
        self.beta_dict["Diversity"] = beta_diversity
        self.beta_dict["Representative"] = beta_representative


