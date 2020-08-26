import hyperopt as hp
import numpy as np


class BetaDictHandler:
    def __init__(self,taskName):

        if taskName == "model_checkerboard":
            self._get_beta_dict_checkerboard()
        else:
            raise ValueError


    def get_hyperopt_space(self, variance_factor = 1):
        space = dict()
        for key, value in self.beta_dict.items():
            desired_mu = value + 0.1  # for computational stability
            desired_sigma = (value + 0.1)*variance_factor # for computational stability
            mu = np.log(desired_mu**2/np.sqrt(desired_mu**2+desired_sigma**2))
            sigma = np.log(1+desired_sigma**2/(desired_mu**2))
            space[key] = hp.hp.lognormal(key,mu,sigma)
        return space


    def _get_beta_dict_checkerboard(self):
        self.beta_dict = dict()
        self.beta_dict["Uncertainty"] = 20
        self.beta_dict["Diversity"] = 20
        self.beta_dict["Representative"] = 0