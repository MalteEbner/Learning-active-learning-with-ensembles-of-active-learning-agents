from typing import List
from AL_environment_MDP.al_mdp_observation import Observation
import numpy as np

class AL_Agent():
    def __init__(self, al_agent_parameters):
        self.al_agent_parameters = al_agent_parameters

    def policy(self, observation: Observation):
        utilities = self.get_utilities(observation)
        if len(utilities) != len(observation.unlabelled_IDs):
            raise ValueError

        batch_size = self.al_agent_parameters.batchSize_agent
        samples_with_maximum_utility = np.argpartition(utilities, -batch_size)[-batch_size:]

        action = samples_with_maximum_utility
        return list(action)

    def get_utilities(self, observation: Observation) -> List[float]:
        raise NotImplementedError

