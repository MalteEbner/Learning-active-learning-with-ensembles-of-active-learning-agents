from typing import List

import numpy as np

from AL_agents.al_agent import AL_Agent
from AL_environment_MDP.al_mdp_observation import Observation


class AL_agent_random_sampling(AL_Agent):

    def get_utilities(self, observation: Observation) -> List[float]:
        utilities = np.random.normal(0, 1, len(observation.unlabelled_IDs))
        return list(utilities)
