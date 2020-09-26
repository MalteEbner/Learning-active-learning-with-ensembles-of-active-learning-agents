from typing import List

import numpy as np

from AL_agents.al_agent import ALAgent
from AL_environment_MDP.al_mdp_observation import Observation


class ALAgentRandomSampling(ALAgent):

    def get_utilities(self, observation: Observation) -> List[float]:
        utilities = np.random.normal(0, 1, len(observation.unlabelled_IDs))
        return list(utilities)
