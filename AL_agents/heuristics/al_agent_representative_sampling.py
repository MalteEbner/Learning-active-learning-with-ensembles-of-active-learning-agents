from typing import List

import numpy as np

from AL_agents.al_agent import AL_Agent
from AL_environment_MDP.al_mdp_observation import Observation


class AL_agent_representative_sampling(AL_Agent):

    def get_utilities(self, observation: Observation) -> List[float]:
        utilities = -1 * observation.get_distances_percentiles_to_all([0.05])[:, 0]
        return list(utilities)
