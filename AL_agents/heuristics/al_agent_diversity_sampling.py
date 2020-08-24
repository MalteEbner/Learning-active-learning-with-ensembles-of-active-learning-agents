from typing import List

import numpy as np

from AL_agents.al_agent import AL_Agent
from AL_environment_MDP.al_mdp_observation import Observation


class AL_agent_diversity_sampling(AL_Agent):

    def get_utilities(self, observation: Observation) -> List[float]:
        utilities = observation.get_min_distances_to_labelled_and_batch()
        return list(utilities)
