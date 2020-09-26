from typing import List

from AL_agents.al_agent import ALAgent
from AL_environment_MDP.al_mdp_observation import Observation


class ALAgentDiversitySampling(ALAgent):

    def get_utilities(self, observation: Observation) -> List[float]:
        utilities = observation.get_min_distances_to_labelled_and_batch()
        return list(utilities)
