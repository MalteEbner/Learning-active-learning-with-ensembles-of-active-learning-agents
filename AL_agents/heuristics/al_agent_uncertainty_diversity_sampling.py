from typing import List

from AL_agents.al_agent import ALAgent
from AL_environment_MDP.al_mdp_observation import Observation


class ALAgentUncertaintyDiversitySampling(ALAgent):

    def get_utilities(self, observation: Observation) -> List[float]:
        uncertainties = observation.get_prediction_entropies()
        distances = observation.get_min_distances_to_labelled_and_batch()
        utilities = uncertainties * distances
        return list(utilities)
