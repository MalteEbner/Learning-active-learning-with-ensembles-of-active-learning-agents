from typing import List

from AL_agents.al_agent import ALAgent
from AL_environment_MDP.al_mdp_observation import Observation


class ALAgentUncertaintySampling(ALAgent):

    def get_utilities(self, observation: Observation) -> List[float]:
        utilities = observation.get_prediction_entropies()
        return list(utilities)
