from typing import List

from AL_agents.al_agent import ALAgent
from AL_environment_MDP.al_mdp_observation import Observation


class ALAgentRepresentativeSampling(ALAgent):

    def get_utilities(self, observation: Observation) -> List[float]:
        utilities = -1 * observation.get_distances_percentiles_to_all([0.05])[:, 0]
        return list(utilities)
