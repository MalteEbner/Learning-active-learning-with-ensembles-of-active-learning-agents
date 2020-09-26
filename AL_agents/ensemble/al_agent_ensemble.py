from typing import List

import numpy as np

from AL_agents.al_agent import ALAgent
from AL_environment_MDP.al_mdp_observation import Observation


class ALAgentEnsemble(ALAgent):
    def __init__(self, al_agent_parameters):
        self.al_agent_parameters = al_agent_parameters

        if hasattr(al_agent_parameters, "beta_dict"):
            self.beta_dict = self.al_agent_parameters.beta_dict
        else:
            self.define_beta_dict()

        self.define_agents_dict()

    def get_utilities(self, observation: Observation) -> List[float]:
        utilities_total = np.zeros((len(observation.unlabelled_IDs),))
        for agentName in self.beta_dict.keys():
            # get utilities according to agent
            utilities = self.agents_dict[agentName].get_utilities(observation)
            utilities = np.array(utilities)
            # normalize them
            utilities = (utilities - np.mean(utilities))
            std_ = np.std(utilities)
            if std_ > 0:
                utilities /= std_
            # perform the linearcombination
            utilities *= self.beta_dict[agentName]
            utilities_total += utilities
        # add random agent
        utilities_total += self.agent_random.get_utilities(observation)
        return list(utilities_total)

    def define_beta_dict(self):
        self.beta_dict = dict()
        self.beta_dict["Uncertainty"] = 1
        self.beta_dict["Diversity"] = 1
        self.beta_dict["Uncertainty_Diversity"] = 1
        self.beta_dict["Representative"] = 1

    def define_agents_dict(self):
        # must import here to prevent cyclic imports
        from AL_agents.al_agent_parameters import ALAgentParameters

        self.agents_dict = dict()
        for agentName in self.beta_dict.keys():
            agent_params = ALAgentParameters(
                agent_name=agentName,
                batch_size_annotation=self.al_agent_parameters.batch_size_annotation,
                batch_size_agent=self.al_agent_parameters.batch_size_agent
            )
            agent = agent_params.create_agent()
            self.agents_dict[agentName] = agent

        agent_random_params = ALAgentParameters("Random",
                                                self.al_agent_parameters.batch_size_annotation,
                                                self.al_agent_parameters.batch_size_agent)
        self.agent_random = agent_random_params.create_agent()
