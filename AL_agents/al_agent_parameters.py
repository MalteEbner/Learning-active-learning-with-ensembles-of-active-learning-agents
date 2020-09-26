from typing import Dict

from AL_agents.heuristics.al_agent_uncertainty_diversity_sampling import ALAgentUncertaintyDiversitySampling
from AL_agents.heuristics.al_agent_uncertainty_sampling import ALAgentUncertaintySampling
from AL_agents.heuristics.al_agent_random_sampling import ALAgentRandomSampling
from AL_agents.heuristics.al_agent_diversity_sampling import ALAgentDiversitySampling
from AL_agents.heuristics.al_agent_representative_sampling import ALAgentRepresentativeSampling
from AL_agents.ensemble.al_agent_ensemble import ALAgentEnsemble


class ALAgentParameters():
    def __init__(self, agent_name: str = 'Random', batch_size_annotation: int = 1,
                 batch_size_agent: int = -1, beta_dict: Dict = None):
        self.agent_name = agent_name
        self.batch_size_annotation = batch_size_annotation
        if batch_size_agent <= 0:
            batch_size_agent = batch_size_annotation
        self.batch_size_agent = batch_size_agent
        if beta_dict is not None:
            self.beta_dict = beta_dict

    def create_agent(self):
        if self.agent_name == 'Random':
            agent = ALAgentRandomSampling(self)
        elif self.agent_name == 'Uncertainty':
            agent = ALAgentUncertaintySampling(self)
        elif self.agent_name == 'Diversity':
            self.batch_size_agent = 1
            agent = ALAgentDiversitySampling(self)
        elif self.agent_name == 'Uncertainty_Diversity':
            self.batch_size_agent = 1
            agent = ALAgentUncertaintyDiversitySampling(self)
        elif self.agent_name == 'Representative':
            agent = ALAgentRepresentativeSampling(self)
        elif self.agent_name == 'Ensemble':
            self.batch_size_agent = 1
            agent = ALAgentEnsemble(self)
        else:
            print(f'ERROR: agentName unknown: {self.agent_name}')
            raise ValueError

        return agent

    def _get_relevant_attribute_dict(self):
        attributes = self.__dict__.copy()
        return attributes

    def __repr__(self):
        return str(self._get_relevant_attribute_dict())

    def __short_repr__(self):
        agent_name = str(self.agent_name)
        if agent_name not in ['Random', 'Diversity', 'Representative']:
            if self.batch_size_annotation > 1:
                agent_name += '_' + str(self.batch_size_annotation)
            else:
                agent_name += '_sequential'
        return agent_name
