from AL_agents.heuristics.al_agent_uncertainty_sampling import AL_agent_uncertainty_sampling
from AL_agents.heuristics.al_agent_random_sampling import AL_agent_random_sampling
from AL_agents.heuristics.al_agent_diversity_sampling import AL_agent_diversity_sampling
from AL_agents.heuristics.al_agent_representative_sampling import AL_agent_representative_sampling
from AL_agents.ensemble.al_agent_ensemble import AL_agent_Ensemble

class AL_Agent_Parameters():
    def __init__(self, agentName: str = "Random",batchSize_annotation=1,
                 batchSize_agent=1, beta_dict=None):
        self.agentName = agentName
        self.batchSize_annotation = batchSize_annotation
        if batchSize_agent <= 0:
            batchSize_agent = batchSize_annotation
        self.batchSize_agent = batchSize_agent
        if beta_dict:
            self.beta_dict = beta_dict

    def createAgent(self):
        if self.agentName == "Random":
            agent = AL_agent_random_sampling(self)
        elif self.agentName == "Uncertainty":
            agent = AL_agent_uncertainty_sampling(self)
        elif self.agentName == "Diversity":
            self.batchSize_agent = 1
            agent = AL_agent_diversity_sampling(self)
        elif self.agentName == "Representative":
            agent = AL_agent_representative_sampling(self)
        elif self.agentName == "Ensemble":
            self.batchSize_agent = 1
            agent = AL_agent_Ensemble(self)
        else:
            print(f"ERROR: agentName unknown: {self.agentName}")
            raise ValueError

        return agent

    def _getRelevantAttributeDict(self):
        attributes = self.__dict__.copy()
        return attributes

    def __repr__(self):
        return str(self._getRelevantAttributeDict())

    def __shortRepr__(self):
        agentName =  str(self.agentName)
        if agentName not in ["Random","Diversity","Representative"]:
            if self.batchSize_annotation > 1:
                agentName += "_" + str(self.batchSize_annotation)
            else:
                agentName += "_sequential"
        return agentName
