
class Task_bAbI_variantParams:
    def __init__(self, challenge_type: str = 'single_supporting_fact_10k', no_epochs=30):
        if challenge_type not in ['single_supporting_fact_10k','two_supporting_facts_10k']:
            raise ValueError
        self.type = challenge_type
        self.no_epochs = no_epochs


    def __shortRepr__(self):
        repr = "bAbI_"
        repr += self.type
        repr.replace('_10','')
        return repr

    def __repr__(self):
        selfDict = self.__dict__
        return str([selfDict[key] for key in sorted(selfDict.keys(), reverse=False)])


    def isEqual(self, other):
        isEqual = True
        isEqual = isEqual and self.challenge
        return isEqual
