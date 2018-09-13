from OneSpecialistClass import OneSpecialistClass as SC

class SpecialistClass():
    """docstring for SpecialistClass."""
    def __init__(self,resultPath,trnParams,num_specialist):
        #super(SpecialistClass, self).__init__()
        #self.arg = arg
        specialist = {}
        for ispec in range(num_specialist):
            specialist[ispec] = SC()
