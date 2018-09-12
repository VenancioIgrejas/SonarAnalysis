import os
import numpy as np
from ModelPath import ModelPath

class SpecPath(ModelPath):
    """docstring forSpecPath."""
    def __init__(self,trnParams,spec_num):
        super(SpecPath, self).__init__(trnParams)
        self.num = spec_num

    def spec_model(self):
        return os.path.join("{0}_spec_".format(self.num),self.model)

    def spec_figures(self):
        return os.path.join("{0}_spec_".format(self.num),self.figures)

if __name__ == '__main__':
    from Functions import TrainParameters
    trnparams = TrainParameters.SpecialistClassificationTrnParams()
    testeSpec = SpecPath(trnparams,1)
    print testeSpec.spec_model()
