import os
import numpy as np
from Functions.ModelPath import ModelPath
from Functions import TrainParameters

class SpecPath(ModelPath):
    """docstring forSpecPath."""
    def __init__(self,trnParams,spec_num):
        super(SpecPath, self).__init__(trnParams)
        self.num = spec_num
        self.path_fold = {}
        self.spec_model = "{0}_spec_".format(spec_num) + self.model()
        self.spec_figures = "{0}_spec_".format(spec_num) + self.figures()
        for ifold in range(trnParams.folds):
            self.path_fold[ifold] = os.path.join(self.spec_model,'{0}_fold'.format(ifold))


    def spec_model_fold(self,num_fold):
        return self.path_fold[num_fold]


if __name__ == '__main__':
    from Functions import TrainParameters
    trnparams = TrainParameters.SpecialistClassificationTrnParams()
    testeSpec = SpecPath(trnparams,1)
    print testeSpec.spec_model_fold(0)
