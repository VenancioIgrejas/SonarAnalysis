import os
import numpy as np
from ModelPath import ModelPath

class SpecPath(ModelPath):
    """docstring forSpecPath."""
    def __init__(self,trnParams,spec_num):
        super(SpecPath, self).__init__(trnParams)
        self.num = spec_num
        self.path_fold = {}
        self.spec_model = os.path.join("{0}_spec_".format(self.num),self.model)
        self.spec_figures = os.path.join("{0}_spec_".format(self.num),self.figures)
        for ifold in range(trnParams.n_folds):
            self.path_fold[ifold] = os.path.join(self.spec_model,'{0}_fold'.format(ifold))


    def spec_model_fold(num_fold):
        return self.path_fold[ifold]


if __name__ == '__main__':
    from Functions import TrainParameters
    trnparams = TrainParameters.SpecialistClassificationTrnParams()
    testeSpec = SpecPath(trnparams,1)
    print testeSpec.spec_model()
