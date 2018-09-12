import os
import numpy as np

class ModelPath(object):
    """docstring for ModelPath."""
    def __init__(self,trnParams):
        #super(ModelPath, self).__init__()
        self.path = trnParams.get_params_str()

    def model(self):
        return os.path.join(self.path,"model")

    def figures(self):
        return os.path.join(self.path,"figures")
