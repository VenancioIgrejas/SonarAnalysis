import os
import numpy as np

class IntegratorPath(object):
    """docstring for ModelPath."""
    def __init__(self,resultPath,fold):
        self.resultpath = resultPath
        self.path_fold = {}
        for i in range(fold):
            self.path_fold[i] = resultPath + '/{0}_fold'.format(i)
        
    def weight_file(self,ifold,spec_num):
        file = self.path_fold[spec_num] + '/{0}_spc'.format(spec_num)
        if not os.path.exists(file):
            os.makedirs(file)
        return file +'/weight.csv'
    
    def analysis_fold(self,num_analise):
        folder = self.resultpath + '/Spec_analysis_{0}'.format(num_analise)
        if not os.path.exists(folder):
            os.makedirs(folder)
        return folder

