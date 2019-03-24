import os
import numpy as np
from joblib import Parallel, delayed
from Functions.TrainParameters import ClassificationFolds
from sklearn.utils.class_weight import compute_class_weight


def class_weight_Keras(y,has_classWeights=True,class_weight='balanced'):

        if has_classWeights:
            return dict(zip(
                           np.unique(y),compute_class_weight(
                           class_weight=class_weight,classes=np.unique(y),y=y)))
        
        return None

class CrossValidation(object):
    """docstring for CVClassifier."""
    def __init__(self, X, y, estimator=None,n_folds=2,dev=False,verbose=False,dir='./'):
        #super(CVClassifier, self).__init__()
        #self.arg = arg

        if not os.path.exists(dir):
            os.makedirs(dir)
            
        self.CVO = ClassificationFolds(dir,n_folds,y,dev,verbose)
        self.dir = dir
        self.data = X
        self.n_folds = n_folds
        self.trgt = y
        self.estimator = estimator

    def get_folder(self,ifold=0):
        path = os.path.join(self.dir,"fold{0:02d}".format(ifold))
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def train_test_split(self,ifold=0):

        train_id, test_id = self.CVO[ifold]

        folder = self.get_folder(ifold)

        return train_id, test_id, folder

    def fit_ifold(self,ifold=0,sample_weight=None):

        train_id, test_id = self.CVO[ifold]

        folder = self.get_folder(ifold)

        params = {'dir':folder}

        self.estimator.set_params(**params)

        self.estimator.fit(X=self.data,
                      y=self.trgt,
                      train_id=train_id,
                      test_id=test_id,
                      sample_weight=sample_weight)


    def predict_ifold(self,ifold=0,mode='test',predict='classes'):

        train_id, test_id = self.CVO[ifold]

        if mode is 'test':
            return self.estimator.predict(self.data[test_id])
        if mode is 'all':
            return self.estimator.predict(self.data)

    def fit(self,n_jobs=1,folds=None,sample_weight=None):

        if folds is None:
            folds = range(self.n_folds)

        if not isinstance(folds,list):
            raise ValueError("expected list type of variable folds, but is {0} type".format(type(folds)))

        Parallel(n_jobs=n_jobs)(
                                delayed(self.fit_ifold)(ifolds,sample_weight)
                                for ifolds in folds)

    def score_ifold(self,ifold=0,mode='test',predict='classes'):

        train_id, test_id = self.CVO[ifold]

        if mode is 'test':
            return self.estimator.score(self.data[test_id])
        if mode is 'all':
            return self.estimator.score(self.data)
 
    def target_true(self, ifold=0, mode='test'):

        train_id, test_id = self.CVO[ifold]

        if mode is 'test':
            return self.trgt[test_id]

        if mode is 'all':
            return self.trgt


class CVEnsemble(CrossValidation):
    """docstring for CVEnsemble."""
    def __init__(self,meta_estimator,X,y,n_folds=2,dev=False,verbose=False,dir='./'):
        super(CVEnsemble, self).__init__(estimator=meta_estimator,X=X,y=y,n_folds=n_folds,dev=dev,verbose=verbose,dir=dir)

    def fit_ifold(self,ifold=0,sample_weight=None):

        train_id, test_id = self.CVO[ifold]

        folder = self.get_folder(ifold)

        params = {'dir': folder}
        trn_params = {'train_id':train_id,
                      'test_id':test_id}

        self.estimator.base_estimator.set_params(**params)

        self.estimator.set_fit_param(**trn_params)

        self.estimator.fit(X=self.data,
                      y=self.trgt,
                      sample_weight=sample_weight)
