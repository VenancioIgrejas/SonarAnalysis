import os
from Functions.TrainParameters import ClassificationFolds


class CrossValidation(object):
    """docstring for CVClassifier."""
    def __init__(self,estimator,X,y,n_folds=2,dev=False,verbose=False,dir='./'):
        #super(CVClassifier, self).__init__()
        #self.arg = arg
        self.CVO = ClassificationFolds(dir,n_folds,y,dev,verbose)
        self.dir = dir
        self.data = X
        self.trgt = y
        self.estimator = estimator

    def get_folder(self,ifold=0):
        path = os.path.join(self.dir,"fold{0:02d}".format(ifold)) + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        return path

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

    def score_ifold(self,ifold=0,mode='test',predict='classes'):

        train_id, test_id = self.CVO[ifold]

        if mode is 'test':
            return self.estimator.score(self.data[test_id])
        if mode is 'all':
            return self.estimator.score(self.data)

class CVEnsemble(CrossValidation):
    """docstring for CVEnsemble."""
    def __init__(self,meta_estimator,X,y,n_folds=2,dev=False,verbose=False,dir='./'):
        super(CVEnsemble, self).__init__(estimator=meta_estimator,X=X,y=y,n_folds=n_folds,dev=dev,verbose=verbose,dir=dir)

    def fit_ifold(self,ifold=0,sample_weight=None):

        train_id, test_id = self.CVO[ifold]

        folder = self.get_folder(ifold)

        params = {'dir':folder}
        trn_params = {'train_id':train_id,
                      'test_id':test_id}

        self.estimator.base_estimator.set_params(**params)

        self.estimator.set_fit_param(**trn_params)

        self.estimator.fit(X=self.data,
                      y=self.trgt,
                      sample_weight=sample_weight)
