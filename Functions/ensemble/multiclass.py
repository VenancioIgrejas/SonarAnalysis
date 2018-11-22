import numpy as np
import os

from sklearn.base import clone
from sklearn.preprocessing import LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier,_fit_binary,_ConstantPredictor
from joblib import Parallel, delayed

def _fit_binary(estimator, X, y,selfClass, classes=None):
    """Fit a single binary estimator."""
    unique_y = np.unique(y)
    if len(unique_y) == 1:
        if classes is not None:
            if y[0] == -1:
                c = 0
            else:
                c = y[0]
            warnings.warn("Label %s is present in all training examples." %
                          str(classes[c]))
        estimator = _ConstantPredictor().fit(X, unique_y)
    else:

        estimator = clone(estimator)

        path = getattr(selfClass,'dir') + '{0:02d}_specialist/'.format(classes[1])

        if not os.path.exists(path):
            os.makedirs(path)

        estimator.set_params(**dict([('dir',path)]))


        estimator.fit(X, y)
    return estimator


class SpecialistClass(OneVsRestClassifier):
    """docstring for SpecialistClass."""
    def __init__(self,estimator,n_jobs=None,dir='./'):
        super(SpecialistClass, self).__init__(estimator=estimator,
                                              n_jobs=n_jobs)

        self.dir = dir

    def fit(self, X, y):
        """Fit underlying estimators.
        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.
        y : (sparse) array-like, shape = [n_samples, ], [n_samples, n_classes]
            Multi-class targets. An indicator matrix turns on multilabel
            classification.
        Returns
        -------
        self
        """
        # A sparse LabelBinarizer, with sparse_output=True, has been shown to
        # outperform or match a dense label binarizer in all cases and has also
        # resulted in less or equal memory consumption in the fit_ovr function
        # overall.
        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y)
        Y = Y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        columns = (col.toarray().ravel() for col in Y.T)
        # In cases where individual estimators are very fast to train setting
        # n_jobs > 1 in can results in slower performance due to the overhead
        # of spawning threads.  See joblib issue #112.
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(delayed(_fit_binary)(
            self.estimator, X,column,self,classes=[
                "not %s" % self.label_binarizer_.classes_[i],
                self.label_binarizer_.classes_[i]])
            for i, column in enumerate(columns))

        return self
