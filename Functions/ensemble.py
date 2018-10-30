import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.base import BaseEstimator,ClassifierMixin


class AdaBoost(AdaBoostClassifier):
    """docstring for AdaBoost."""
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.0,
                 algorithm='SAMME.R',
                 random_state=None):
        super(AdaBoost, self).__init__(base_estimator=base_estimator
                                       , n_estimators=n_estimators,
                                       learning_rate=learning_rate,
                                       algorithm=algorithm,
                                       random_state=random_state)
        self.le_=None

    def set_classes(self,y):
        """ Enconde label"""
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_

    def _predict_mjvt(self,X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self.estimators_]).T

    def predict_maj(self,X):
        """ Predict class labels for X using majory voting.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.
        """

        if self.le_ is None:
            raise Exception('call first set_classes function')

        predictions = self._predict_mjvt(X)
        maj = np.apply_along_axis(lambda x: np.argmax(
                                        np.bincount(x)),
                                        axis=1,arr=predictions)

        maj = self.le_.inverse_transform(maj)

        return maj
