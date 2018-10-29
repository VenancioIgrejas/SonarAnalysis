import numpy as np

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
        pass

    def majority_voting_function(self,X):
        if self.algorithm == 'SAMME.R':
            pass

        return None
