import numpy as np
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble.base import BaseEnsemble,_set_random_states
from sklearn.base import BaseEstimator,ClassifierMixin,clone


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
        self.kwarg = {}


    def _make_estimator(self, append=True, random_state=None):
        """Make and configure a copy of the `base_estimator_` attribute.
        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        estimator = clone(self.base_estimator_)

        path = getattr(estimator,'dir') + '{0}_estimator/'.format(len(self.estimators_))

        if not os.path.exists(path):
            os.makedirs(path)

        estimator.set_params(**dict([('dir',path)]))

        if random_state is not None:
            _set_random_states(estimator, random_state)

        if append:
            self.estimators_.append(estimator)

        return estimator

    def set_fit_param(**kwarg):
        self.kwarg = kwarg 

    def _boost_discrete(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME discrete algorithm."""
        estimator = self._make_estimator(random_state=random_state)

        kwarg = self.fit_kwarg

        estimator.fit(X, y, sample_weight=sample_weight,**kwarg)

        y_predict = estimator.predict(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        n_classes = self.n_classes_

        # Stop if the error is at least as bad as random guessing
        if estimator_error >= 1. - (1. / n_classes):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError('BaseClassifier in AdaBoostClassifier '
                                 'ensemble is worse than random, ensemble '
                                 'can not be fit.')
            return None, None, None

        # Boost weight using multi-class AdaBoost SAMME alg
        estimator_weight = self.learning_rate * (
            np.log((1. - estimator_error) / estimator_error) +
            np.log(n_classes - 1.))

        # Only boost the weights if I will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight * incorrect *
                                    ((sample_weight > 0) |
                                     (estimator_weight < 0)))

        return sample_weight, estimator_weight, estimator_error

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
