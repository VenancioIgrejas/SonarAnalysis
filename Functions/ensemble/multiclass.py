import os

import numpy as np
from numpy import linalg

import heapq

from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier, _fit_binary, _ConstantPredictor, _num_samples, _predict_binary, check_is_fitted
from sklearn.multiclass import _fit_ovo_binary
from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import (_check_partial_fit_first_call,
                               check_classification_targets,
                               _ovr_decision_function)

from joblib import Parallel, delayed

from Functions.util import run_once,rm_permutation_tuple

def _fit_binary(estimator, X, y,selfClass,sample_weight,classes=None):
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

        #check if estimator has 'dir' as parameter
        #then create a dir to save meta_date of the estimator
        if hasattr(estimator, 'dir'):
          path = getattr(selfClass, 'dir') + '{0:02d}_specialist/'.format(selfClass.i_estimator)
          selfClass.i_estimator+=1

          if not os.path.exists(path):
              os.makedirs(path)

          estimator.set_params(**dict([('dir',path)]))

        #DEPRECABLE: it is better use validation_id istead of validation_data
        if hasattr(estimator, 'validation_data'):
             val_data = getattr(estimator,'validation_data')

             if not val_data is None:
                 #transform val_data in list for change value (tuples are immutable)
                 val_data = [i for i in val_data]
                 print(selfClass.val_colum_generetor(val_data[1]))
                 val_data[1] = selfClass.val_columns.next()
                 print val_data[1]

                 estimator.set_params(**dict([('validation_data',tuple(val_data))]))

        estimator.fit(X, y, sample_weight=sample_weight)
    return estimator

def _second_argmax_cm(cm):
    """ return list of indexes according second value maximun in each row"""
    list_indexes  = []
    for irow,rows in enumerate(cm):
        res = heapq.nlargest(2,rows)
        icolumn = np.where(res[1] == rows)[0][0]
        if icolumn == irow:
            raise ValueError("the second value max can not be "
            "the accuracy of class {0}".format(irow+1))
        list_indexes.append((irow,icolumn))
    return list_indexes


class SpecialistClass(OneVsRestClassifier):
    """docstring for SpecialistClass."""
    def __init__(self,estimator,n_jobs=None,dir='./',pred_mode="max_criteria",verbose=0):
        super(SpecialistClass, self).__init__(estimator=estimator,
                                              n_jobs=n_jobs)

        self.dir = dir
        self.pred_mode = pred_mode
        self.verbose = verbose
        self.i_estimator = 0
        self.weight_integrator = None

    @run_once
    def val_colum_generetor(self,y_val):
        label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = label_binarizer_.fit_transform(y_val)
        Y = Y.tocsc()
        self.val_columns = (col.toarray().ravel() for col in Y.T)
        return 0

    def fit(self, X, y, sample_weight=None):
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
        self.i_estimator = 0

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
            self.estimator, X, column, self, sample_weight, classes=[
                "not %s" % self.label_binarizer_.classes_[i],
                self.label_binarizer_.classes_[i]])
            for i, column in enumerate(columns))

        # Creating vector trgt for each
        if not self.pred_mode == "max_criteria":
          print("aqui")
          return self

        #creating vector trgt for each'
        if self.verbose:
          print("creating weights for the integration of specialists")

        #get validation data of estimators
        try:
          _, test_id = self.estimator.get_params()['validation_id']
        except KeyError:
          raise Exception("%s doesn't have \"validation_id\" as parameter,it's necessary for creating weights in maximum criterion" %str(self.estimator))

        val_trgt_col = (col.toarray().ravel() for col in Y[test_id].T)
        n_samples = _num_samples(X)
        n_specialist = len(self.classes_)

        #create vector of prediction size NxK, where N is number of feactures and K is number of classes
        vector_pred = np.vstack((_predict_binary(e,X[test_id]) for e in self.estimators_)).T

        self.weight_integrator = {}

        for i,column in enumerate(val_trgt_col):
            self.weight_integrator[i] = np.dot(linalg.pinv(vector_pred),column.T)

        if self.verbose:
            print("finished creation of weights")

        return self

    def predict(self, X):
        """Predict multi-class targets using underlying estimators.
        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.
        Returns
        -------
        y : (sparse) array-like, shape = [n_samples, ], [n_samples, n_classes].
            Predicted multi-class targets.
        """
        check_is_fitted(self, 'estimators_')
        if (hasattr(self.estimators_[0], "decision_function") and
                is_classifier(self.estimators_[0])):
            thresh = 0
        else:
            thresh = .5

        n_samples = _num_samples(X)
        if self.label_binarizer_.y_type_ == "multiclass":
            if self.pred_mode == "max_criteria":
                #code for use maxima criteria predict
                if self.verbose:
                    print("start prediction using maximum criterion")
                n_class = len(self.classes_)
                vector_pred = np.vstack((_predict_binary(e,X) for e in self.estimators_)).T
                output_pred = np.vstack(
                    (np.dot(vector_pred,self.weight_integrator[key_weight]
                           ) for key_weight in range(n_class))).T
                return self.classes_[np.argmax(output_pred,axis=1)]
            maxima = np.empty(n_samples, dtype=float)
            maxima.fill(-np.inf)
            argmaxima = np.zeros(n_samples, dtype=int)
            for i, e in enumerate(self.estimators_):
                pred = _predict_binary(e, X)
                np.maximum(maxima, pred, out=maxima)
                argmaxima[maxima == pred] = i
            return self.classes_[np.array(argmaxima.T)]
        else:
            indices = array.array('i')
            indptr = array.array('i', [0])
            for e in self.estimators_:
                indices.extend(np.where(_predict_binary(e, X) > thresh)[0])
                indptr.append(len(indices))
            data = np.ones(len(indices), dtype=int)
            indicator = sp.csc_matrix((data, indices, indptr),
                                      shape=(n_samples, len(self.estimators_)))
            return self.label_binarizer_.inverse_transform(indicator)

class HierarqNet(object):
    """docstring for HierarqNet.

    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing `fit` and one of `decision_function`
        or `predict_proba`.

    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    """
    def __init__(self, estimator, n_jobs=None, dir='./'):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.dir = dir

    def fit(self, X, y):
        """fit underlying estimators """

        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'])
        check_classification_targets(y)

        self.classes_ = np.unique(y)
        if len(self.classes_) == 1:
            raise ValueError("HierarqNet can not be fit when only one"
                             " class is present.")

        n_classes = self.classes_.shape[0]

        estimator_lv1 = clone(self.estimator)

        #check if estimator has 'dir' as parameter
        #then create a dir to save meta_date of the estimator
        if hasattr(estimator_lv1, 'dir'):
          path = getattr(self, 'dir') + 'lv1_estimator/'

          if not os.path.exists(path):
              os.makedirs(path)

        estimator_lv1.set_params(**dict([('dir',path)]))

        #get validation data of estimators
        try:
            _, test_id = self.estimator.get_params()['validation_id']
        except KeyError:
            raise ValueError("%s doesn't have \"validation_id\" as parameter,"
            "it's necessary for create the confusion matrix" %str(self.estimator))

        estimator_lv1.fit(X, y)
        y_pred = estimator_lv1.predict(X[test_id])
        y_true = y[test_id]

        cm = confusion_matrix(y_true, y_pred)

        print(cm)

        indexes_cm = _second_argmax_cm(cm)

        indexes_cm = rm_permutation_tuple(indexes_cm)

        estimators_indices = list(zip(*(Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_ovo_binary)
            (self.estimator, X, y, self.classes_[irow], self.classes_[icolumn])
            for irow,icolumn in indexes_cm))))

        self.estimators_ = estimators_indices[0]

        return estimators_indices
