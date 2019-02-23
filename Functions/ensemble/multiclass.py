import os
import sys

import numpy as np
from numpy import linalg
import pandas as pd

import array
import scipy.sparse as sp
import heapq

from sklearn.base import clone,is_classifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier, _fit_binary, _ConstantPredictor, _num_samples, _predict_binary, check_is_fitted
from sklearn.multiclass import _fit_ovo_binary
from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import (_check_partial_fit_first_call,
                               check_classification_targets,
                               _ovr_decision_function)

from joblib import Parallel, delayed

from keras import backend as K

from Functions.ensemble.utils import logical_or
from Functions.preprocessing import CrossValidation

from Functions.util import percent_of_each_class, file_exist
from Functions.util import run_once,rm_permutation_tuple

from lps_toolbox.metrics.classification import sp_index

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

        #try this part of code after train
        #if selfClass.clear_session:
        #    K.clear_session()
        #    if selfClass.verbose:
        #        print("")

        estimator.fit(X, y, sample_weight=sample_weight)
    return estimator

#DEPRECABLE: HierarqNet Not using confusion matrix as decision fit
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

def _best_estimator(estimator, X, y, cv, n_jobs=None,verbose=False):
    """ fit k-folders estimators and choose the best according to SP criterion
    """
    estimators_ = [clone(estimator) for i in range(cv.n_folds)]

    sp_estimators = []

    if file_exist(cv.dir +'SP_folds.csv'):
        #Load model file case exist SP of all folders
        file_sp = cv.dir +'SP_folds.csv'
        if verbose:
            print("SP of folders already exists in {0} file".format(file_sp))
        
        SP_folds = pd.read_csv(file_sp)['SP'].values
    
        if verbose:
            print("fold {0} has the highest sp value: {1:.2f}".format(np.argmax(SP_folds)+1,SP_folds[np.argmax(SP_folds)]))
        
        train_id, test_id, folder = cv.train_test_split(np.argmax(SP_folds))

        best_estimator = estimators_[np.argmax(SP_folds)]
        if hasattr(best_estimator,'validation_id') and hasattr(best_estimator, 'dir'):

            best_estimator.set_params(**dict([('dir',folder),('validation_id',(train_id,test_id))]))
            best_estimator.fit(X, y)
        else:
            #case self.estimator is a meta estimator (ex. Specialist Class)
            if hasattr(best_estimator.estimator,'validation_id') and hasattr(best_estimator.estimator, 'dir'):
                best_estimator.set_params(**dict([('dir',folder)]))
                #print iestimator.get_params()
                best_estimator.estimator.set_params(**dict([('validation_id',(train_id,test_id))]))
                #sys.exit()
                best_estimator.fit(X, y)
            else:
                raise ValueError("{0} not has 'validation_id' or 'dir' as parameters ".format(best_estimator))

        return best_estimator

    for ifold in range(cv.n_folds):
        if verbose:
            print("fold {0} of {1} ".format(ifold+1,cv.n_folds))

        iestimator = estimators_[ifold]

        train_id, test_id, folder = cv.train_test_split(ifold)

        if hasattr(iestimator,'validation_id') and hasattr(iestimator, 'dir'):

            iestimator.set_params(**dict([('dir',folder),('validation_id',(train_id,test_id))]))
            iestimator.fit(X, y)
        else:
            #case self.estimator is a meta estimator (ex. Specialist Class)
            if hasattr(iestimator.estimator,'validation_id') and hasattr(iestimator.estimator, 'dir'):
                iestimator.set_params(**dict([('dir',folder)]))
                #print iestimator.get_params()
                iestimator.estimator.set_params(**dict([('validation_id',(train_id,test_id))]))
                #sys.exit()
                iestimator.fit(X, y)
            else:
                raise ValueError("{0} not has 'validation_id' or 'dir' as parameters ".format(iestimator))

        pred_sparce = iestimator.predict(X,predict='sparce')

        if (len(np.unique(y)) != 2) | (pred_sparce.shape[0] != len(pred_sparce.flatten())):
            y_pred = np.argmax(pred_sparce,axis=1)
            values = np.array([row[np.argmax(row)] for row in pred_sparce])
        else:
            #binary classification, we use t
            threshold = 0
            # if sample is higher then threshold, this sample belongs to class 1 otherwise 0
            y_pred =  np.array(map(lambda x: 1 if x >= threshold else 0,pred_sparce))
            values = pred_sparce

        folds = -np.ones(shape=(y.shape[0],),dtype=int)
        folds[train_id] = 0
        folds[test_id] = 1

        

        

        #save true target and predicted target as csv file
        pd.DataFrame({'true':y[test_id],'predict':y_pred[test_id]}).to_csv(folder+'true_predict.csv')



        
        if (pred_sparce.shape[0] != len(pred_sparce.flatten())):
            #if pred_sparce is binary, so the output of neuron will be the values in hierarq_analy_fold
            #create csv file for output of each neuron
            table_output = dict([('neuron_{0}'.format(i),pred_sparce[:,i]) for i in range(pred_sparce.shape[1])])
            table_output['fold_{0}'.format(ifold)] = folds
            pd.DataFrame(table_output).to_csv(folder+'output_sparce.csv',index=False)

        pd.DataFrame({
                      'target_{0}'.format(ifold):y,
                      'HierarqNet(resultados)_{0}'.format(ifold):y_pred,
                      'HierarqNet(valor)_{0}'.format(ifold):values,
                      'fold_{0}'.format(ifold):folds}).to_csv(folder+'hierarq_analy_fold_{0}.csv'.format(ifold),index=False)

        #save confusion matrix as csv file without index and header
        pd.DataFrame(confusion_matrix(y[test_id],y_pred[test_id])).to_csv(folder+'confusion_matrix.csv',header=False,index=False)

        sp = sp_index(y[test_id], y_pred[test_id])

        if verbose:
            print("sp value of fold {0} is {1:.3f}".format(ifold+1,sp))

        sp_estimators.append(sp)

        K.clear_session()


    pd.DataFrame({'SP':sp_estimators}).to_csv(cv.dir +'SP_folds.csv',index=False)

    bestSP_id = np.argmax(sp_estimators)



    if verbose:
        print("fold {0} has the highest sp value: {1:.2f}".format(bestSP_id+1,sp_estimators[bestSP_id]))

    return estimators_[bestSP_id].fit(X, y)

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
          return self

        #creating vector trgt for each'
        if self.verbose:
          print("creating weights for the integration of specialists")

        #get validation data of estimators
        try:
          train_id, _ = self.estimator.get_params()['validation_id']
        except KeyError:
          raise Exception("%s doesn't have \"validation_id\" as parameter,it's necessary for creating weights in maximum criterion" %str(self.estimator))

        val_trgt_col = (col.toarray().ravel() for col in Y[train_id].T)
        n_samples = _num_samples(X)
        n_specialist = len(self.classes_)

        #create vector of prediction size NxK, where N is number of feactures and K is number of classes

        #the function _predict_binary return score and in this case, the score is 
        # score = np.ravel(estimator.decision_function(X))
        # where ravel return a contiguous flattened array.
        matrix_pred = np.vstack((_predict_binary(e,X[train_id]) for e in self.estimators_)).T

        self.weight_integrator = {}

        self.column = []
        for i,column in enumerate(val_trgt_col):
            column = np.array(map(lambda x:2*x-1,column))
            self.column.append(column)
            self.weight_integrator[i] = np.dot(linalg.pinv(matrix_pred),column.T)

        if self.verbose:
            print("finished creation of weights")


        return self

    def predict(self, X, predict='classes'):
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
        
        output_sparse_each_spec = []
        if self.label_binarizer_.y_type_ == "multiclass":
            if self.pred_mode == "max_criteria":
                #code for use maxima criteria predict
                if self.verbose:
                    print("start prediction using maximum criterion")
                n_class = len(self.classes_)
                matrix_pred = np.vstack((_predict_binary(e,X) for e in self.estimators_)).T
                for ispec, istimator in enumerate(self.estimators_):
                    print("[+] start predict of spec {0}".format(ispec))
                    output_sparse_each_spec.append(istimator.decision_function(X,predict='sparce'))

                output_pred = np.vstack(
                    (np.dot(matrix_pred,self.weight_integrator[key_weight]
                           ) for key_weight in range(n_class))).T

                self._output_pred = output_pred
                self._matrix_pred= matrix_pred
                self._output_sparse_each_spec=output_sparse_each_spec

                if predict is 'sparce':
                    return output_pred

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
    """docstring for HierarqNet"""
    def __init__(self,estimator,n_jobs=None,verbose=False,n_folds=None,dir='./'):
        
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.n_folds = n_folds
        self.dir=dir
        self.classes_ = None
        self.classesLVL1_ = None
        self.estimators_ = {}

    def _fit_estimator(self, X, y, estimator, path_est, group_classes):
        # lebrar de tirar o estimator como parametro, eh mais facil usar o metodo da propria classe

        name_class = path_est.split('_')[1][:-1]

        if self.verbose:
            print("[+] training class {0}".format(name_class))

        estimator = clone(self.estimator)

        if hasattr(estimator, 'dir'):
            path = getattr(self, 'dir') + path_est#'estimator_A/'

            if not os.path.exists(path):
                os.makedirs(path)
        
            estimator.set_params(**dict([('dir',path)]))

        #classes_of_A = [[9,10,13,14,16],
        #                [23,1,2,22],
        #                [21]]

        classe = 0
        concat_df_X = []
        concat_df_y = []

        ids = []
        for super_class in group_classes:
            super_class = map(lambda x:x-1, super_class)

            samples = logical_or(y, super_class)

            ids.append(samples)
            concat_df_X.append(pd.DataFrame(X[samples]))
            concat_df_y.append(pd.DataFrame(classe*np.ones((y[samples].shape[0],),dtype=int)))

            classe +=1

        X_new = pd.concat(concat_df_X).values

        y_new = pd.concat(concat_df_y).values

        #print path

        y_new = y_new.reshape((y_new.shape[0],))
        #print y_new
        percent_of_each_class(X_new, y_new)

        cv = CrossValidation(X= X_new, y= y_new, 
                             n_folds=self.n_folds, dev=False,
                             verbose=self.verbose, dir=path)
        #return cv,y_new,y

        estimator = _best_estimator(estimator=estimator, 
                                        X= X_new,
                                        y= y_new, 
                                        cv=cv, 
                                        n_jobs=None,
                                        verbose=self.verbose)


        return estimator, X_new, y_new, ids

    def _fit_lvl1(self, estimator, X, y):

        self.classes_of_lvl1   = [[9,10,13,14,16,23,1,2,22,21],
                             [4,6,8,12,17,19],
                             [11,24],
                             [5,7,15,3,18,20]]


        self.estimators_['S'] = self._fit_estimator(X, y,
                                                    estimator,
                                                    path_est='estimator_super/',
                                                    group_classes=self.classes_of_lvl1)

    def _fit_lvl2(self, estimator, X, y):

        if self.verbose:
            print "Start train of Medium Class (A,B,C,D) - LVL2"

        # estimator A
        self.classes_of_A = [[9,10,13,14,16],
                        [23,1,2,22],
                        [21]]

        self.estimators_['A'] = self._fit_estimator(X, y,
                                                estimator, 
                                                path_est='estimator_A/',
                                                group_classes=self.classes_of_A)


        # Estimator B

        self.classes_of_B = [[4],
                        [6],
                        [8],
                        [12],
                        [17],
                        [19]]

        self.estimators_['B'] = self._fit_estimator(X, y,
                                                estimator, 
                                                path_est='estimator_B/',
                                                group_classes=self.classes_of_B)

        # Estimator C

        self.classes_of_C = [[11],
                        [24]]
        
        self.estimators_['C'] = self._fit_estimator(X, y,
                                                estimator, 
                                                path_est='estimator_C/',
                                                group_classes=self.classes_of_C)

        #estimator D

        self.classes_of_D = [[5,7,15],
                        [3,18,20]]


        self.estimators_['D'] = self._fit_estimator(X, y,
                                                estimator, 
                                                path_est='estimator_D/',
                                                group_classes=self.classes_of_D)

    def _fit_lvl3(self, estimator, X, y):

        if self.verbose:
            print "Start train of Medium Class (AA,AB,DA,DB) - LVL3"


        #estimator AA
        self.classes_of_AA = [[9],
                         [10],
                         [13],
                         [14],
                         [16]]

        self.estimators_['AA'] = self._fit_estimator(X, y,
                                                estimator, 
                                                path_est='estimator_AA/',
                                                group_classes=self.classes_of_AA)

        #estimator AB
        self.classes_of_AB = [[23],
                         [1],
                         [2],
                         [22]]

        self.estimators_['AB'] = self._fit_estimator(X, y,
                                                estimator, 
                                                path_est='estimator_AB/',
                                                group_classes=self.classes_of_AB)

        #estimator AC only have one class (21), so we dont need train

        #estimator DA
        self.classes_of_DA = [[5],
                         [7],
                         [15]]

        self.estimators_['DA'] = self._fit_estimator(X, y,
                                                estimator, 
                                                path_est='estimator_DA/',
                                                group_classes=self.classes_of_DA)

        #estimator DB
        self.classes_of_DB = [[3],
                         [18],
                         [20]]

        self.estimators_['DB'] = self._fit_estimator(X, y,
                                                estimator, 
                                                path_est='estimator_DB/',
                                                group_classes=self.classes_of_DB)

    def _predict_estimator(self, X, name):
        """ return prection of estimador "name"
            
            Parameters
        ----------
        X : array-like, (number of samples X )
            Dataset

        name : {'S', 'A', 'B', 'C', 'D', 'AA','AB','AC', 'DA', 'DB'}
            name of choosed estimator for predict
        
        """

        shape_train  = self.estimators_[name][1].shape

        shape_pred = X.shape

        if shape_train != shape_pred:
            raise ValueError("shape of data is {0}, but excepted {0}".format(shape_pred,shape_train))

        return self.estimators_[name][0].predict(X) 

    def fit(self, X, y):
        """ fit underlying estimators """

        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'])
        check_classification_targets(y)

        self.classes_ = np.unique(y)
        if not len(self.classes_) == 24:
            raise ValueError("HierarqNet only can be fit with 24 classes")

        n_classes = self.classes_.shape[0]

        if hasattr(self.estimator,'validation_id'):
            if self.estimator.get_params()['validation_id'] != (None,None):
                raise ValueError("{0} except '(None,None)' as value of 'validation_id' parameter".format(self.estimator))

        if self.verbose:
            print("[+] start train of level 1 estimators")
        self._fit_lvl1(self.estimator, X, y)

        if self.verbose:
            print("[+] start train of level 2 estimators")
        self._fit_lvl2(self.estimator, X, y)
        
        if self.verbose:
            print("[+] start train of level 3 estimators")
        self._fit_lvl3(self.estimator, X, y)

    def predict(self, X):


        # global_pred will be updated by each class in the hierarchy
        global_pred = -np.ones(X.shape[0],dtype=int)
        
        #vector that return the position of global_pred
        # this vector will be use used for change each value of 
        # global_pred
        position = np.array([i for i in range(X.shape[0])])

        #predict of S -> [0,1,2,3]
        y_pred_lvl1 = self._predict_estimator(X, name='S')

        

        samples_lvl1 = ()
        for iclass in np.unique(y_pred_lvl1):
            samples_lvl1 += (y_pred_lvl1==iclass,)

        

        #--predict of A -> [0,1,2]
        samples_A = ()
        y_pred_A = self.estimators_['A'][0].predict(X[samples_lvl1[0]])
        for iclass in np.unique(y_pred_A):
            samples_A += (y_pred_A==iclass,)

        
        #----predict of AA -> [0,1,2,3,4,5]
        samples_AA = ()
        y_pred_AA = self.estimators_['AA'][0].predict(
                        X[samples_lvl1[0]][samples_A[0]]
                        )
        for iclass in np.unique(y_pred_AA):
            samples_AA += (y_pred_AA==iclass,)

        for iclass, true_class in enumerate([9,10,13,14,16]):
            global_pred[position
                       [samples_lvl1[0]]
                       [samples_A[0]]
                       [samples_AA[iclass]]
                       ] = true_class

        
        #----predict of AB -> [0,1,2,3]
        samples_AB = ()
        y_pred_AB = self.estimators_['AB'][0].predict(
                        X[samples_lvl1[0]][samples_A[1]]
                        )
        for iclass in np.unique(y_pred_AB):
            samples_AB += (y_pred_AB==iclass,)

        for iclass, true_class in enumerate([23,1,2,22]):
            global_pred[position
                       [samples_lvl1[0]]
                       [samples_A[1]]
                       [samples_AB[iclass]]
                       ] = true_class

        

        #----predict of AC -> [0]

        #    class AC have only the class 21, so all samples
        #    related to class 3 of A will be 21

        samples_AC = samples_A[2]
        global_pred[position
                   [samples_lvl1[0]]
                   [samples_AC]
                   ] = 21


        #--predict of B -> [0,1,2,3,4,5]

        samples_B = ()
        y_pred_B = self.estimators_['B'][0].predict(X[samples_lvl1[1]])
        for iclass in np.unique(y_pred_B):
            samples_B += (y_pred_B==iclass,)

        for iclass, true_class in enumerate([4,6,8,12,17,19]):
            global_pred[position
                       [samples_lvl1[1]]
                       [samples_B[iclass]]
                       ] = true_class

        #--predict of C -> [0,1]
        
        samples_C = ()
        y_pred_C = self.estimators_['C'][0].predict(X[samples_lvl1[2]])
        for iclass in np.unique(y_pred_C):
            samples_C += (y_pred_C==iclass,)

        for iclass, true_class in enumerate([11,24]):
            global_pred[position
                       [samples_lvl1[2]]
                       [samples_C[iclass]]
                       ] = true_class

        #--predict of D -> [0,1]
        samples_D = ()
        y_pred_D = self.estimators_['D'][0].predict(X[samples_lvl1[3]])
        for iclass in np.unique(y_pred_D):
            samples_D += (y_pred_D==iclass,)

        #----predict of DA -> [0,1,2]
        samples_DA = ()
        y_pred_DA = self.estimators_['DA'][0].predict(
                        X[samples_lvl1[3]][samples_D[0]]
                        )
        for iclass in np.unique(y_pred_DA):
            samples_DA += (y_pred_DA==iclass,)

        for iclass, true_class in enumerate([5,7,15]):
            global_pred[position
                       [samples_lvl1[3]]
                       [samples_D[0]]
                       [samples_DA[iclass]]
                       ] = true_class

        #----predict of DB -> [0,1,2]
        samples_DB = ()
        y_pred_DB = self.estimators_['DB'][0].predict(
                        X[samples_lvl1[3]][samples_D[1]]
                        )
        for iclass in np.unique(y_pred_DB):
            samples_DB += (y_pred_DB==iclass,)

        for iclass, true_class in enumerate([3,18,20]):
            global_pred[position
                       [samples_lvl1[3]]
                       [samples_D[1]]
                       [samples_DB[iclass]]
                       ] = true_class


        if any([i==-1 for i in global_pred]):
            raise ValueError("some sample does not belong"
                            " to one of the 24 classes")


        #transform global_pred for shape like y_true
        #ex: class 1 is represented as 0, class 2 is represented as 1
        #    and go on

        global_pred = map(lambda x:x-1,global_pred)

        return global_pred
