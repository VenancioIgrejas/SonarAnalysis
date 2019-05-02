import os
import sys
import time
import datetime
from joblib import dump, load

import numpy as np
import pandas as pd

from Functions.preprocessing import CrossValidation

from sklearn import svm
from sklearn import datasets
from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix

from multiprocessing import Pool
from Functions.StatFunctions import sp_index

from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment('SVC_dev')

ex.observers.append(MongoObserver.create(
    url='mongodb://venancio:xxxxx@mongodb-1911-0.cloudclusters.net:10009/Development?authSource=admin&ssl=true&ssl_ca_certs=/home/venancio/Downloads/ca.pem',
    db_name='Development',
    ssl=True,
    ssl_ca_certs='/home/venancio/Downloads/ca.pem'
))

@ex.config
def cfg():
    MODEL_DIR = os.path.join(os.environ['OUTPUTDATAPATH'], 'SVM')
    datasets='Iris'
    ifold = 0
    n_folds = 10
    verbose = 1
    verbose_train = 0
    C = 1
    kernel = 'rbf'
    gamma = 'auto'
    tol = 1e-3
    cache_size = 200
    class_weight = 'balanced'
    max_iter = -1
    decision_functions_shape = 'ovr'
    num_processes = 3


@ex.automain
def run(_run, MODEL_DIR, n_folds, ifold, verbose, verbose_train, C, kernel, gamma, tol, cache_size, class_weight, max_iter, decision_functions_shape, num_processes):
    iris = datasets.load_iris()

    all_data = iris.data
    all_trgt = iris.target

    # In[3]:

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    cv = CrossValidation(X=all_data,
                         y=all_trgt,
                         estimator=None,
                         n_folds=n_folds,
                         dev=True,
                         verbose=verbose,
                         dir=MODEL_DIR)

    params_SVC = {
        'C': C,
        'kernel': kernel,
        'gamma': gamma,
        'tol': tol,
        'cache_size': cache_size,
        'class_weight': class_weight,
        'max_iter': max_iter,
        'decision_function_shape': decision_functions_shape
    }

    # In[15]:


    train_id, test_id, folder = cv.train_test_split(ifold=ifold)

    pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svm_clf', svm.SVC(verbose=verbose_train, **params_SVC))
        ])

    pipe.fit(X=all_data[train_id], y=all_trgt[train_id])

    pred = pipe.predict(all_data)

    dump(pipe, folder + '/model.jbl', compress=9)

    _run.add_artifact(filename=folder + '/model.jbl',name='model')

    # only for control of Net
    csv_master = folder + '/master_table.csv'
    if not os.path.exists(csv_master):
        fold_vect = np.zeros(shape=all_trgt.shape, dtype=int)
        fold_vect[test_id] = 1

        pd.DataFrame({'target': all_trgt, 'fold_0{0}'.format(ifold): fold_vect, 'predict': pred}).to_csv(csv_master,
                                                                                                             index=False)

    _run.add_artifact(filename=csv_master, name='table master')
    return sp_index(y_true=all_trgt[test_id], y_pred=pred[test_id])