import os
import sys
import json

import numpy as np
import pandas as pd
from Functions.StatFunctions import sp_index

from Functions.preprocessing import CrossValidation
from Functions.kerasclass import MLPKeras, Preprocessing
from Functions.callbackKeras import metricsAdd, StopTraining
from keras.callbacks import CSVLogger, ModelCheckpoint

from Functions.dataset.shipClasses import LoadData
from Functions.dataset.path import BDControl

from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver

ex = Experiment('MLP_Simples')

ex.observers.append(MongoObserver.create(
    url='mongodb://venancio:xxxxx@mongodb-1911-0.cloudclusters.net:10009/Development?authSource=admin&ssl=true&ssl_ca_certs=/home/venancio/Downloads/ca.pem',
    db_name='Development',
    ssl=True,
    ssl_ca_certs='/home/venancio/Downloads/ca.pem'
))

ex.observers.append(FileStorageObserver.create(
    basedir=os.path.join(os.environ['OUTPUTDATAPATH'], 'Classification', 'bd')
))


@ex.config
def conf():
    neurons = 100
    ifold = 0
    activation = ('tanh', 'tanh')
    optimizer = 'adam'
    optimizer_kwargs = dict(lr=0.001)
    loss = 'mean_squared_error'
    n_init = 3
    batch_size = 512
    epochs = 1000
    dev = True
    database = '24classes'
    weights = True
    folds = 10
    verbose_train = False
    verbose = True
    dev = False


#  processes = 3       sacred doesn't suport multiprocess


@ex.automain
def run(_run,
        database,
        dev,
        loss,
        neurons,
        activation,
        optimizer,
        optimizer_kwargs,
        n_init,
        batch_size,
        epochs,
        weights,
        folds,
        ifold,
        verbose_train,
        verbose):
    analysis_name = 'Classification'

    data_path = os.environ['OUTPUTDATAPATH']
    results_path = os.environ['PACKAGE_NAME']

    # Function for Dataset

    dt_lofar_power = LoadData(database=database, dev=dev)
    dt_lofar_power.infoData()
    all_data, all_trgt = dt_lofar_power.getData()

    metrics = ['acc', 'sp']

    info_net = {'weight': weights, 'n_inits': n_init, 'n_neurons': neurons,
                'optmizerAlgorithm': optimizer, 'n_folds': folds,
                'batch_size': batch_size,
                'loss': loss, 'n_epochs': epochs,
                'metrics': ','.join(metrics),
                'output_activation': activation[1],
                'hidden_activation': activation[0],
                'PCD': False,
                'database': database,
                'type_arq': 'MLP',
                'version': 'V1',
                'analysis_name': analysis_name,
                'dev': dev}

    bdc = BDControl(main_path=results_path, columns=info_net.keys())

    results_path_specific = bdc.get_path(**info_net)

    if not os.path.exists(results_path_specific):
        os.makedirs(results_path_specific)

    _run.info["Path_father"] = results_path_specific

    # save configuration inside of path case something wrong happen
    with open(results_path_specific + '/info_net.json', 'w') as fp:
        json.dump(info_net, fp)

    cv = CrossValidation(X=all_data,
                         y=all_trgt,
                         estimator=None,
                         n_folds=folds,
                         dev=dev,
                         verbose=verbose,
                         dir=results_path_specific)

    train_id, test_id, folder = cv.train_test_split(ifold=ifold)

    _run.info["Path"] = folder

    # only for control of Net
    csv_master = folder + '/master_table.csv'
    if not os.path.exists(csv_master):
        fold_vect = np.zeros(shape=all_trgt.shape, dtype=int)
        fold_vect[test_id] = 1

        pd.DataFrame({'target': all_trgt, 'fold_{0}'.format(ifold): fold_vect}).to_csv(csv_master, index=False)

    # pre-processing of data

    ppc = Preprocessing()

    ppc.set_transform(X=all_data[train_id], y=all_trgt[train_id], fit=True)
    X_scaler_train = ppc.get_transform()

    ppc.set_transform(X=all_data[test_id], y=all_trgt[test_id], fit=False)
    X_scaler_test = ppc.get_transform()

    if weights:
        ppc.set_class_weight(y=all_trgt)

    ppc.set_sparce(y=all_trgt[train_id])
    y_sparce_train = ppc.get_sparce()

    ppc.set_sparce(y=all_trgt[test_id])
    y_sparce_test = ppc.get_sparce()

    # callbacks of Keras fit

    st = StopTraining(restore_best_weights=True,
                      verbose=True,
                      patience=25,
                      min_delta=10 ^ -4)

    ma = metricsAdd('sp', verbose=verbose_train)

    csv = CSVLogger(filename='./')

    mcheck = ModelCheckpoint(filepath='./', verbose=verbose_train, save_weights_only=False)

    mlp = MLPKeras(hidden_layer_sizes=(neurons,),
                   activation=activation,
                   optimizer=optimizer,
                   loss=loss,
                   optimizer_kwargs=optimizer_kwargs,
                   n_init=n_init,
                   fit_kwargs={'batch_size': batch_size,
                               'epochs': epochs,
                               'verbose': verbose_train,
                               'validation_data': (X_scaler_test, y_sparce_test),
                               'class_weight': ppc.get_weights()},
                   callbacks_list=[ma, st, csv, mcheck],
                   dir=folder)

    mlp.fit(X=all_data, y=all_trgt)

    pred = mlp.predict(X=all_data)

    pd.DataFrame(pred, columns=[
        'neuron_{0}'.format(i) for i in range(pred.shape[1])
    ]).to_csv(folder + '/predict.csv', index=False)

    return sp_index(y_true=all_trgt[test_id], y_pred=np.argmax(pred, axis=1)[test_id])
