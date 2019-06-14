import os
import sys
import time
import datetime
import telegram_send
import json 

import argparse
#Argument Parse config

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau



import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from sklearn import metrics

from contextlib import contextmanager

from multiprocessing import Pool, TimeoutError


from Functions import TrainParameters
from Functions.dataset.path import BDControl

#from sklearn.neural_network import MLPClassifier
from Functions.preprocessing import CrossValidation,CVEnsemble
from Functions.ensemble import SpecialistClassification
from Functions.kerasclass import MLPKeras, Preprocessing
from Functions.callbackKeras import metricsAdd, StopTraining, EarlyStoppingKeras, ModelCheckpointKeras, CSVLoggerKeras, ReduceLROnPlateauKeras
from Functions.dataset.shipClasses import LoadData


from sklearn import datasets
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split,StratifiedKFold

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-n", "--neurons", default=10,type=int, help="Select number of neurouns in Neural Network")
parser.add_argument("-e", "--epochs", default=1000,type=int, help="Select number of epochs in Neural Network")
parser.add_argument("-i", "--init",default=10,type=int, help="Select number of inicialization in Neural Network")
parser.add_argument("-b", "--batch",default=32,type=int, help="Select number of inicialization in Neural Network")
parser.add_argument("-f", "--fold",default=10,type=int, help= "Select number of folds in Neural Network")
parser.add_argument("-lr", "--learningrt", default=0.001,type=float, help="Select number of epochs in Neural Network")
parser.add_argument("-o", "--optmizer",default='adam',type=str,help= "Select the optmizer Algorithm for Neural Network")
parser.add_argument("-l", "--loss",default='mean_squared_error',type=str,help= "Select loss type for Neural Network") # 'binary_crossentropy'
parser.add_argument("-w", "--weight",default=1,type=int,help= "apply weight in stocastic gradient descent")
parser.add_argument("-v","--verbose",default=1,type=int,help= "verbose flag")
parser.add_argument("-p","--processes", default=3,type=int, help="number of cpus for multiprocessing")
parser.add_argument("-d","--database",default='24classes',type=str,help= "specific the database for train_predict")
parser.add_argument("--dev",default=0,type=int,help= "development flag")
parser.add_argument("--ifold",default=0,type=int,help= "choose which fold to train")
parser.add_argument("--output-activation",default='tanh',type=str,help= "choose output layer activation function")
parser.add_argument("--hidden-activation",default='tanh',type=str,help= "choose hidden layer activation function")

args = parser.parse_args()

debug = 1

#args.dev=False



if args.processes == -1:
    num_processes = multiprocessing.cpu_count()
else:
    num_processes = args.processes

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print('[{0}] done in {1}'.format(name,time.time() - t0))


analysis_name = 'Classification'

data_path = os.environ['OUTPUTDATAPATH']
results_path = os.environ['PACKAGE_NAME']

dev = args.dev

#Function for Dataset

database = args.database

dt_lofar_power = LoadData(database=database,dev=args.dev)
dt_lofar_power.infoData()
all_data,all_trgt= dt_lofar_power.getData()

metrics = ['acc']#,'sp']

all_callbacks=[
  metricsAdd('sp',args.verbose),
  EarlyStoppingKeras(restore_best_weights=True, #always true if possible, for save best weights netwotk
                                          verbose=args.verbose,
                                          patience=20,
                                          min_delta=0),# o problema estava aqui!! nao sei usar esse parametro
  ModelCheckpointKeras(filepath='best_model.h5', save_best_only=True, verbose=args.verbose),
  CSVLoggerKeras('./teste.csv'),
  ReduceLROnPlateauKeras(verbose=args.verbose, patience=5, factor=0.1,min_lr=1e-10),
]


info_net = {'weight':args.weight,'n_inits':args.init,'n_neurons':args.neurons,
            'optmizerAlgorithm':args.optmizer,'n_folds':args.fold,
            'batch_size':args.batch,
            'loss':args.loss,'n_epochs':args.epochs,
            'metrics':','.join(metrics),
            'output_activation':args.output_activation,
            'hidden_activation':args.hidden_activation,
            'PCD':False,
            'database':database,
            'type_arq':'Especialista',
            'analysis_name':analysis_name,
            'callbacks':str([icall.name() for icall in all_callbacks]),
            'dev':args.dev}

bdc = BDControl(main_path=results_path,columns=info_net.keys())

results_path_specific = bdc.get_path(**info_net)

if not os.path.exists(results_path_specific):
    os.makedirs(results_path_specific)

#save configuration inside of path case something wrong happen
with open(results_path_specific+'/info_net.json', 'w') as fp:
    json.dump(info_net, fp)


from Functions.preprocessing import CrossValidation

cv = CrossValidation(X = all_data,
 					  y = all_trgt, 
 					  estimator=None,
 					  n_folds=args.fold,
 					  dev=args.dev,
 					  verbose=args.verbose,
 					  dir=results_path_specific)



train_id, test_id, folder = cv.train_test_split(ifold=args.ifold)

# In[9]:




#Preprocessing of signal
prpro = Preprocessing(limits='lin')

X_proc_train = prpro.set_transform(X=all_data[train_id],fit=True).get_transform()
X_proc_alldata = prpro.set_transform(X=all_data, fit=False).get_transform()
# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn

mlp = MLPKeras(hidden_layer_sizes=(args.neurons,),
               activation=(args.hidden_activation,args.output_activation),
               optimizer=args.optmizer,
               verbose=args.verbose,
               n_init=args.init,
               optimizer_kwargs={'lr':args.learningrt},
               loss=args.loss,
               metrics=['accuracy'],
               compile_kwargs={},
               validation_id=(None,None),
               fit_kwargs={'batch_size':args.batch, 
                           'epochs':args.epochs, 
                           'verbose':args.verbose
                           },
               callbacks_list=all_callbacks,
               dir='./')



classes_name = ['{0:02d}_specialist'.format(i) for i in range(len(np.unique(all_trgt)))]
sp = SpecialistClassification(mlp, classes_name,dir=folder,verbose=args.verbose,integrator='wta')


t0 = time.time()
list_member_paramns=[]
# train
with timer('fit Specialist each class'):
  #sp.fit(all_data,all_trgt)
  for i in range(len(np.unique(all_trgt))):
    list_member_paramns.append({
      'member':classes_name[i],
      'X':X_proc_alldata,
      'y':all_trgt,
      'train_id':train_id,
      'test_id':test_id,
      'ifold':args.ifold
      })

  def f_member_train(paramns_fit_member):
    sp.fit_only_member(**paramns_fit_member)

  if debug < 1:

    p = Pool(processes=num_processes)
    p.map(f_member_train,list_member_paramns)
    p.close()
    p.join()
  else:
    for i in list_member_paramns:
      f_member_train(i)


with timer('fit Specialist Class'):
  sp.fit(X=all_data, y=all_trgt, train_id=train_id, test_id=test_id, ifold=args.ifold)

telegram_send.send(messages=["O seu treinamento Especialista(iris) com os seguintes parametros acabou de terminar!\n"+
                             "\nFOLD: "+str(args.ifold)+
                             "\nNEURONS: "+str(args.neurons)+
                             "\nData: "+str(database)+
                             "\nBATCH: "+str(args.batch)+
                             "\nInit: "+str(args.init)+
                             "\nEle levou "+str(datetime.timedelta(seconds=time.time() - t0))+" ao total"])
# In[11]:
with timer('Predict Specialist Class'):
    sp.predict(X_proc_alldata)