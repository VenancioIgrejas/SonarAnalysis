import os
import sys
import time
import datetime
import telegram_send


sys.path.insert(0,'/home/venancio/Workspace/SonarAnalysis/Packages/Classification')




import argparse
#Argument Parse config

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-n", "--neurons", default=10,type=int, help="Select number of neurouns in Neural Network")
parser.add_argument("-e", "--epochs", default=1000,type=int, help="Select number of epochs in Neural Network")
parser.add_argument("-i", "--init",default=10,type=int, help="Select number of inicialization in Neural Network")
parser.add_argument("-b", "--batch",default=32,type=int, help="Select number of inicialization in Neural Network")
parser.add_argument("-f", "--fold",default=10,type=int, help= "Select number of folds in Neural Network")
parser.add_argument("-o", "--optmizer",default='adam',type=str,help= "Select the optmizer Algorithm for Neural Network")
parser.add_argument("-l", "--loss",default='mean_squared_error',type=str,help= "Select loss type for Neural Network") # 'binary_crossentropy'
parser.add_argument("-w", "--weight",default=True,type=bool,help= "apply weight in stocastic gradient descent")
parser.add_argument("-v","--verbose",default=True,type=bool,help= "verbose flag")
parser.add_argument("-p","--processes", default=3,type=int, help="number of cpus for multiprocessing")
parser.add_argument("--pcd",default=10,type=int, help="Select number of inicialization in Neural Network")
parser.add_argument("--dev",default=False,type=bool,help= "development flag")
parser.add_argument("-d","--database",default='24classes',type=str,help= "specific the database for train_predict")
parser.add_argument("--ifold",default=0,type=int,help= "choose which fold to train")
parser.add_argument("--output-activation",default='tanh',type=str,help= "choose output layer activation function")
parser.add_argument("--hidden-activation",default='tanh',type=str,help= "choose hidden layer activation function")

args = parser.parse_args()

args.dev=False


from classificationConfig import CONFIG
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from sklearn import metrics

from contextlib import contextmanager

from multiprocessing import Pool, TimeoutError


from Functions import TrainParameters

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

data_path = CONFIG['OUTPUTDATAPATH']
results_path = CONFIG['PACKAGE_NAME']

dev = args.dev

#Function for Dataset
from Functions.dataset.shipClasses import LoadData
import numpy as np

database = args.database

dt_lofar_power = LoadData(database=database,dev=args.dev)
dt_lofar_power.infoData()
all_data,all_trgt= dt_lofar_power.getData()

metrics = ['acc','sp']

trparamns = TrainParameters.SpecialistClassificationTrnParams(weight=args.weight,n_inits=args.init,n_neurons=args.neurons,
                                                              optmizerAlgorithm=args.optmizer,n_folds=args.fold,batch_size=args.batch,
                                                              loss=args.loss,n_epochs=args.epochs,verbose=args.verbose,metrics=metrics,
                                                              output_activation=args.output_activation,
                                                              hidden_activation=args.hidden_activation,)

if args.dev:
	results_path_specific = results_path + '/dev' + '/PCD_Hierarq_{0}_'.format(database) + trparamns.get_params_str()
else:
	results_path_specific = results_path + '/PCD_Hierarq_{0}_'.format(database) + trparamns.get_params_str()



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

from Functions.mlpClassification import MLPKeras
from Functions.preprocessing import CrossValidation,CVEnsemble
from Functions.ensemble import SpecialistClass,HierarqNet,HierarqClassification
from Functions.principal_components import PCDCooperative, TrnParams

import sklearn as sk

if database=='31classes':
    n_member = 6
    classes={'class_S':[[1,2,3,4],
                        [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]],

            'class_A':[[1],
                       [2],
                       [3],
                       [4]],

            'class_B':[[5],
                       [6],
                       [7,8,9,10,11],
                       [12],
                       [13,14,15],
                       [16],
                       [17],
                       [18,19,20,21,22,23,24,25,26,27,28,29,30]],
             
            'class_B3':[[7],
                        [8],
                        [9],
                        [10],
                        [11]],

            'class_B5':[[13],
                        [14],
                        [15]],
            
            'class_B8':[[18],
                        [19],
                        [20],
                        [21],
                        [22],
                        [23],
                        [24],
                        [25],
                        [26],
                        [27],
                        [28],
                        [29],
                        [30]],}

    map_lvl = {
        'class_S':0,
        'class_A':1,
        'class_B':1,
        'class_B3':2,
        'class_B5':2,
        'class_B8':2
    }

else:
    n_member = 9
    classes={'class_S':[[9,10,13,14,16,23,1,2,22,21],
                                 [4,6,8,12,17,19],
                                 [11,24],
                                 [5,7,15,3,18,20]],
            'class_A':[[9,10,13,14,16],
                            [23,1,2,22],
                            [21]],
            'class_B':[[4],
                            [6],
                            [8],
                            [12],
                            [17],
                            [19]],
            'class_D':[[5,7,15],
                    [3,18,20]],
             
            'class_C':[[11],
                        [24]],
            'class_AA':[[9],
                             [10],
                             [13],
                             [14],
                             [16]],
            'class_AB':[[23],
                             [1],
                             [2],
                             [22]],
            'class_DA':[[5],
                             [7],
                             [15]],
            'class_DB':[[3],
                             [18],
                             [20]]}

    map_lvl = {
        'class_S':0,
        'class_A':1,
        'class_AA':2,
        'class_AB':2,
        'class_AC':2,
        'class_B':1,
        'class_C':1,
        'class_D':1,
        'class_DA':2,
        'class_DB':2
    }

trn = TrnParams(learning_rate=0.001,
                monitor='val_loss',
                verbose=args.verbose,
                train_verbose=args.verbose,
                batch_size=args.batch,
                n_inits=args.init,
                n_epochs=args.epochs)

pcd = PCDCooperative(n_components=args.pcd,
                     is_save=True,
                     trn_params=trn,
                     validation_id=(None,None))

hn = HierarqClassification(estimator=pcd,
                           dict_classes=classes,
                           map_members=map_lvl,
                           verbose=args.verbose,
                           n_lvl=3,
                           dir=folder)

# In[10]:

classes_name = classes.keys()

# train
t0 = time.time()

list_member_paramns = []
with timer('fit Hierarque each class'):

  for i in range(n_member):
    list_member_paramns.append({
      'member':classes_name[i],
      'X':all_data,
      'y':all_trgt,
      'train_id':train_id,
      'test_id':test_id,
      'ifold':args.ifold
      })

  def f_member_train(paramns_fit_member):
    #sp.fit_only_member(member=classes_name[args.member_specialist], X=all_data, y=all_trgt, train_id=train_id, test_id=test_id, ifold=args.ifold)
    hn.fit_only_member(**paramns_fit_member)
  #sp.fit(all_data,all_trgt)
  p = Pool(processes=num_processes)
  p.map(f_member_train,list_member_paramns)
  p.close()
  p.join()
  #for i in list_member_paramns:
  #  hn.fit_only_member(**i)

with timer('fit Hierarque Class'):
    hn.fit(X=all_data, y=all_trgt, train_id=train_id, test_id=test_id, ifold=args.ifold)

telegram_send.send(messages=["O seu treinamento Hierarquico com PCDS com os seguintes parametros acabou de terminar!",
                             "FOLD: "+str(args.ifold),
                             "PCD: "+str(args.pcd),
                             "BATCH: "+str(args.batch),
                             "Init: "+str(args.init),
                             "Ele levou "+str(datetime.timedelta(seconds=time.time() - t0))+" ao total"])

# In[11]:
with timer('Predict Hierarque Class'):
    hn.predict(all_data)