import os
import sys
import time
import datetime
import telegram_send
import json 

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
parser.add_argument("-d","--database",default='24classes',type=str,help= "specific the database for train_predict")
parser.add_argument("--dev",default=False,type=bool,help= "development flag")
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
from Functions.dataset.path import BDControl

from Functions.mlpClassification import MLPKeras
from Functions.preprocessing import CrossValidation,CVEnsemble
from Functions.ensemble import SpecialistClass,HierarqNet,HierarqClassification
from Functions.ensemble.hierarque_classification import SpecialistClassification

import sklearn as sk

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

# trparamns = TrainParameters.SpecialistClassificationTrnParams(weight=args.weight,n_inits=args.init,n_neurons=args.neurons,
#                                                               optmizerAlgorithm=args.optmizer,n_folds=args.fold,batch_size=args.batch,
#                                                               loss=args.loss,n_epochs=args.epochs,verbose=args.verbose,metrics=metrics,
#                                                               output_activation=args.output_activation,
#                                                               hidden_activation=args.hidden_activation,)

str_spec ='_'
spec_estimators=None

# if False:#database=='31classes':
#   spec_estimators = ['class_B8']
#   str_spec ='_'
#   for spec in spec_estimators:
#       str_spec = str_spec + spec.split('_')[-1] + '_'

# if args.dev:
#     results_path_specific = results_path + '/dev' + '/Hierarq{0}{1}_'.format(str_spec,database) + trparamns.get_params_str()
# else:
#     results_path_specific = results_path + '/Hierarq{0}{1}_'.format(str_spec,database) + trparamns.get_params_str()

info_net = {'weight':args.weight,'n_inits':args.init,'n_neurons':args.neurons,
            'optmizerAlgorithm':args.optmizer,'n_folds':args.fold,
            'batch_size':args.batch,
            'loss':args.loss,'n_epochs':args.epochs,
            'metrics':','.join(metrics),
            'output_activation':args.output_activation,
            'hidden_activation':args.hidden_activation,
            'PCD':False,
            'database':database,
            'type_arq':'MLP',
            'analysis_name':analysis_name,
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


def train_pool(ifold):
    train_id, test_id, folder = cv.train_test_split(ifold=ifold)

    #only for control of Net
    csv_master = folder+'/master_table.csv'
    if not os.path.exists(csv_master):
        fold_vect = np.zeros(shape=all_trgt.shape,dtype=int)
        fold_vect[test_id] = 1

        pd.DataFrame({'target':all_trgt,'fold_{0}'.format(ifold):fold_vect}).to_csv(csv_master,index=False)
    



    # In[9]:



    mlp = MLPKeras(hidden_layer_sizes=(args.neurons,),
                     activation=(args.hidden_activation,args.output_activation),
                     optimize=args.optmizer,
                     loss=args.loss,
                     n_init=args.init,
                     batch_size=args.batch,
                     epoch=args.epochs,
                     shuffle=True,
                     random_state=None,
                     verbose=args.verbose,
                     train_log=True,
                     early_stopping = True,
                     patience=20,
                     save_best_model = True,
                     class_weight = args.weight,
                     monitor='sp',
                     mode='max',
                     metrics=metrics,
                     validation_id=(train_id,test_id),
                     validation_fraction=0.0,
                     dir=folder)


    # train
    t0 = time.time()
    # list_member_paramns = []
    # with timer('fit Hierarque each class'):

    #   for i in range(n_member):
    #     list_member_paramns.append({
    #       'member':classes_name[i],
    #       'X':all_data,
    #       'y':all_trgt,
    #       'train_id':train_id,
    #       'test_id':test_id,
    #       'ifold':args.ifold
    #       })

    #   def f_member_train(paramns_fit_member):
    #     hn.fit_only_member(**paramns_fit_member)

    #   p = Pool(processes=num_processes)
    #   p.map(f_member_train,list_member_paramns)
    #   p.close()
    #   p.join()

    with timer('fit MLP'):
        #hn.fit(X=all_data, y=all_trgt, train_id=train_id, test_id=test_id, ifold=args.ifold)
        mlp.fit(X=all_data, y=all_trgt)
    telegram_send.send(messages=["O seu treinamento MLP simples com os seguintes parametros acabou de terminar!\n"+
                                 "\nFOLD: "+str(ifold)+
                                 "\nNEURONS: "+str(args.neurons)+
                                 "\nData: "+str(database)+
                                 "\nBATCH: "+str(args.batch)+
                                 "\nInit: "+str(args.init)+
                                 "\nEle levou "+str(datetime.timedelta(seconds=time.time() - t0))+" ao total"])

    # In[11]:
    with timer('Predict MLP'):
        pred = mlp.predict(X=all_data,predict='sparce')
        pd.DataFrame(pred,columns=['neuron_{0}'.format(i) for i in range(pred.shape[1])]).to_csv(folder+'/predict.csv',index=False)

with timer('fit all folds of MLP'):

      p = Pool(processes=num_processes)
      p.map(train_pool,range(args.fold))
      p.close()
      p.join()