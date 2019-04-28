import os
import sys
import time

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
parser.add_argument("--dev",default=False,type=bool,help= "development flag")
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

import multiprocessing

from Functions import TrainParameters

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
  
dt_24 = LoadData(dev=args.dev)
dt_24.infoData()
all_data,all_trgt= dt_24.getData()

metrics = ['acc','sp']

trparamns = TrainParameters.SpecialistClassificationTrnParams(weight=args.weight,n_inits=args.init,n_neurons=args.neurons,
                                                              optmizerAlgorithm=args.optmizer,n_folds=args.fold,batch_size=args.batch,
                                                              loss=args.loss,n_epochs=args.epochs,verbose=args.verbose,metrics=metrics,
                                                              output_activation=args.output_activation,
                                                              hidden_activation=args.hidden_activation,)



spec_estimators = ['class_S','class_A','class_AA']
str_spec ='_'
for spec in spec_estimators:
    str_spec = str_spec + spec.split('_')[-1] + '_'


if args.dev:
	results_path_specific = results_path + '/dev' + '/Hierarq'+str_spec+'withFolds_' + trparamns.get_params_str()
else:
	results_path_specific = results_path + '/Hierarq_spec'+str_spec+'withFolds_' + trparamns.get_params_str()



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
from Functions.ensemble.hierarque_classification import SpecialistClassification

import sklearn as sk

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
                 validation_id=(None,None),
                 validation_fraction=0.0,
                 dir='./')


classes_name = ['{0:02d}_specialist'.format(i) for i in range(len(np.unique(all_trgt)))]

#sp = SpecialistClass(mlp,dir=folder,verbose=args.verbose)
sp = SpecialistClassification(mlp, classes_name,dir='./',verbose=args.verbose,integrator='wta')

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

dict_estimators = dict([(key,mlp) for key in map_lvl.keys()])
dict_estimators.update(dict([(estimator,sp) for estimator in spec_estimators]))

hn = HierarqClassification(estimator=dict_estimators,
                           dict_classes=classes,
                           map_members=map_lvl,
                           n_lvl=3,
                           dir=folder,
                           verbose=args.verbose)

# In[10]:

# train
with timer('fit Hierarque Class'):
    hn.fit(X=all_data, y=all_trgt, train_id=train_id, test_id=test_id, ifold=args.ifold)

# In[11]:
with timer('Predict Hierarque Class'):
    hn.predict(all_data)