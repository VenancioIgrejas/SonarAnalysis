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
parser.add_argument("--dev",default=False,type=bool,help= "development flag")
parser.add_argument("--pcd",default=10,type=int, help="Select number of inicialization in Neural Network")
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

if args.dev:
	results_path_specific = results_path + '/dev' + '/PCD_Boosting_' + trparamns.get_params_str()
else:
	results_path_specific = results_path + '/PCD_Boosting_' + trparamns.get_params_str()



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


#from sklearn.neural_network import MLPClassifier
from Functions.ensemble.hierarque_classification import SpecialistClassification
from sklearn.model_selection import train_test_split,StratifiedKFold
from Functions.ensemble import AdaBoost
from Functions.principal_components import PCDCooperative, TrnParams

# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn
from keras import backend as K

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
                     validation_id=(train_id,test_id))

#sp = SpecialistClass(mlp,dir=folder,verbose=args.verbose)
boost = AdaBoost(base_estimator=pcd,
                 n_estimators=20,
                 learning_rate=1.0,
                 algorithm='SAMME',
                 random_state=None,
                 dir=folder)

# train
t0 = time.time()
with timer('fit Specialist Class'):
  #sp.fit(all_data,all_trgt)
  boost.fit(X=all_data, y=all_trgt)

telegram_send.send(messages=["O seu treinamento AdaBoosting  com PCDS com os seguintes parametros acabou de terminar!",
                             "FOLD: "+str(args.ifold),
                             "PCD: "+str(args.pcd),
                             "BATCH: "+str(args.batch),
                             "Init: "+str(args.init),
                             "Ele levou "+str(datetime.timedelta(seconds=time.time() - t0))+" ao total"])


# In[11]:
with timer('Predict Specialist Class'):
    sp.predict(all_data)