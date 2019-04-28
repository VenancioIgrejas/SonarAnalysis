import os
import sys
import time

sys.path.insert(0,'/home/venancio/WorkPlace/SonarAnalysis/Packages/Classification')




import argparse
#Argument Parse config

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-n", "--neurons", default=30,type=int, help="Select number of neurouns in Neural Network")
parser.add_argument("-e", "--epochs", default=1000,type=int, help="Select number of epochs in Neural Network")
parser.add_argument("-i", "--init",default=10,type=int, help="Select number of inicialization in Neural Network")
parser.add_argument("-b", "--batch",default=32,type=int, help="Select number of inicialization in Neural Network")
parser.add_argument("-f", "--fold",default=10,type=int, help= "Select number of folds in Neural Network")
parser.add_argument("-o", "--optmizer",default='adam',type=str,help= "Select the optmizer Algorithm for Neural Network")
parser.add_argument("-l", "--loss",default='mean_squared_error',type=str,help= "Select loss type for Neural Network") # 'binary_crossentropy'
parser.add_argument("-s", "--specialist",default=0,type=int,help= "number of specialist for training")
parser.add_argument("-w", "--weight",default=True,type=bool,help= "apply weight in stocastic gradient descent")
parser.add_argument("-v","--verbose",default=True,type=bool,help= "verbose flag")
parser.add_argument("--dev",default=True,type=bool,help= "development flag")
parser.add_argument("--ifold",default=0,type=int,help= "choose which fold to train")
parser.add_argument("--output-activation",default='tanh',type=str,help= "choose output layer activation function")
parser.add_argument("--hidden-activation",default='tanh',type=str,help= "choose hidden layer activation function")

args = parser.parse_args()

from classificationConfig import CONFIG
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from sklearn import metrics

import multiprocessing

from NeuralNetworkAnalysis import NeuralNetworkAnalysis as nnm

from OneSpecialistClass import OneSpecialistClass as osc
from Functions import TrainParameters

num_processes = multiprocessing.cpu_count()


analysis_name = 'Classification'

data_path = CONFIG['OUTPUTDATAPATH']
results_path = CONFIG['PACKAGE_NAME']

dev = args.dev

#Function for Dataset
from Functions.dataset.shipClasses import LoadData
import numpy as np

dt_24 = LoadData()
dt_24.infoData()
all_data,all_trgt= dt_24.getData()

results_path_specific = results_path + '/Esp_24classes_40nn_tanh2_adam_mse_init5_batNone_epo1000_weightTrue/'

trparamns = TrainParameters.SpecialistClassificationTrnParams(weight=args.weight,n_inits=args.init,n_neurons=args.neurons,
                                                              optmizerAlgorithm=args.optmizer,n_folds=args.fold,batch_size=args.batch,
                                                              loss=args.loss,n_epochs=args.epochs,verbose=args.verbose,
                                                              output_activation=args.output_activation,
                                                              hidden_activation=args.hidden_activation,)

if args.dev:
	results_path_specific = '/dev' + '/Spec_24class_' + trparamns.get_params_str()
else:
	results_path_specific = '/Spec_24class_' + trparamns.get_params_str()


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
from Functions.preprocessing import CrossValidation,CVEnsemble
from Functions.ensemble import SpecialistClass
from Functions.mlpClassification import MLPKeras
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split,StratifiedKFold

# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

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
                 metrics=['acc'],
                 validation_id=(train_id,test_id),
                 validation_fraction=0.0,
                 dir='./')

sp = SpecialistClass(mlp,dir=folder,verbose=args.verbose)

# In[10]:

# train
with timer('fit Specialist Class'):
    sp.fit(all_data,all_trgt)

# In[11]:


from sklearn.metrics import confusion_matrix
pd.DataFrame(confusion_matrix(all_trgt[test_id],sp.predict(all_data[test_id]))).to_csv(folder+'confusion_matrix.csv',header=False,index=False)

#---------------------------------------------------------------------------------
#DataModule

analysis = nnm(analysis_name="NeuralNetwork",development_flag=True,development_events=400,verbose=False)
#analysis.infoData()
#analysis.setRangeClass([i-1 for i in class_interest])
#analysis.balanceData()
#analysis.infoData()
weight_class = analysis.getClassesWeight(flag=True)
all_data,all_trgt,_ = analysis.getData()
X_train,X_test,y_train,y_test = analysis.train_test_split()

if dev:
    results_path = results_path+'/dev/{0}_data_{1}_classes'.format(X_train.shape[0],len(analysis.class_labels))

trparamns = TrainParameters.SpecialistClassificationTrnParams(weight=args.weight,n_inits=args.init,n_neurons=args.neurons,
                                                              optmizerAlgorithm=args.optmizer,n_folds=args.fold,batch_size=args.batch,
                                                              loss=args.loss,n_epochs=args.epochs,verbose=True,
                                                              output_activation=args.output_activation,
                                                              hidden_activation=args.hidden_activation)
#,output_activation='sigmoid',hidden_activation='relu')

spc = osc(results_path,trparamns,int(args.specialist))
print spc.result_path_fold[0]
cvo = spc.set_cross_validation(all_trgt=y_train)
models = spc.train_n_folds(X_train,y_train)
