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
parser.add_argument("-d","--database",default='24classes',type=str,help= "specific the database for train_predict")
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
	results_path_specific = results_path + '/dev' + '/PCD_Spec_{0}_'.format(database) + trparamns.get_params_str()
else:
	results_path_specific = results_path + '/PCD_Spec_{0}_'.format(database) + trparamns.get_params_str()



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
                     validation_id=(None,None))

classes_name = ['{0:02d}_specialist'.format(i) for i in range(len(np.unique(all_trgt)))]

#sp = SpecialistClass(mlp,dir=folder,verbose=args.verbose)
sp = SpecialistClassification(pcd, classes_name,dir=folder,verbose=args.verbose,integrator='wta')

# train
list_member_paramns=[]

# train
with timer('fit Specialist each class'):
  #sp.fit(all_data,all_trgt)
  for i in range(len(np.unique(all_trgt))):
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
    sp.fit_only_member(**paramns_fit_member)
  #sp.fit(all_data,all_trgt)
  p = Pool(processes=num_processes)
  p.map(f_member_train,list_member_paramns)
  p.close()
  p.join()

with timer('fit Specialist Class'):
  sp.fit(X=all_data, y=all_trgt, train_id=train_id, test_id=test_id, ifold=args.ifold)

telegram_send.send(messages=["O seu treinamento Especialista com PCDS com os seguintes parametros acabou de terminar!",
                             "FOLD: "+str(args.ifold),
                             "Data: "+str(args.database),
                             "PCD: "+str(args.pcd),
                             "BATCH: "+str(args.batch),
                             "Init: "+str(args.init),
                             "Ele levou "+str(datetime.timedelta(seconds=time.time() - t0))+" ao total"])


# In[11]:
with timer('Predict Specialist Class'):
    sp.predict(all_data)