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

if args.dev:
	results_path_specific = results_path + '/dev' + '/Hierarq_' + trparamns.get_params_str() + '/'
else:
	results_path_specific = results_path + '/Hierarq_' + trparamns.get_params_str() + '/'




# In[9]:


#from sklearn.neural_network import MLPClassifier
from Functions.preprocessing import CrossValidation,CVEnsemble
from Functions.ensemble import HierarqNet
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
                 monitor='sp',
                 mode='max',
                 metrics=metrics,
                 validation_id=(None,None),
                 validation_fraction=0.0,
                 dir='./')

hn = HierarqNet(mlp,n_jobs=None,n_folds=args.fold,verbose=args.verbose,dir=results_path_specific)

# In[10]:

# train
with timer('fit Specialist Class'):
    hn.fit(all_data,all_trgt)

# In[11]:

print("Start create csv for analysis")

for name, i in hn.estimators_.iteritems():
    estimador, _, y_new, ids = i
    index_ynew = np.array(range(all_trgt.shape[0]))
    index = np.concatenate([index_ynew[ids[i]]for i in range(len(ids))])


    csv_folds = []

    df_index = pd.DataFrame({'index':index})

    csv_folds.append(df_index)
    
    path = estimador.get_params()['dir'][:-7]

    

    for ifold in range(args.fold):
        file = path + 'fold0{0}/'.format(ifold) + 'hierarq_analy_fold_{0}.csv'.format(ifold)
        csv_folds.append(pd.read_csv(file))

    pd.concat(csv_folds,axis=1).to_csv(path+'hierarq_analy_{0}_estimator.csv'.format(str(name)),index=False)

print("predict HierarqNet")

hn.predict(all_data)
from sklearn.metrics import confusion_matrix
pd.DataFrame({'true':all_trgt,'predict':hn.predict(all_data)}).to_csv(results_path_specific+'true_predict.csv')
pd.DataFrame(confusion_matrix(all_trgt,hn.predict(all_data))).to_csv(results_path_specific+'confusion_matrix.csv',header=False,index=False)
