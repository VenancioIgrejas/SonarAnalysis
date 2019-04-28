
# coding: utf-8

# In[1]:


import os
import sys
import time
import datetime
import telegram_send
import json 
from joblib import dump, load


import numpy as np
import pandas as pd

from Functions.preprocessing import CrossValidation
from Functions.dataset.shipClasses import LoadData
from Functions.dataset.path import BDControl


from sklearn import svm
from sklearn.pipeline import Pipeline


from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import confusion_matrix

from multiprocessing import Pool, TimeoutError
from lps_toolbox.metrics.classification import sp_index


from contextlib import contextmanager

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print('[{0}] done in {1}'.format(name,time.time() - t0))


import argparse
#Argument Parse config

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-f", "--fold",default=10,type=int, help= "Select number of folds in Neural Network")
parser.add_argument("-c", "--penalty", default=1,type=int, help="Select number of neurouns in Neural Network")
parser.add_argument("-k", "--kernel", default='rbf',type=str, help="Select number of neurouns in Neural Network")
parser.add_argument("-g", "--gamma", default='auto',type=str, help="Select number of neurouns in Neural Network")
parser.add_argument("-t", "--tol", default=1e-3,type=float, help="Select number of neurouns in Neural Network")
parser.add_argument("--cache", default=200,type=int, help="Select number of neurouns in Neural Network")
parser.add_argument("-w", "--weight", default='balanced',type=str, help="Select number of neurouns in Neural Network")
parser.add_argument("-m", "--maxiter", default=-1,type=int, help="Select number of neurouns in Neural Network")

parser.add_argument("-s", "--decisionFunctionShape", default='ovr',type=str, help="Select number of neurouns in Neural Network")

parser.add_argument("--dev",default=False,type=bool,help= "development flag")
parser.add_argument("--devSize",default=1000,type=int,help= "development flag")
parser.add_argument("-d","--database",default='24classes',type=str,help= "specific the database for train_predict")
parser.add_argument("-p","--processes", default=3,type=int, help="number of cpus for multiprocessing")


args = parser.parse_args()


if args.processes == -1:
    num_processes = multiprocessing.cpu_count()
else:
    num_processes = args.processes


if args.gamma=='auto':
    gamma=args.gamma
else:
    gamma=float(args.gamma)

params_SVC = {
            'C':args.penalty,
            'kernel':args.kernel,
            'gamma':gamma,
            'tol':args.tol,
            'cache_size':args.cache,
            'class_weight':args.weight,
            'max_iter':args.maxiter,
            'decision_function_shape':args.decisionFunctionShape
}

analysis_name = 'Classification'

data_path = os.environ['OUTPUTDATAPATH']
results_path = os.environ['PACKAGE_NAME']

args.dev=False

# In[2]:


#Function for Dataset

dt_lofar_power = LoadData(database=args.database,dev=args.dev)#,dev_size=args.devSize)
dt_lofar_power.infoData()
all_data,all_trgt= dt_lofar_power.getData()


info_net = {'n_folds':args.fold,
            'C':args.penalty,
            'kernel':args.kernel,
            'gamma':gamma,
            'tol':args.tol,
            'cache_size':args.cache,
            'class_weight':args.weight,
            'max_iter':args.maxiter,
            'decision_function_shape':args.decisionFunctionShape,
            'database':args.database,
            'type_arq':'SVC',
            'analysis_name':analysis_name,
            'dev':args.dev}

bdc = BDControl(main_path=results_path,columns=info_net.keys(),bd_file='bd_SVM.csv')

results_path_specific = bdc.get_path(**info_net)

if not os.path.exists(results_path_specific):
    os.makedirs(results_path_specific)

#save configuration inside of path case something wrong happen
with open(results_path_specific+'/info_net.json', 'w') as fp:
    json.dump(info_net, fp)



# In[3]:

cv = CrossValidation(X = all_data,
            y = all_trgt, 
            estimator=None,
            n_folds=args.fold,
            dev=args.dev,
            verbose=True,
            dir=results_path_specific)




# In[15]:

def train_SVM(ifold):

    train_id, test_id, folder = cv.train_test_split(ifold=ifold)



    pipe = Pipeline([
        ('scaler',StandardScaler()),
        ('svm_clf',svm.SVC(verbose=True,**params_SVC))
    ])

    t0 = time.time()
    with timer('fit SVM'):
    	pipe.fit(X=all_data[train_id],y=all_trgt[train_id])



    pred = pipe.predict(all_data)

    telegram_send.send(messages=["O seu treinamento SVM(SVC) com os seguintes parametros acabou de terminar!\n"+
                                 "\nFOLD: "+str(ifold)+
                                 "\nData: "+str(args.database)+
                                 "\nEle levou "+str(datetime.timedelta(seconds=time.time() - t0))+" ao total"])


    s = dump(pipe,folder+'/save_model.jbl',compress=9)



    #only for control of Net
    csv_master = folder+'/master_table.csv'
    if not os.path.exists(csv_master):
        fold_vect = np.zeros(shape=all_trgt.shape,dtype=int)
        fold_vect[test_id] = 1

        pd.DataFrame({'target':all_trgt,'fold_0{0}'.format(ifold):fold_vect,'predict':pred}).to_csv(csv_master,index=False)


p = Pool(processes=num_processes)
p.map(train_SVM,range(args.fold))
p.close()
p.join()
