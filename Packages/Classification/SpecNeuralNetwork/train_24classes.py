import os
import sys
import time
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
parser.add_argument("--dev",default=False,type=bool,help= "development flag")
parser.add_argument("--ifold",default=0,type=int,help= "choose which fold to train")
parser.add_argument("--n_member",default=24,type=int,help= "choose which specialist will be train")
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
	results_path_specific = results_path + '/dev' + '/Spec_24class_' + trparamns.get_params_str()
else:
	results_path_specific = results_path + '/Spec_24class_' + trparamns.get_params_str()



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
from Functions.ensemble.hierarque_classification import SpecialistClassification
from Functions.mlpClassification import MLPKeras
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split,StratifiedKFold

# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn
from keras import backend as K
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
                 monitor='val_loss',
                 mode='auto',
                 metrics=metrics,
                 validation_id=(None,None),
                 validation_fraction=0.0,
                 dir='./')

classes_name = ['{0:02d}_specialist'.format(i) for i in range(len(np.unique(all_trgt)))]

#sp = SpecialistClass(mlp,dir=folder,verbose=args.verbose)
sp = SpecialistClassification(mlp, classes_name,dir=folder,verbose=args.verbose,integrator='wta')

# train
with timer('fit Specialist Class'):
  #sp.fit(all_data,all_trgt)
  for i in range(args.n_member):
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

  sp.fit(X=all_data, y=all_trgt, train_id=train_id, test_id=test_id, ifold=args.ifold)

# In[11]:
with timer('Predict Specialist Class'):
    sp.predict(all_data)





# fold = -np.ones((all_data.shape[0],), dtype=int)

# fold[train_id] = 0
# fold[test_id] = 1
# # In[11]:

# pred_all = sp.predict(all_data)

# output_sparce = sp._output_pred
# matrix_pred = sp._matrix_pred
# spec_output_sparce = sp._output_sparse_each_spec

# pred = pred_all[test_id]
# values = np.array([row[np.argmax(row)] for row in output_sparce])



# output_value_matrix = np.vstack(spec_output_sparce).T
# #sys.exit()
# #tmp = []
# #for ispec,output_spc in enumerate(spec_output_sparce):
# #    tmp.append(pd.DataFrame(np.array(map(lambda x:x[np.argmax(x)],output_spc))))

# #output_value_matrix = pd.concat(tmp,axis=1).values

# #generate csv files as analisys
# from sklearn.metrics import confusion_matrix
# pd.DataFrame(confusion_matrix(all_trgt[test_id],pred)).to_csv(folder+'confusion_matrix.csv',header=False,index=False)
# pd.DataFrame({'ClassEspe(resultados)_{0}'.format(args.ifold):pred_all,
#               'ClassEspe(valor)_{0}'.format(args.ifold):values,
#               'ClassEspe_WTA(resultados)_{0}'.format(args.ifold):np.argmax(output_value_matrix,axis=1),
#               'ClassEspe_WTA(valor)_{0}'.format(args.ifold):np.array(map(lambda x:x[np.argmax(x)],output_value_matrix)),
#               'fold_{0}'.format(args.ifold):fold}).to_csv(folder+'specialist_analy_trainIDWeights_fold_{0}.csv'.format(args.ifold),index=False)
# pd.DataFrame(dict([('weight_ispec{0:02d}_{1}'.format(ispec,args.ifold),vector_weight) for ispec,vector_weight in sp.weight_integrator.iteritems()])).to_csv(folder+'weights_fold_{0}.csv'.format(args.ifold),index=False)
# pd.DataFrame(output_sparce).to_csv(folder+'output_sparce.csv',header=False,index=False)
# pd.DataFrame(matrix_pred).to_csv(folder+'matrix_sparce.csv',header=False,index=False)

K.clear_session()


