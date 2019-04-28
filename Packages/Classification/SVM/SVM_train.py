
# coding: utf-8

# In[1]:


import os
import sys
import time
import datetime
import telegram_send
import json 

import numpy as np
import pandas as pd

from Functions.preprocessing import CrossValidation
from Functions.dataset.shipClasses import LoadData


from sklearn import svm
from sklearn.pipeline import Pipeline


from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

from lps_toolbox.metrics.classification import sp_index


from contextlib import contextmanager

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print('[{0}] done in {1}'.format(name,time.time() - t0))

analysis_name = 'Classification'

data_path = os.environ['OUTPUTDATAPATH']
results_path = os.environ['PACKAGE_NAME']

dev = False

database = '24classes'


# In[2]:


#Function for Dataset

dt_lofar_power = LoadData(database=database,dev=dev,dev_size=1000)
dt_lofar_power.infoData()
all_data,all_trgt= dt_lofar_power.getData()


# In[3]:


ifold = 0


cv = CrossValidation(X = all_data,
            y = all_trgt, 
            estimator=None,
            n_folds=10,
            dev=True,
            verbose=True,
            dir='./')

train_id, test_id, folder = cv.train_test_split(ifold=ifold)


# In[15]:


pipe = Pipeline([
    ('scaler',StandardScaler()),
    ('svm_clf',svm.SVC(verbose=True,class_weight='balanced'))
])

with timer('fit SVM'):
	pipe.fit(X=all_data[train_id],y=all_trgt[train_id])


# In[16]:


pred = pipe.predict(all_data[test_id])
cm = confusion_matrix(y_pred=pred,y_true=all_trgt[test_id])
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 


# In[17]:


# fig = plt.figure(figsize=(25,25))
# ax = fig.add_subplot(111)
# ax.set_aspect(1)

# sns.heatmap(cm_norm,annot=True,ax=ax,cmap=plt.cm.Greys)


# In[23]:


print('SP of fold{0} : {0:0.2f}'.format(ifold+1,sp_index(y_pred=pred,y_true=all_trgt[test_id])))

