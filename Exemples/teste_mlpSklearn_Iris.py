#import module
import sys
import numpy as np
import time
from contextlib import contextmanager

sys.path.insert(0, '/home/venancio/jupy-note')

from sklearn.datasets import load_iris

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from Functions.mlpClassification import MLPSKlearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder


################## auxiliar functions ###############################
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print('[{0}] done in {1} s'.format(name,time.time() - t0))
    print('\n')
#####################################################################

#load data and use Pandas
with timer('load data'):
    iris = load_iris()
    df_data = pd.DataFrame(iris.data,columns=iris.feature_names)
    df_trgt = pd.DataFrame(iris.target,columns=['target'])
    df = pd.concat([df_data,df_trgt],axis=1)
    print "iris: target->{0} \n feactures->{1}\n".format(iris.target,iris.feature_names)
    print df.head()

#prepare data (no reason)
with timer('prepare data'):
    X_train, X_test, y_train, y_test = train_test_split(df_data.values,df_trgt.values,test_size=0.2)

    # scaler = StandardScaler().fit(X_train)
    # preproc_data_train = scaler.transform(X_train)
    # preproc_data_teste = scaler.transform(X_test)
    #
    # OHT = OneHotEncoder(sparse=False)
    # sparce_trgt = OHT.fit_transform(y_train)

# #train
# with timer('train data'):
#     mlp = MLPSKlearn()
#     mlp.fit(X_train,y_train)
#
# #predict
# with timer('predict data'):
#     pred = mlp.predict(X_test)
#
#     print pred
#
#Adaboosting
mlp = MLPSKlearn()
boost = AdaBoostClassifier(base_estimator=mlp,
                           n_estimators=10,
                           learning_rate=1,
                           algorithm= 'SAMME',
                           random_state=None)

#boost.fit(X_train,y_train)
mlp.fit(X_train,y_train)

#print classification_report(output,y_test)
