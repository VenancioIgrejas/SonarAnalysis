import numpy as np


from keras.callbacks import Callback

from keras import backend as K
from keras.utils import to_categorical, get_custom_objects
from sklearn.metrics import recall_score

def set_early_stopping():
    raise NotImplementedError

def set_model_checkpoint():
    raise NotImplementedError


class MyHistory(Callback):
    def on_train_begin(self,logs={}):
        print 'start training'


    def on_batch_end(self,batch,logs={}):
        print 'batch({0}): loss -> {1}'.format(batch,logs['loss'])


class metricsAdd(Callback):
    def __init__(self, monitor, verbose=0,verbose_train=1):
        super(Callback,self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.verbose_train = verbose_train
        self.monitor_train = []
    
    def on_epoch_end(self, batch, logs={}):
        
        if not self.monitor in ['sp']:
            return
        
        X_val, y_val = self.validation_data[0], self.validation_data[1]
        y_predict = np.asarray(self.model.predict(X_val))
        
        
        
        if self.monitor == 'sp':
            num_classes = y_val.shape[1]
        
            y_val = np.argmax(y_val, axis=1)
            y_predict = np.argmax(y_predict, axis=1)

            average = None

            if num_classes==2:
        		average = 'binary'        		

            recall = recall_score(y_val, y_predict, average=average)
            sp = np.sqrt(np.sum(recall) / num_classes *
                         np.power(np.prod(recall), 1.0 / float(num_classes)))
        
            logs['sp'] = sp
            
            self.monitor_train.append(sp)
        
            monitor_value = logs.get('sp')
            if self.verbose > 0:
                print("\n Monitor - epoch: %d - val score(%s): %.6f" % (batch+1,self.monitor, monitor_value))
        
        return
    
    def on_train_end(self, logs=None):
        best_epoch = np.argmax(self.monitor_train)
        best_value = self.monitor_train[best_epoch]
        if self.verbose_train > 0:
            print("\n[+]End Train - best epoch: %d - best val score(%s): %.6f" % (best_epoch+1,self.monitor, best_value))
        
        return
