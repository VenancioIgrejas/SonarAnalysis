import numpy as np
import warnings

from keras.callbacks import EarlyStopping, Callback

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
    
    # def on_train_end(self, logs=None):
    #     best_epoch = np.argmax(self.monitor_train)
    #     best_value = self.monitor_train[best_epoch]
    #     if self.verbose_train > 0:
    #         print("\n[+]End Train - best epoch: %d - best val score(%s): %.6f" % (best_epoch+1,self.monitor, best_value))
        
    #     return

class StopTraining(EarlyStopping):
    def __init__(self, monitor='val_loss', second_monitor='sp',min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False):
        super(StopTraining,self).__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode, baseline=baseline, restore_best_weights=restore_best_weights)
        self.s_monitor = second_monitor
        self.monitor_op_s = np.greater
        self.monitor_op_loss = np.less
        self.best_epoch = 0
        self.min_delta_s = self.min_delta

        if self.monitor_op == np.greater:
            self.min_delta_s *= 1
        else:
            self.min_delta_s *= -1
        
    
    def on_train_begin(self,logs=None):
        super(StopTraining,self).on_train_begin(logs)
        self.best_s = -np.Inf
        
    
    def on_epoch_end(self, epoch, logs=None):

        logs['best_epoch'] = 0

        current = self.get_monitor_value(logs,self.monitor)
        if current is None:
            return
        
        current_second = self.get_monitor_value(logs, self.s_monitor)
        if current is None:
            return
        
        current_loss = self.get_monitor_value(logs, 'loss')
        if current is None:
            return        
        
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.monitor_op_s(current_second - self.min_delta_s,self.best_s):
                self.best_s = current_second
                if self.restore_best_weights :#and:
                    self.best_epoch = epoch
                    logs['best_epoch'] = 1
                    self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of '
                              'the best epoch')
                    self.model.set_weights(self.best_weights)
                    
    def on_train_end(self, logs=None):
        super(StopTraining,self).on_train_end(logs)
        
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: best weights ' % (self.best_epoch + 1))
                    
    def get_monitor_value(self, logs, monitor):
        monitor_value = logs.get(monitor)
        if monitor_value is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
        return monitor_value