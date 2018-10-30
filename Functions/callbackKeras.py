from keras.callbacks import Callback


def set_early_stopping():
    raise NotImplementedError

def set_model_checkpoint():
    raise NotImplementedError


class MyHistory(Callback):
    def on_train_begin(self,logs={}):
        print 'start training'


    def on_batch_end(self,batch,logs={}):
        print 'batch({0}): loss -> {1}'.format(batch,logs['loss'])
