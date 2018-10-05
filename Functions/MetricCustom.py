import keras.backend as K
import sklearn.metrics as Met
from StatFunctions import sp
import tensorflow as tf


def ind_SP(y_true, y_pred):

    value_max = 1
    value_min = -1
    num_classes = 24

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, value_min, value_max)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, value_min, value_max)))
    possible_positives = K.sum(K.round(K.clip(y_true, value_min, value_max)))

    precision_tf = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())

    sp_tf = K.sqrt(K.mean(recall)*K.prod(K.pow(recall,1/float(num_classes))))

    return sp_tf

def f1_score(y_true, y_pred):
    """
    f1 score

    :param y_true:
    :param y_pred:
    :return:
    """
    tp_3d = K.concatenate(
        [
            K.cast(y_true, 'bool'),
            K.cast(K.round(y_pred), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    fp_3d = K.concatenate(
        [
            K.cast(K.abs(y_true - K.ones_like(y_true)), 'bool'),
            K.cast(K.round(y_pred), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    fn_3d = K.concatenate(
        [
            K.cast(y_true, 'bool'),
            K.cast(K.abs(K.round(y_pred) - K.ones_like(y_pred)), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    tp = K.sum(K.cast(K.all(tp_3d, axis=1), 'int32'))
    fp = K.sum(K.cast(K.all(fp_3d, axis=1), 'int32'))
    fn = K.sum(K.cast(K.all(fn_3d, axis=1), 'int32'))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * ((precision * recall) / (precision + recall))
