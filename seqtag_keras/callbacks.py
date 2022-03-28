"""
Custom callbacks.
"""
import numpy as np
import tensorflow as tf
from numpy import average
from seqeval.metrics import classification_report, f1_score
from tensorflow.keras.callbacks import Callback
from tensorflow_addons.metrics import F1Score as TFAF1Score


class F1score(Callback):

    def __init__(self, seq, preprocessor=None):
        super(F1score, self).__init__()
        self.seq = seq
        self.p = preprocessor

    def get_lengths(self, y_true):
        lengths = []
        for y in y_true:
            try:
                i = list(y).index(0)
            except ValueError:
                i = len(y)
            lengths.append(i)

        return lengths

    def on_epoch_end(self, epoch, logs={}):
        label_true = []
        label_pred = []
        for i in range(len(self.seq)):
            x_true, y_true = self.seq[i]
            lengths = self.get_lengths(y_true)
            y_pred = self.model.predict_on_batch(x_true)

            y_true = self.p.inverse_transform(y_true, lengths)
            y_pred = self.p.inverse_transform(y_pred, lengths)

            label_true.extend(y_true)
            label_pred.extend(y_pred)

        score = f1_score(label_true, label_pred)
        print(' - f1: {:04.2f}'.format(score * 100))
        print(classification_report(label_true, label_pred))
        logs['f1'] = score


class F1scoreTF(Callback):

    def __init__(self, x_test, y_test, preprocessor):
        super(F1scoreTF, self).__init__()
        self.p = preprocessor
        self.x_test = x_test
        self.y_true = [self.p.inverse_transform_tf(curr_y) for curr_y in y_test]

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict_on_batch(self.x_test)
        y_pred = [self.p.inverse_transform_tf(curr_y) for curr_y in map(
            lambda lhs, rhs: lhs[:len(rhs)], y_pred, self.y_true)]

        score = f1_score(self.y_true, y_pred)
        print(' - f1: {:04.2f}'.format(score * 100))
        print(classification_report(self.y_true, y_pred))
        logs['f1'] = score
