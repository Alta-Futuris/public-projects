import os
#from keras import backend as K
import keras


class MyLogger(keras.callbacks.Callback):
    def __init__(self, n):
        self.n = n   # print loss & acc every n epochs

    def log(self, epoch, logs = {}):
        if epoch % self.n == 0:
            curr_loss = logs.get('loss')
            curr_acc = logs.get('acc') * 100
            curr_F1Score = logs.get('f1_score') * 100
            curr_precision = logs.get('precision') * 100
            curr_recall = logs.get('recall') * 100
            print("epoch = %4d  loss = %0.6f  acc = %0.2f f1_score = %0.2f precision = %0.2f recall = %0.2f%%"\
                    % (epoch, curr_loss, curr_acc, curr_F1Score, curr_precision, curr_recall))

        


