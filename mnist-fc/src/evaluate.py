#importing necessary libraries
from keras.datasets import mnist 
from keras.models import load_model
from keras.utils import np_utils
import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils.metrics import f1_score, precision, recall
from utils.logger1 import TestingCallback, NTestBatchCSVLogger
from datetime import datetime

#Parsing the dataset path, model, and locating the outputs
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset', default='./dataset/MNIST/mnist.npz', help="MNIST dataset")
parser.add_argument('--model', default = 'models/mnist1.h5', type = str, action = 'store', help="MNIST model")
parser.add_argument('--output', default = 'outputs/', type = str, action = 'store', help="Output Path")
args = parser.parse_args()

#parameters for logging the evaluation results
mylogger = TestingCallback()
filename_logger = "logs/test_logs/batch_test_logs_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+".csv"
CSV_test_batch_logger = NTestBatchCSVLogger(filename_logger, separator=',', append=False)

#Loading the test folder
(X_train, y_train), (X_test, y_test) = mnist.load_data(args.dataset)

#Loading model with the custom objects
model1 = load_model(args.model, custom_objects={"f1_score": f1_score, "precision": precision, "recall": recall })

#Data pre-processing
X_test = X_test.reshape(10000, 784)   # reshape 10,000 28 x 28 matrices into 10,000 784-length vectors.
X_test = X_test.astype('float32')
X_test /= 255
nb_classes = 10 # number of unique digits
Y_test = np_utils.to_categorical(y_test, nb_classes)

#Evaluating the model
score = model1.evaluate(X_test, Y_test, callbacks = [TestingCallback(), CSV_test_batch_logger])
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Test f1_score:', score[2])
print('Test precision:', score[3])
print('Test recall:', score[4])

# The predict_classes function outputs the highest probability class
# according to the trained classifier for each input example.
predicted_classes = model1.predict(X_test)

#plot the results with predicted and test values
plt.figure()
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(X_test[i].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(np.where(predicted_classes[i] == np.amax(predicted_classes[i])), y_test[i]))
    
plt.tight_layout()
filename_2 = "output_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+".png"
plt.savefig(args.output+filename_2)
print(X_test[0].shape)
