import numpy as np                   
import matplotlib.pyplot as plt      
import random
import keras
from keras.datasets import mnist     
from keras.utils import np_utils  
#from model import FC_model
from utils.metrics import f1_score, precision, recall
from mnist import MNIST
import argparse
from keras.models import model_from_json
from utils.logger1 import MyLogger, NBatchCSVLogger, NEpochCSVLogger
from datetime import datetime

# Arguments for the training
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset', default='dataset/MNIST/mnist.npz', help="MNIST dataset")
parser.add_argument('--config', default = 'base_config.json', type = str, action = 'store', help="Configuration file")
args = parser.parse_args()

batch_size = 128
epochs = 5

#Loading data
(X_train, y_train), (X_test, y_test) = mnist.load_data(args.dataset)
print("Shapes of dataset")
print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape) 

#data prep-processing
X_train = X_train.reshape(60000, 784) # reshape 60,000 28 x 28 matrices into 60,000 784-length vectors.
X_test = X_test.reshape(10000, 784)   # reshape 10,000 28 x 28 matrices into 10,000 784-length vectors.
X_train = X_train.astype('float32')   # change integers to 32-bit floating point numbers
X_test = X_test.astype('float32')
X_train /= 255                        # normalize each value for each pixel for the entire vector for each input
X_test /= 255
print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

nb_classes = 10 # number of unique digits
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

dataset_path = args.dataset
print(dataset_path)
filename = args.config
print("The path of the file is:", filename)

#Reading the config file
json_file = open(filename,'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', f1_score, precision, recall])


#creating logs
mylogger = MyLogger(n=epochs)
filename_batch_train_logs = "outputs/logs/train_logs/batch_train_logs_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+".csv"
CSV_batch_logger = NBatchCSVLogger(filename_batch_train_logs, separator=',', append=False)
filename_epoch_train_logs = "outputs/logs/train_logs/epoch_train_logs_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+".csv"
CSV_epoch_logger = NEpochCSVLogger(filename_epoch_train_logs, separator=',', append=False)

#training the model
history = model.fit(X_train, Y_train, batch_size=batch_size,
                    epochs=epochs, validation_split = 0.20,
                    verbose=0,
                    callbacks = [mylogger, CSV_batch_logger, CSV_epoch_logger])

#saving the model
filename = "mnist_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+".h5"
model.save('models/'+filename)

