import numpy as np                   
import matplotlib.pyplot as plt      
import random                        
from keras.datasets import mnist     
from keras.utils import np_utils  
from model import FC_model
from utils.metrics import f1_score, precision, recall
from mnist import MNIST
import argparse
from keras.models import model_from_json

# Arguments for the training
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('dataset', default='\\wsl$\\Ubuntu\\home\\shaheera\\mnist-fc\\dataset\\MNIST\\mnist.npz', help="MNIST dataset")
#parser.add_argument('configuration', default='\\wsl$\\Ubuntu\\home\\shaheera\\mnist-fc\\base_config.json', help="Configuration")
args = parser.parse_args()

(X_train, y_train), (X_test, y_test) = mnist.load_data(args.dataset)
print("Shapes of dataset")
print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape) 

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


#json_file = open(args.configuration, 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#model = model_from_json(loaded_model_json)
model = FC_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1_score, precision, recall])
model.fit(X_train, Y_train,
          batch_size=128, epochs=5,
          verbose=1)

print(model.summary())

