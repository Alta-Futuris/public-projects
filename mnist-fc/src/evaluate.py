from keras.datasets import mnist 
from keras.models import load_model
from keras.utils import np_utils
import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils.metrics import f1_score, precision, recall
from utils.logger1 import TestingCallback, NTestBatchCSVLogger


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset', default='dataset/MNIST/mnist.npz', help="MNIST dataset")
parser.add_argument('--model', default = 'models/mnist1.h5', type = str, action = 'store', help="MNIST model")
parser.add_argument('--output', default = 'outputs/', type = str, action = 'store', help="Output Path")
args = parser.parse_args()

mylogger = TestingCallback()
CSV_test_batch_logger = NTestBatchCSVLogger("logs/batch_test_logs.csv", separator=',', append=False)

(X_train, y_train), (X_test, y_test) = mnist.load_data(args.dataset)

model1 = load_model(args.model, custom_objects={"f1_score": f1_score, "precision": precision, "recall": recall })

X_test = X_test.reshape(10000, 784)   # reshape 10,000 28 x 28 matrices into 10,000 784-length vectors.
X_test = X_test.astype('float32')
X_test /= 255
nb_classes = 10 # number of unique digits
Y_test = np_utils.to_categorical(y_test, nb_classes)

score = model1.evaluate(X_test, Y_test, callbacks = [TestingCallback(), CSV_test_batch_logger])
print('Test score:', score[0])
print('Test accuracy:', score[1])

# The predict_classes function outputs the highest probability class
# according to the trained classifier for each input example.
predicted_classes = model1.predict(X_test)
# Check which items we got right / wrong
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]

plt.figure()
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))
    
plt.tight_layout()
plt.savefig(args.output+'output1.png')
plt.figure()
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))
    
plt.tight_layout()
plt.savefig(args.output+'output2.png')
