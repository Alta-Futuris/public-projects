from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation

print("Mnist_Model")
def FC_model():
  model = Sequential()
  model.add(Dense(512, input_shape=(784,)))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))
  model.add(Dense(512))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))
  model.add(Dense(512))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))
  model.add(Dense(10))
  model.add(Activation('softmax'))
  return model
