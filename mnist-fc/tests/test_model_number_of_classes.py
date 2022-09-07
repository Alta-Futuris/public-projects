import pytest
from os import path
import numpy as np
from keras.models import load_model
#from unittest.mock import Mock
import sys
sys.path.insert(0, r'\\wsl$\Ubuntu\home\shaheera\mnist-fc')
from utils.metrics import f1_score, precision, recall


filename = "./models/mnist.h5"

#Test for checking number of classes
def inputdata(num_classes):
    if (num_classes>10 or num_classes < 0):
        raise Exception("Not the Valid number of classes. The number should be 10")
    return np.random.rand(10, 784)
def test_model_output_positive_case():
    num_classes = 10
    model = load_model(filename, custom_objects={"f1_score": f1_score, "precision": precision, "recall": recall})
    image = inputdata(num_classes)
    image = image.astype('float32')
    pred = model(image)
    #10 classe
    assert len(pred[0]) == 10
    
def test_model_output_negative_case():
    num_classes = 5
    model = load_model(filename, custom_objects={"f1_score": f1_score, "precision": precision, "recall": recall})
    image = inputdata(num_classes)
    image = image.astype('float32')
    pred = model(image)
    #10 classes
    assert len(pred[0]) != 5, "The number of classes should be equal to 10"

@pytest.mark.parametrize("value", [12, 50, 100])
def test_model_output_exception_case(value):
    with pytest.raises(Exception) as exc_info:
        inputdata(value)
    assert exc_info.type is Exception
    
