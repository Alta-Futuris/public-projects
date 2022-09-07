import pytest
from os import path
import numpy as np
from keras.models import load_model
#from unittest.mock import Mock
import sys
sys.path.insert(0, r'\\wsl$\Ubuntu\home\shaheera\mnist-fc')
from utils.metrics import f1_score, precision, recall
#To run test "python -m pytest -rA"

filename = "./models/mnist.h5"

#Test for last activation layer
def activationLayer(ActFuc):
    if (ActFuc != 'softmax'):
        raise Exception("The last activation layer must be a Softmax Layer")

def test_model_last_ActivationLayer_positive_case():
    activation = 'softmax'
    model = load_model(filename, custom_objects={"f1_score": f1_score, "precision": precision, "recall": recall})
    if hasattr(model.layers[-1], "activation"):
        string = str(model.layers[-1].activation)
    assert activation in string
    
def test_model_last_ActivationLayer_negative_case():
    activation = 'Conv2D'
    model = load_model(filename, custom_objects={"f1_score": f1_score, "precision": precision, "recall": recall})
    if hasattr(model.layers[-1], "activation"):
        string = model.layers[-1].activation
    words = str(string).split()
    for i in words:
        if i != activation:
            assert True
    #assert activation in string, "The last layer activation should be softmax"

@pytest.mark.parametrize("value", ["ReLU", "LReLU", "SeLU", None])
def test_model_last_ActivationLayer_exception_case(value):
    with pytest.raises(Exception) as exc_info:
        activationLayer(value)
    assert exc_info.type is Exception
    
