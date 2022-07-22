import torch
import torchvision
from torchvision.datasets import MNIST
import mlflow.pytorch
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from model import ModelParser
from model import MnistModel


test_dataset = MNIST(root='data/', 
                     train=False,
                     transform=transforms.ToTensor())
len(test_dataset)
j=0

#   # Inference after loading the logged model
# model_uri = "runs:/{}/model".format(run.info.run_id)
# loaded_model = mlflow.pytorch.load_model(model_uri)

# Model initializing

parser = ModelParser("../base_config.json")
layers = parser.get_list()
kwargs = parser.get_hp()[0]
model = MnistModel(layers)
model = model.build_model()

# loading the model

model.load_state_dict((torch.load('../Models/MNIST_128_25_0.5_SGD_cross_entropy.pth')))

for i in range(len(test_dataset)):
    image, label = test_dataset[i]

    pred = model(out)
    _, pred = torch.max(pred, dim = 1)
#     print(pred)
#     print(label)
    if pred == label:
        j+=1
print('Test dataset accauracy :', j/len(test_dataset))


