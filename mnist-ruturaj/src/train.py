import torch
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional
from Dataclasses.hyperpar import Hyperparameters 
from modelparser import ModelParser
from model import ModelParser
from model import MnistModel

# dataset = MNIST(root = 'data/', train = True, download = True)
# test_dataset = MNIST(root = 'data/', train = False)

def preprocess(batch_size):
    # Converting data into tensor
    dataset = MNIST(root = 'data/', train = True, download = True, transform = transforms.ToTensor() )

    # Splitting the data into training and validation sets
    train_data, val_data = random_split(dataset, [50000, 10000])

    # Converting data into batches for training and validation
    batch_size = batch_size
    train_loader = DataLoader(train_data,batch_size, shuffle = True)
    # Shuffle set to true to get different batch of data every epoch
    val_loader = DataLoader(val_data, batch_size)

    return train_loader,val_loader

def accuracy(pred, labels):
    _, pred = torch.max(pred).item()
    acc = torch.sum(pred == labels).item()/ len(pred)
    return acc


def fit(epochs, lr, train_loader, val_loader):
    
    # Defining loss function
    loss_fn = torch.nn.functional.cross_entropy

    opt = torch.optim.SGD(model.parameters(), lr)
    dl = []
    da = []

    for epoch in range(epochs):

        l = []
        a = []

        for image, labels in train_loader:
            pred = model(image.reshape(-1,784))
            # loss calculation
            loss = loss_fn(pred, labels)
            # Gradient calculation
            loss.backward()
            # Weight adjusting
            opt.step()
            # Setting gradient to zero
            opt.zero_grad()
            

        for image, labels in val_loader:
            pred = model(image.reshape(-1,784))
            # loss calculation
            loss = loss_fn(pred, labels)
            # accuracy calulation
            _, pred = torch.max(pred, dim = 1)
            acc = (torch.sum(pred == labels).item()/len(pred))

            l.append(loss)
            a.append(acc)


        dl.append(sum(l)/len(l))
        da.append(sum(a)/len(a))

#         print('Epoch loss : ', torch.stack(l).mean())
#         print('Epoch accu : ', torch.stack(a).mean())
#         print(dl)
        print('Accuracy of epoch {}, is {}'.format((epoch + 1),  sum(a)/len(a)))
        print('Loss of epoch {}, is {}'.format((epoch + 1), sum(l)/len(l) ))
    return da,dl


if __name__ == "__main__":
    parser = ModelParser("../base_config.json")
    layers = parser.get_list()
    hp = parser.get_hp()
    model = MnistModel(layers)
    model = model.build_model()
    for i in hp:
        kwargs = i
    Hyperparameters(**kwargs)
    batch_size = Hyperparameters.batch_size
    epochs = Hyperparameters.epochs
    lr = Hyperparameters.lr
    opt = Hyperparameters.opt
    train_loader,val_loader=preprocess(batch_size)
    fit(epochs, lr, train_loader, val_loader)

    