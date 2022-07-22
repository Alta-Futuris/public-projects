from Dataclasses.layer import Layer
import json
import torch
from modelparser import ModelParser
from Dataclasses.hyperpar import Hyperparameters

class MnistModel(torch.nn.Module):
    def __init__(self,layer_list):
        super().__init__()
        self.layer_list = layer_list

    def _activation_mapper(self, act_string):

        if act_string == "Relu":
            return torch.nn.ReLU()
        
        elif act_string == 'Sigmoid':
            return torch.nn.Sigmoid()
        
        else:
            pass

    def build_model(self):

        module_list = list()

        for layer_ix, layer in enumerate(self.layer_list):
            
            if "Linear" in layer.Layer_name:
                linear_layer = torch.nn.Linear(layer.Num_inputs, layer.Num_outputs, bias = layer.Bias)

            module_list.append(linear_layer)

            act = self._activation_mapper(layer.Activation)
            module_list.append(act)

            dpt = torch.nn.Dropout2d(layer.dropout) if layer.Dropout else None
            if dpt:
                module_list.append(dpt)

            self.pred = torch.nn.Sequential(*module_list)
        return self.pred


if __name__ == "__main__":
    parser = ModelParser("../base_config.json")
    layers = parser.get_list()
    kwargs = parser.get_hp()[0]
    model = MnistModel(layers)
    Hyperparameters(**kwargs)

    