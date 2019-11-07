from models import networks
from torch import nn

class PyTorchParser():
    """Extract separate layers from PyTorch nn"""
    def __init__(self, model):
        self.model = model
        
    def get_layers(self):
        layers = list()
        if type(self.model) is networks.UnetGenerator:
            layers = list(self.model.children())
            all_unwraped = False
            while all_unwraped is False:
                all_unwraped = True
                for i in range(len(layers)):
                    if type(layers[i]) is networks.UnetSkipConnectionBlock or \
                       type(layers[i]) is nn.Sequential:
                        if i == len(layers)-1:
                            layers = layers[:i] + list(layers[i].children())
                        elif i == 0:
                            layers = list(layers[i].children()) + layers[i+1:]
                        else:
                            layers = layers[:i] + list(layers[i].children()) + layers[i+1:]
                        all_unwraped = False
                        break

            print("Number of layers: %d" % len(layers))
            for layer in layers:
                print("\t", layer)
            print("\n\n")
        else:
            raise Exception("This type of architecture is not supproted yet")
        return layers