from models import networks
from torch import nn

class PyTorchParser():
    """Extract separate layers from PyTorch nn"""
    def __init__(self, model):
        self.model = model
        
    def get_layers(self):
        """Extract layers from the model as a list of layers
            
            Return value: list() - a list of layers or
                          [list(), list()] - a list of layers and a list of
                                skip connections
        """
        layers = list()
        # UnetGenerator parser
        if type(self.model) is networks.UnetGenerator:
            layers = list(self.model.children())
            all_unwraped = False
            skip_connections = list()
            marker = 0
            # Unwrap all of the UnetSkipConnectionBlock's
            # Putting markers before and after them, in order to
            # Restore skip connections afrterwards
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
                            if type(layers[i]) is networks.UnetSkipConnectionBlock:
                                layers = layers[:i] + [marker] +\
                                                 list(layers[i].children()) +\
                                         [marker] + layers[i+1:]
                            else:
                                layers = layers[:i] + list(layers[i].children()) +\
                                            layers[i+1:]
                            marker += 1
                        all_unwraped = False
                        break
            contains_markers = True
            # Remove all markers, saving the positions of layers they signal to connect
            while contains_markers:
                contains_markers = False
                layer_counter=0
                for i in range(len(layers)):
                    if type(layers[i]) is int:
                        marker = layers[i]
                        connection_from = layer_counter
                        for j in range (i+1, len(layers)):
                            if type(layers[j]) is int:
                                if layers[j] == marker:
                                    break
                            else:
                                layer_counter += 1
                        connection_to = layer_counter
                        skip_connections.append((connection_from, connection_to))
                        layers.remove(marker)
                        layers.remove(marker)
                        contains_markers = True
                        break
                    else:
                        layer_counter += 1
            return [layers, skip_connections]
        else:
            raise Exception("This type of architecture is not supproted yet")
            return None