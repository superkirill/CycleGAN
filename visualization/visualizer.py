import numpy as np
import cv2
from models import networks
from torch import nn, device, cuda
from . import parser

class Visualizer(object):
    """Visualizisation of the nn's layers responses"""
    def __init__(self, model, framework="PyTorch"):
        self.parser = parser.Parser(framework, model)
        self.layers = self.parser.get_layers()
        self.device = device("cuda" if cuda.is_available() else "cpu")

    def view_response(self, data, layer=0):
        """Display the response of a certain layer to the input data
        
            Parameters:
                data -- input data of the network
                layer -- non-negative integer representing the number of a layer
                        a response of which is to be viewed
        """
        target_layer = self.layers[layer]
        # Copy data to the gpu for faster forward propagation
        data_on_gpu = data.to(self.device)
        response_on_gpu = self.forward(data_on_gpu, layer)
        # Move the resulting response back to the cpu
        response_on_cpu = response_on_gpu.cpu()
        # Shape is (1, num_filters, resolution_x, resolution_y)
        print("\tShape of the response:", response_on_cpu.shape, " on the layer %d" % layer)
        # Convert the response from its tensor form to a numpy array
        response_on_cpu = (response_on_cpu[0][:].detach().numpy()*255).astype(np.uint8)
        # Compute the number of filters per row and column in the resulting image
        grid_size = int(round(np.sqrt(response_on_cpu.shape[0])))
        grid_size = grid_size+1 if (grid_size**2 != response_on_cpu.shape[0]) else grid_size
        image = None
        # Concatenate the responses of all filters in one image
        for j in range(grid_size):
            # Make a row of the image
            row = response_on_cpu[j*grid_size]
            for i in range(1,grid_size):
                if i+j*grid_size < response_on_cpu.shape[0]:
                    row = np.concatenate((row, response_on_cpu[i+j*grid_size]), axis=1)
            # Concatenate the image with the new row
            if image is None:
                image = row
            else:
                # If the number of filters does not allow to make a square image
                # Fill the remaining part with zeros and exit the loop
                if row.shape[1] < image.shape[1]:
                    row = np.concatenate((row, np.zeros((row.shape[0],image.shape[1]-row.shape[1]),dtype=np.uint8)), axis=1)
                    image = np.concatenate((image,row),axis=0)
                    break
                image = np.concatenate((image,row),axis=0)
        # Display the image
        cv2.imshow("Response of layer %d" % layer, image)
        cv2.waitKey()


    def forward(self, data, layer=0):
        """Run forwrard propagation until the specified layer
        
            Parameters:
                data -- input data
                layer -- non-negative integer representing the number of a layer
                        a response of which is to be viewed

            Return value:
                torch.tensor -- response of the given layer
        """
        target_layer = self.layers[layer]
        if layer > 0:
            data = self.forward(data, layer-1)
            return target_layer(data)
        else:
            return target_layer(data)
