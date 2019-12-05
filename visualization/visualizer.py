import numpy as np
import os
from PIL import Image
from torch import cat, nn
from torch import device, cuda, Tensor
from torchvision.transforms import transforms
from . import parser, image_viewer
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QTreeWidgetItem, QMessageBox


class Visualizer(QtWidgets.QMainWindow):
    """Visualization of the nn's layers responses"""
    def __init__(self, model, framework="PyTorch", opt=None, parent=None):
        super(Visualizer, self).__init__(parent)
        # Save model
        self.model = model
        # Save framework
        self.framework = framework
        # Load user interface design
        uic.loadUi('visualization/design.ui', self)
        # Load parser for a network model
        self.parser = parser.Parser(framework, model.netG_B.module)
        # Extract layers from the model
        self.layers = self.parser.get_layers()
        # If returned a UNet's layers with skip_connections
        if len(self.layers) == 2 and type(self.layers[0] is list):
            self.skip_connections = self.layers[1]
            self.layers = self.layers[0]
        # Set image import options
        self.opt = opt
        # Detect system's processing device
        self.device = device("cuda" if cuda.is_available() else "cpu")
        # Set input data for the network
        self.input = None
        # Set the number of layer the response of which is being displayed
        self.current_layer = 0
        # Set path to the dataset
        self.data_path = "D:/Документы/University of Bordeaux/TRDP/Code/Datasets"
        # Replace standard PyQt5 QGraphicsView widgets
        # with ImageViewer widgets borrowed from
        # https://stackoverflow.com/questions/35508711/how-to-enable-pan-and-zoom-in-a-qgraphicsview
        geom_old = self.graphicsView_response.geometry()
        self.graphicsView_response.deleteLater()
        self.graphicsView_response = image_viewer.ImageViewer(self)
        self.graphicsView_response.setGeometry(geom_old)
        geom_old = self.graphicsView_original.geometry()
        self.graphicsView_original.deleteLater()
        self.graphicsView_original = image_viewer.ImageViewer(self)
        self.graphicsView_original.setGeometry(geom_old)
        # Set some layout parameters
        self.button_change_direction.setIcon(QIcon("visualization/change_direction.png"))
        self.model_name.setText(str(self.model.model_names))
        # Connect Qt signals and slots
        self.tree_network.itemClicked.connect(self.architecture_clicked)
        self.tree_data.itemClicked.connect(self.data_clicked)
        self.button_change_direction.pressed.connect(self.switch_direction)
        # self.graphicsView_response.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        # self.graphicsView_response.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        # Load data
        self.load_data(self.data_path, self.tree_data)
        self.load_architecture(self.tree_network)

    def print_layers(self):
        """Print layers of the model to the console

            Parameters:
                ---
        """
        print("\n\nNumber of layers: %d" % len(self.layers))
        for i in range(len(self.layers)):
            print("\t %d:" % i, self.layers[i])
        print("\n\n")

        for i in range(len(self.skip_connections)):
            print("\t %d:" % i, self.skip_connections[i])
        print("\n\n")

    def view_response(self, data, layer=0):
        """Display the response of a certain layer to the input data
        
            Parameters:
                data -- input data of the network
                layer -- non-negative integer representing the number of a layer
                        a response of which is to be viewed
        """
        # Copy data to the gpu for faster forward propagation
        data_on_gpu = data.to(self.device)
        self.input = data_on_gpu
        response_on_gpu = self.forward(data_on_gpu, layer, is_top_level_call=True)
        # Move the resulting response back to the cpu
        response_on_cpu = response_on_gpu.cpu()
        # Shape is (1, num_filters, resolution_x, resolution_y)
        # print("\tShape of the response:", response_on_cpu.shape, " on the layer %d" % layer)
        # Convert the response from its tensor form to a numpy array
        response_on_cpu = (response_on_cpu[0][:].detach().numpy()*255).astype(np.uint8)
        # Compute the number of filters per row and column in the resulting image
        grid_size = int(round(np.sqrt(response_on_cpu.shape[0])))
        grid_size = grid_size+1 if (grid_size**2 != response_on_cpu.shape[0]) else grid_size
        # If the tensor has size 3 on axis 0, then it is an RGB image
        if response_on_cpu.shape[0] == 3:
            self.display(response_on_gpu.cpu(), window="right")
        else:
            image = None
            # Concatenate the responses of all filters in one image
            for j in range(grid_size):
                # Make a row of the image
                if j * grid_size < response_on_cpu.shape[0]:
                    row = response_on_cpu[j*grid_size]
                else:
                    break
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
            self.display(image, window="right")

    def forward(self, data, layer=0, is_top_level_call=False):
        """Run forwrard propagation until the specified layer
        
            Parameters:
                data -- input data
                layer -- non-negative integer representing the number of a layer
                        a response of which is to be viewed
                is_top_level_call -- a boolean representing if a current call
                        is a top level or a recursive call

            Return value:
                torch.tensor -- response of the given layer
        """
        target_layer = self.layers[layer]
        if layer in list(zip(*self.skip_connections))[1]:
            connection = self.skip_connections[list(zip(*self.skip_connections))[1].index(layer)][0]
            x = self.forward(self.input, connection)
            data = self.forward(data, layer-1)
            if is_top_level_call:
                return target_layer(data)
            else:
                return target_layer(cat([x, data], 1))
        if layer > 0:
            data = self.forward(data, layer-1)
            return target_layer(data)
        else:
            return target_layer(data)

    def tensor2im(self, input_image, imtype=np.uint8):
        """"Converts a Tensor array into a numpy image array.

        Parameters:
            input_image (tensor) --  the input image tensor array
            imtype (type)        --  the desired type of the converted numpy array
        """
        if not isinstance(input_image, np.ndarray):
            if isinstance(input_image, Tensor):  # get the data from a variable
                image_tensor = input_image.data
            else:
                return input_image
            image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
            if image_numpy.shape[0] == 1:  # grayscale to RGB
                image_numpy = np.tile(image_numpy, (3, 1, 1))
            image_numpy = (np.transpose(image_numpy,
                                        (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        else:  # if it is a numpy array, do nothing
            image_numpy = input_image
        return image_numpy.astype(imtype)

    def display(self, data, window="left"):
        """Display an image represented as a tensor

            Parameters:
                data -- input tensor or np.array representing an image
                window -- a window in which the image will be displayed
        """
        image = self.tensor2im(data)
        if len(image.shape) > 2:
            height, width, channel = image.shape
            image_format = QImage.Format_RGB888
            bytes_per_line = 3 * width
        else:
            height, width = image.shape
            image_format = QImage.Format_Grayscale8
            bytes_per_line = width
        if window == "left":
            view = self.graphicsView_original
        else:
            view = self.graphicsView_response
        qt_image = QImage(image.tobytes(), width, height, bytes_per_line, image_format)
        view.setPhoto(QPixmap(qt_image))

    def load_architecture(self, tree):
        """Display the model's architecture in a tree-view

            Parameters:
                tree - QTreeWidget in which the architecture is displayed
        """
        tree.setHeaderLabel("Layer")
        for index, layer in enumerate(self.layers):
            item = QtWidgets.QTreeWidgetItem()
            if type(layer) is nn.modules.conv.Conv2d:
                item.setIcon(0, QIcon('visualization/conv.png'))
            elif type(layer) is nn.modules.conv.ConvTranspose2d:
                item.setIcon(0, QIcon('visualization/deconv.png'))
            elif type(layer) is nn.modules.activation.ReLU:
                item.setIcon(0, QIcon('visualization/relu.png'))
            elif type(layer) is nn.modules.activation.LeakyReLU:
                item.setIcon(0, QIcon('visualization/leakyrelu.png'))
            elif type(layer) is nn.modules.activation.Tanh:
                item.setIcon(0, QIcon('visualization/tanh.png'))
            elif type(layer) is nn.modules.instancenorm.InstanceNorm2d:
                item.setIcon(0, QIcon('visualization/instancenorm2d.png'))
            item.setText(0, str(index+1) + ": " + str(layer))
            tree.addTopLevelItem(
                item
            )

    def load_data(self, path, tree):
        """Display the train/test data in a tree-view recursively

            Parameters:
                path - a string presenting a root directory
                tree - a QTreeWidget object in which the data is loaded
        """
        for element in os.listdir(path):
            path_info = path + "/" + element
            parent_itm = QTreeWidgetItem(tree, [os.path.basename(element)])
            if os.path.isdir(path_info):
                self.load_data(path_info, parent_itm)
                parent_itm.setIcon(0, QIcon('visualization/folder.ico'))
            else:
                parent_itm.setIcon(0, QIcon('visualization/file.ico'))

    def show_window(self):
        """Display the GUI window"""
        self.show()

    def architecture_clicked(self, item):
        """Handle a mouse click on the QTreeWidget containing
            network's architecture by displaying the response of
            a respective layer

            Parameters:
                item - QTreeWidgetItem that is currently selected
        """
        self.current_layer = self.tree_network.indexOfTopLevelItem(item)
        self.view_response(self.input, layer=self.current_layer)
        self.title_response.setText("Response of layer %d" % (self.tree_network.indexOfTopLevelItem(item) + 1))

    def data_clicked(self, item):
        """Handle a mouse click on the QTreeWidget containing
            train/test - set the image at the specified path
            as self.input

            Parameters:
                item - QTreeWidgetItem that is currently selected
        """
        path = self.get_item_path(item)
        if not os.path.isdir(path):
            # Load image
            image = Image.open(path)
            # Resize the image
            image = image.resize((self.opt.load_size, self.opt.load_size))
            # Get the number of channels of input image
            btoA = self.opt.direction == 'BtoA'
            input_nc = self.opt.output_nc if btoA else self.opt.input_nc
            grayscale = (input_nc == 1)
            # Convert to RGB or Gray scale
            if grayscale:
                image = image.convert('L')
                transform_list = [transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,))]
            else:
                image = image.convert('RGB')
                transform_list = [transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5),
                                                       (0.5, 0.5, 0.5))]
            trans = transforms.Compose(transform_list)
            image = trans(image)
            self.input = image.unsqueeze(0)
            self.display(data=self.input, window="left")
            self.view_response(self.input, layer=self.current_layer)
            self.title_response.setText("Response of layer %d" % (self.current_layer+1))

    def switch_direction(self):
        """Switch direction in a CycleGAN model"""
        # Load parser for a network model
        if self.opt.direction == 'BtoA':
            self.parser = parser.Parser(self.framework, self.model.netG_A.module)
            self.direction.setText("From A to B")
            self.opt.direction = 'AtoB'
        else:
            self.parser = parser.Parser(self.framework, self.model.netG_B.module)
            self.direction.setText("From B to A")
            self.opt.direction = 'BtoA'
        # Extract layers from the model
        self.layers = self.parser.get_layers()
        # If returned a UNet's layers with skip_connections
        if len(self.layers) == 2 and type(self.layers[0] is list):
            self.skip_connections = self.layers[1]
            self.layers = self.layers[0]

    def get_item_path(self, item):
        """Get a full path to the object in the file system

            Parameters:
                item -- QTreeWidgetItem representing a file
                    or a directory
            Return value:
                str -- a full path to the file/directory
        """
        path = item.text(0)
        if item.parent():
            path = self.get_item_path(item.parent()) + "/" + path
        else:
            path = self.data_path + "/" + path
        return path