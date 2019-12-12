import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
from torch import cat, nn
from torch import device, cuda, Tensor
from torchvision.transforms import transforms
from . import parser, image_viewer
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QTreeWidgetItem, QMessageBox
from PyQt5.QtCore import QRunnable, pyqtSlot, QThreadPool
from data import create_dataset
from .communicate import Communicate


class Training(QRunnable):

    def __init__(self, app):
        super(Training, self).__init__()
        self.app = app
        self.c = app.c
        self.c.fixed_requested.connect(self.fixed_requested)
        self.c.random_requested.connect(self.random_requested)
        self.interrupt_requested = False
        self.running = False
        self.stop_training = False
        self.input_A2B = None
        self.input_B2A = None
        self.show_fixed = False

    def fixed_requested(self, image, input_type):
        if input_type == "A2B":
            self.input_A2B = image
        else:
            self.input_B2A = image
        self.show_fixed = True

    def random_requested(self):
        self.show_fixed = False

    def request_interrupt(self):
        self.interrupt_requested = True

    def release_interrupt(self):
        self.interrupt_requested = False

    def stop(self):
        self.stop_training = True

    def is_running(self):
        return self.running

    @pyqtSlot()
    def run(self):
        self.running = True
        self.app.opt.display_freq = 1
        # Initialize lists storing training losses
        loss_D_A = []
        loss_G_A = []
        loss_cycle_A = []
        loss_D_B = []
        loss_G_B = []
        loss_cycle_B = []
        iter_axis = [0]
        # Display initial learning rate
        self.app.current_learning_rate.setText("%.2e" % self.app.model.optimizers[0].param_groups[0]['lr'])
        # Outer loop for different epochs
        # Save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        for epoch in range(self.app.opt.epoch_count,
                           self.app.opt.niter + self.app.opt.niter_decay + 1):
            epoch_start_time = time.time()  # timer for entire epoch
            epoch_iter = 0  # the number of training iterations in current epoch
            # self.app.opt.display_freq = 1
            for i, data in enumerate(self.app.dataset):  # inner loop within one epoch
                # Pause the training process if the pause button in the GUI was pressed
                while self.interrupt_requested is True:
                    pass
                # Stop the training process if the stop button in the GUI was pressed
                if self.stop_training is True:
                    break
                # Update the number of iterations
                self.app.total_iters += self.app.opt.batch_size
                epoch_iter += self.app.opt.batch_size
                self.app.model.set_input(data)  # unpack data from dataset and self.apply preprocessing
                self.app.model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

                if self.app.total_iters % self.app.opt.display_freq == 0:  # display images
                    if self.show_fixed is False:
                        # Select images automatically
                        if self.app.device.type == "cuda":
                            imaA2B = self.app.view_response(self.app.model.real_A.cpu(), self.app.net["layersA2B"], self.app.net["skipA2B"],
                                                            layer=self.app.current_layer_A2B)
                            imaB2A = self.app.view_response(self.app.model.real_B.cpu(), self.app.net["layersB2A"], self.app.net["skipB2A"],
                                                            layer=self.app.current_layer_B2A)
                            imaA = self.app.model.real_A.cpu()
                            imaB = self.app.model.real_B.cpu()
                        else:
                            imaA2B = self.app.view_response(self.app.model.real_A, self.app.net["layersA2B"], self.app.net["skipA2B"],
                                                            layer=self.app.current_layer_A2B)
                            imaB2A = self.app.view_response(self.app.model.real_A, self.app.net["layersB2A"], self.app.net["skipB2A"],
                                                            layer=self.app.current_layer_B2A)
                            imaA = self.app.model.real_A
                            imaB = self.app.model.real_B
                    else:
                        # Select images specified by the user
                        # for the A2B network
                        if self.input_A2B is not None:
                            imaA2B = self.app.view_response(self.input_A2B, self.app.net["layersA2B"], self.app.net["skipA2B"],
                                                            layer=self.app.current_layer_A2B)
                            imaA = self.app.tensor2im(self.input_A2B)
                            # If the batch size is 1,
                            # show the image passing through the network at the current iteration
                            if self.app.opt.batch_size == 1:
                                real_A = self.app.tensor2im(data['A'])
                                imaA[:32, :32] = cv2.resize(real_A, dsize=(32, 32),
                                                            interpolation=cv2.INTER_CUBIC)[:32, :32]
                                # Draw a border around the small image
                                imaA[:32, 32] = 255
                                imaA[32, :32] = 255
                        else:
                            # If the user has not chosen any image, show zeros
                            imaA = np.zeros((256, 256))
                            imaA2B = np.zeros((256, 256))
                        # Select images specified by the user
                        # for the B2A network
                        if self.input_B2A is not None:
                            imaB2A = self.app.view_response(self.input_B2A, self.app.net["layersB2A"], self.app.net["skipB2A"],
                                                            layer=self.app.current_layer_B2A)
                            imaB = self.app.tensor2im(self.input_B2A)
                            # If the batch size is 1,
                            # show the image passing through the network at the current iteration
                            if self.app.opt.batch_size == 1:
                                real_B = self.app.tensor2im(data['B'])
                                imaB[:32, :32] = cv2.resize(real_B, dsize=(32, 32),
                                                            interpolation=cv2.INTER_CUBIC)[:32, :32]
                                # Draw a border around the small image
                                imaB[:32, 32] = 255
                                imaB[32, :32] = 255
                        else:
                            # If the user has not chosen any image, show zeros
                            imaB = np.zeros((256, 256))
                            imaB2A = np.zeros((256, 256))

                    # Request the redraw of graphics widgets
                    self.c.display_requested.emit(self.app.tensor2im(imaA2B), "A2B")
                    self.c.display_requested.emit(self.app.tensor2im(imaB2A), "B2A")
                    self.c.display_requested.emit(self.app.tensor2im(imaA), "A")
                    self.c.display_requested.emit(self.app.tensor2im(imaB), "B")
                    self.app.current_iteration.setText(str(epoch_iter))

                    # Compute the losses
                    losses = self.app.model.get_current_losses()
                    loss_D_A.append(losses['D_A'])
                    loss_G_A.append(losses['G_A'])
                    loss_cycle_A.append(losses['cycle_A'])
                    loss_D_B.append(losses['D_B'])
                    loss_G_B.append(losses['G_B'])
                    loss_cycle_B.append(losses['cycle_B'])
                    iter_axis.append(iter_axis[-1] + 1)

                    # Create a figure
                    fig_losses = plt.figure(figsize=(8, 6), dpi=200)
                    plot = fig_losses.add_subplot(111)

                    # Plot the losses on the figure
                    plt.tight_layout()
                    plot.plot(iter_axis[1:], loss_D_A, 'r', label='D_A')
                    plot.plot(iter_axis[1:], loss_G_A, 'm-.', label='G_A')
                    plot.plot(iter_axis[1:], loss_cycle_A, 'y--', label='Cycle_A')
                    plot.plot(iter_axis[1:], loss_D_B, 'b', label='D_B')
                    plot.plot(iter_axis[1:], loss_G_B, 'g-.', label='G_B')
                    plot.plot(iter_axis[1:], loss_cycle_B, 'c--', label='Cycle_B')
                    plot.legend()

                    # Convert the figure to a np.ndarray
                    fig_losses.canvas.draw()
                    data = np.fromstring(fig_losses.canvas.tostring_rgb(), dtype=np.uint8)
                    data = data.reshape(fig_losses.canvas.get_width_height()[::-1] + (3,))

                    # Request the redraw of the graphics widget
                    self.c.display_requested.emit(data, "losses")

                    # Close the figure
                    plt.close(fig=fig_losses)

            if epoch % self.app.opt.save_epoch_freq == 0:  # cache the model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, self.app.total_iters))
                self.app.model.save_networks('latest')
                self.app.model.save_networks(epoch)

            # Update the number of the current epoch in the GUI
            self.app.current_epoch.setText(str(epoch))

            print('End of epoch %d / %d \t Time Taken: %d sec' % (
                epoch, self.app.opt.niter + self.app.opt.niter_decay, time.time() - epoch_start_time))

            # Update the learning rate at the end of every epoch.
            self.app.model.update_learning_rate()
            # Update the learning rate in the GUI
            self.app.current_learning_rate.setText("%.2e" % self.app.model.optimizers[0].param_groups[0]['lr'])
            # Stop the training process if the stop button was pressed in the GUI
            if self.stop_training is True:
                break
        # Update the state of the process
        self.running = False
        self.stop_training = False


class VisualizerTrain(QtWidgets.QMainWindow):
    """Visualization of the nn's layers responses"""

    def __init__(self, model, framework="PyTorch", opt=None):
        super(VisualizerTrain, self).__init__()
        # Set the tool seaborn for plotting
        sns.set()
        # Save model
        self.model = model
        # Save framework
        self.framework = framework
        # Load user interface design
        uic.loadUi('visualization/design_training.ui', self)
        # Load parser for a network model
        self.parserA2B = parser.Parser(framework, model.netG_A.module)
        self.parserB2A = parser.Parser(framework, model.netG_B.module)
        # Extract layers from the model
        layersA2B = self.parserA2B.get_layers()
        layersB2A = self.parserB2A.get_layers()
        # If returned a UNet's layers with skip_connections
        if len(layersA2B) == 2 and type(layersA2B[0] is list):
            self.net = {"layersA2B": layersA2B[0], "skipA2B": layersA2B[1]}
        if len(layersB2A) == 2 and type(layersB2A[0] is list):
            self.net["layersB2A"] = layersB2A[0]
            self.net["skipB2A"] = layersB2A[1]
        # Set image import options
        self.opt = opt
        # Detect system's processing device
        self.device = device("cuda" if cuda.is_available() else "cpu")
        # Set the number of layer the response of which is being displayed
        self.current_layer_A2B = 0
        self.current_layer_B2A = 0
        # the total number of training iterations
        self.total_iters = 0
        # Dataset used for training
        self.dataset = create_dataset(self.opt)
        # Set path to the dataset
        self.data_path = self.opt.dataroot
        # Replace standard PyQt5 QGraphicsView widgets
        # with ImageViewer widgets borrowed from
        # https://stackoverflow.com/questions/35508711/how-to-enable-pan-and-zoom-in-a-qgraphicsview
        geom_old = self.graphicsView_datasetA.geometry()
        self.graphicsView_datasetA.deleteLater()
        self.graphicsView_datasetA = image_viewer.ImageViewer(self)
        self.graphicsView_datasetA.setGeometry(geom_old)
        geom_old = self.graphicsView_datasetB.geometry()
        self.graphicsView_datasetB.deleteLater()
        self.graphicsView_datasetB = image_viewer.ImageViewer(self)
        self.graphicsView_datasetB.setGeometry(geom_old)
        geom_old = self.graphicsView_response_A2B.geometry()
        self.graphicsView_response_A2B.deleteLater()
        self.graphicsView_response_A2B = image_viewer.ImageViewer(self)
        self.graphicsView_response_A2B.setGeometry(geom_old)
        geom_old = self.graphicsView_response_B2A.geometry()
        self.graphicsView_response_B2A.deleteLater()
        self.graphicsView_response_B2A = image_viewer.ImageViewer(self)
        self.graphicsView_response_B2A.setGeometry(geom_old)
        # Input data for Colorization
        self.input_rgb = None
        self.input_gray = None
        geom_old = self.graphicsView_losses.geometry()
        self.graphicsView_losses.deleteLater()
        self.graphicsView_losses = image_viewer.ImageViewer(self)
        self.graphicsView_losses.setGeometry(geom_old)
        # Connect Qt signals and slots
        self.tree_network_A2B.itemClicked.connect(self.architectureA2B_clicked)
        self.tree_network_B2A.itemClicked.connect(self.architectureB2A_clicked)
        self.tree_dataset.itemClicked.connect(self.dataset_clicked)
        self.start_training.pressed.connect(self.start_training_clicked)
        self.pause_training.pressed.connect(self.pause_training_clicked)
        self.stop_training.pressed.connect(self.stop_training_clicked)
        self.n_epochs.editingFinished.connect(self.set_epochs)
        self.batch_size.editingFinished.connect(self.set_batch_size)
        self.radioButton_random_selection.toggled.connect(self.set_random_selection)
        self.radioButton_fixed_selection.toggled.connect(self.set_fixed_selection)
        # Load data
        self.load_data(self.data_path, self.tree_dataset)
        self.load_architecture(self.tree_network_A2B, self.net["layersA2B"])
        self.load_architecture(self.tree_network_B2A, self.net["layersB2A"])
        # Take the training to a separate thread
        self.threadpool = QThreadPool()
        self.c = Communicate()
        self.c.display_requested.connect(self.display)
        self.training = Training(self)

    def set_random_selection(self):
        """Handle a mouse click on the QRadioButton by setting
            the mode of data selection for visualization
        """
        self.c.random_requested.emit()

    def set_fixed_selection(self):
        """Handle a mouse click on the QRadioButton by setting
            the mode of data selection for visualization
        """
        if not self.input_gray is None and not self.input_rgb is None:
            self.c.fixed_requested.emit(self.input_rgb, "A2B")
            self.c.fixed_requested.emit(self.input_gray, "B2A")

    def print_layers(self):
        """Print layers of the model to the console"""
        print("\n\nNumber of layers: %d" % len(self.net["layersA2B"]))
        for i in range(len(self.net["layersA2B"])):
            print("\t %d:" % i, self.net["layersA2B"][i])
        print("\n\n")

        for i in range(len(self.net["skipA2B"])):
            print("\t %d:" % i, self.net["skipA2B"][i])
        print("\n\n")

        print("\n\nNumber of layers: %d" % len(self.net["layersB2A"]))
        for i in range(len(self.net["layersB2A"])):
            print("\t %d:" % i, self.net["layersB2A"][i])
        print("\n\n")

        for i in range(len(self.net["skipB2A"])):
            print("\t %d:" % i, self.net["skipB2A"][i])
        print("\n\n")

    def view_response(self, data, net, skip_connections=[], layer=0):
        """Display the response of a certain layer to the input data

            Parameters:
                data - input data of the network
                net - model of the network
                skip_connections - a list containing unet's skip-connections
                layer - non-negative integer representing the number of a layer
                        a response of which is to be viewed
        """
        # Copy data to the gpu for faster forward propagation
        data_on_gpu = data.to(self.device)
        self.input = data_on_gpu
        response_on_gpu = self.forward(data_on_gpu, net, skip_connections, layer, is_top_level_call=True)
        # Move the resulting response back to the cpu
        response_on_cpu = response_on_gpu.cpu()
        # Shape is (1, num_filters, resolution_x, resolution_y)
        # print("\tShape of the response:", response_on_cpu.shape, " on the layer %d" % layer)
        # Convert the response from its tensor form to a numpy array
        response_on_cpu = (response_on_cpu[0][:].detach().numpy() * 255).astype(np.uint8)
        # Compute the number of filters per row and column in the resulting image
        grid_size = int(round(np.sqrt(response_on_cpu.shape[0])))
        grid_size = grid_size + 1 if (grid_size ** 2 != response_on_cpu.shape[0]) else grid_size
        # If the tensor has size 3 on axis 0, then it is an RGB image
        if response_on_cpu.shape[0] == 3:
            return response_on_gpu.cpu()
        else:
            image = None
            # Concatenate the responses of all filters in one image
            for j in range(grid_size):
                # Make a row of the image
                if j * grid_size < response_on_cpu.shape[0]:
                    row = response_on_cpu[j * grid_size]
                else:
                    break
                for i in range(1, grid_size):
                    if i + j * grid_size < response_on_cpu.shape[0]:
                        row = np.concatenate((row, response_on_cpu[i + j * grid_size]), axis=1)
                # Concatenate the image with the new row
                if image is None:
                    image = row
                else:
                    # If the number of filters does not allow to make a square image
                    # Fill the remaining part with zeros and exit the loop
                    if row.shape[1] < image.shape[1]:
                        row = np.concatenate(
                            (row, np.zeros((row.shape[0], image.shape[1] - row.shape[1]), dtype=np.uint8)),
                            axis=1)
                        image = np.concatenate((image, row), axis=0)
                        break
                    image = np.concatenate((image, row), axis=0)
            # Display the image
            return image

    def forward(self, data, net, skip_connections=[], layer=0, is_top_level_call=False):
        """Run forwrard propagation until the specified layer

            Parameters:
                data - input data
                net - model of the neural network
                skip_connections - indices of the layers connected by the unet
                        skip-connections
                layer - non-negative integer representing the number of a layer
                        a response of which is to be viewed
                is_top_level_call - a boolean representing if a current call
                        is a top level or a recursive call

            Return value:
                torch.tensor - response of the given layer
        """
        target_layer = net[layer]
        if layer in list(zip(*skip_connections))[1]:
            connection = skip_connections[list(zip(*skip_connections))[1].index(layer)][0]
            x = self.forward(self.input, net, skip_connections, connection)
            data = self.forward(data, net, skip_connections, layer - 1)
            if is_top_level_call:
                return target_layer(data)
            else:
                return target_layer(cat([x, data], 1))
        if layer > 0:
            data = self.forward(data, net, skip_connections, layer - 1)
            return target_layer(data)
        else:
            return target_layer(data)

    def tensor2im(self, input_image, imtype=np.uint8):
        """"Converts a Tensor array into a numpy image array.

            Parameters:
                input_image (tensor) -  the input image tensor array
                imtype (type)        -  the desired type of the converted numpy array

            Return:
                np.ndarray object representing the image
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

    def display(self, data, window="A"):
        """Display an image represented as a tensor

            Parameters:
                data - input tensor or np.array representing an image
                window - a window in which the image will be displayed
        """
        image = self.tensor2im(data)
        if len(image.shape) > 2:
            height, width, channel = image.shape
            format = QImage.Format_RGB888
            bytes_per_line = 3 * width
        else:
            height, width = image.shape
            format = QImage.Format_Grayscale8
            bytes_per_line = width
        if window == "A":
            view = self.graphicsView_datasetA
        elif window == "B":
            view = self.graphicsView_datasetB
        elif window == "A2B":
            view = self.graphicsView_response_A2B
        elif window == "B2A":
            view = self.graphicsView_response_B2A
        elif window == "losses":
            view = self.graphicsView_losses
        else:
            pass
        qt_image = QImage(image.tobytes(), width, height, bytes_per_line, format)
        view.setPhoto(QPixmap(qt_image))

    def load_architecture(self, tree, net):
        """Display the model's architecture in a tree-view

            Parameters:
                tree - QTreeWidget in which the architecture is displayed
                net - model of the network
        """
        tree.setHeaderLabel("Layer")
        for index, layer in enumerate(net):
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
            item.setText(0, str(index + 1) + ": " + str(layer))
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

    def architectureA2B_clicked(self, item):
        """Handle a mouse click on the QTreeWidget containing
            network's architecture by displaying the response of
            a respective layer

            Parameters:
                item - QTreeWidgetItem that is currently selected
        """
        self.current_layer_A2B = self.tree_network_A2B.indexOfTopLevelItem(item)
        if self.device.type == "cuda":
            ima = self.view_response(self.model.real_A.cpu(),
                               self.net["layersA2B"],
                               self.net["skipA2B"],
                               layer=self.current_layer_A2B)
        else:
            ima = self.view_response(self.model.real_A,
                               self.net["layersA2B"],
                               self.net["skipA2B"],
                               layer=self.current_layer_A2B)
        self.display(ima, window="A2B")

    def architectureB2A_clicked(self, item):
        """Handle a mouse click on the QTreeWidget containing
            network's architecture by displaying the response of
            a respective layer

            Parameters:
                item - QTreeWidgetItem that is currently selected
        """
        self.current_layer_B2A = self.tree_network_B2A.indexOfTopLevelItem(item)
        if self.device.type == "cuda":
            ima = self.view_response(self.model.real_B.cpu(),
                               self.net["layersB2A"],
                               self.net["skipB2A"],
                               layer=self.current_layer_B2A)
        else:
            ima = self.view_response(self.model.real_B.cpu(),
                               self.net["layersB2A"],
                               self.net["skipB2A"],
                               layer=self.current_layer_B2A)
        self.display(ima, window="B2A")

    def dataset_clicked(self, item):
        """Handle a mouse click on the QTreeWidget containing
            train/test - set the image at the specified path
            as self.input

            Parameters:
                item - QTreeWidgetItem that is currently selected
        """
        # Get a path to the item
        path = self.get_item_path(item)
        # If the item is a directory
        if os.path.isdir(path):
            # Try to open a dataset located in the directory
            try:
                self.opt.dataroot = path
                self.dataset = create_dataset(self.opt)
                self.model.set_input([data for _, data in enumerate(self.dataset)][0])
                self.model.test()
            except:
                QMessageBox.warning(None, "Dataset selection",
                                    "A valid dataset root directory must contain the following directories:"
                                    "trainA, trainB, testA, testB")
                return
            # And refresh the graphics widgets
            if self.device.type == "cuda":
                self.display(self.model.real_A.cpu(), window="A")
                self.display(self.model.real_B.cpu(), window="B")
                self.display(self.model.fake_B.cpu(), window="A2B")
                self.display(self.model.fake_A.cpu(), window="B2A")
            else:
                self.display(self.model.real_A, window="A")
                self.display(self.model.real_B, window="B")
                self.display(self.model.fake_B, window="A2B")
                self.display(self.model.fake_A, window="B2A")
        # If the item is not a directory
        else:
            # Load an image
            image = Image.open(path)
            # Resize the image
            image = image.resize((self.opt.load_size, self.opt.load_size))
            # If the image is from the domain A
            if "trainA" in path or "testA" in path:
                # Convert to RGB
                image = image.convert('RGB')
                # Convert to tensor and normalize
                transform_list = [transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5),
                                                       (0.5, 0.5, 0.5))]
                trans = transforms.Compose(transform_list)
                image = trans(image)
                # Save as input
                self.input_rgb = image.unsqueeze(0)
                # Request an update of input in the training process
                self.c.fixed_requested.emit(self.input_rgb, "A2B")
            # If the image is from the domain B
            elif "trainB" in path or "testB" in path:
                # Convert grayscale
                image = image.convert('L')
                # Convert to tensor and normalize
                transform_list = [transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,))]
                trans = transforms.Compose(transform_list)
                image = trans(image)
                # Save as input
                self.input_gray = image.unsqueeze(0)
                # Request an update of input in the training process
                self.c.fixed_requested.emit(self.input_gray, "B2A")
            self.radioButton_fixed_selection.setChecked(True)

    def start_training_clicked(self):
        """Handle a mouse click on the QPushButton
            by starting the training process
        """
        if self.training.is_running() is False:
            # Read and update the parameters
            self.set_epochs()
            self.set_batch_size()
            # Create a training process
            self.threadpool.clear()
            self.training = Training(self)
            # Run the training process
            self.threadpool.start(self.training)
        else:
            # If training was previously stopped
            # then resume it
            self.training.release_interrupt()
        self.pause_training.setEnabled(True)
        self.stop_training.setEnabled(True)
        self.start_training.setEnabled(False)

    def pause_training_clicked(self):
        """Handle a mouse click on the QPushButton
            by pausing the training process
        """
        self.training.request_interrupt()
        self.start_training.setEnabled(True)
        self.stop_training.setEnabled(True)
        self.pause_training.setEnabled(False)

    def stop_training_clicked(self):
        """Handle a mouse click on the QPushButton
            by stopping the training process
        """
        self.training.stop()
        self.start_training.setEnabled(True)
        self.stop_training.setEnabled(False)
        self.pause_training.setEnabled(False)

    def set_epochs(self):
        """Read the contents of the QLineEdit containing
            the number of training epochs and update the
            corresponding field
        """
        self.opt.niter = int(self.n_epochs.text())

    def set_batch_size(self):
        """Read the contents of the QLineEdit containing
            the batch size epochs and update the
            corresponding field
        """
        self.opt.batch_size = int(self.batch_size.text())

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
