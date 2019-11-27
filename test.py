"""
    Test a CycleGAN model:
        python test.py --dataroot Datasets/Colorization --name cyclegan_colorization --model cycle_gan
         --load_size 128 --netG unet_128 --direction BtoA
"""
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from visualization import visualizer, data_generator
from PyQt5 import QtWidgets

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # model.set_input([data for _, data in enumerate(dataset)][0])
    # model.test()
    # import numpy as np
    # import cv2
    # for _, data in enumerate(dataset):
    #     image = (data['A'][0].numpy() * 255).astype(np.uint8)
    #     import matplotlib.pyplot as plt
    #     from util import util
    #     image = util.tensor2im(data['A'])
    #     plt.imshow(image)
    #     plt.show()
    # import sys
    # sys.exit(0)

    app = QtWidgets.QApplication([])
    visualizer = visualizer.Visualizer(model, framework="PyTorch", opt=opt)
    visualizer.show_window()
    app.exec_()

    # data_app = QtWidgets.QApplication([])
    # generator = data_generator.DataGenerator()
    # data_app.exec_()
