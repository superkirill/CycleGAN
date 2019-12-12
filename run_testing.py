"""
    Test a CycleGAN model:
        python test.py --dataroot "PATH/TO/DATASET"
"""
from options.all_options import AllOptions
from data import create_dataset
from models import create_model
from visualization import visualizer
from PyQt5 import QtWidgets

if __name__ == '__main__':
    opt = AllOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 0    # test code only supports num_threads = 1
    opt.batch_size = 1     # test code only supports batch_size = 1
    opt.name = "unet_128"  # name of the model to load from "./checkpoints/"
    opt.netG = "unet_128"  # type of the generator network
    opt.input_nc = 3       # input channels
    opt.output_nc = 1      # output channels
    opt.direction = 'BtoA'
    opt.isTrain = False
    opt.no_flip = True     # no flip

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    app = QtWidgets.QApplication([])
    visualizer = visualizer.Visualizer(model, framework="PyTorch", opt=opt)
    visualizer.show_window()
    app.exec_()



