"""
    Train a CycleGAN model:
        python train.py --dataroot "PATH/TO/DATASET"
"""
from options.all_options import AllOptions
from models import create_model
from visualization import visualizer_train
from PyQt5 import QtWidgets

if __name__ == '__main__':
    opt = AllOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.name = "cyclegan_colorization_visualizer_development"   # model name; it will be saved in "./checkpoints/"
    opt.netG = "unet_shallow"   # type of the generator network
    opt.direction = 'AtoB'      # direction of image translation (here, A - RGB, B - Gray scale)
    opt.lambda_identity = 0.0   # set to 0, to exclude the identity loss from the objective function
    opt.load_size = 128         # resize the images to 128x128
    opt.isTrain = True
    opt.serial_batches = False  # disable data shuffling
    opt.no_flip = True          # no flip

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    app = QtWidgets.QApplication([])
    visualizer = visualizer_train.VisualizerTrain(model, framework="PyTorch", opt=opt)
    visualizer.show_window()
    app.exec_()
