"""
    Test a CycleGAN model:
        python test.py --dataroot Datasets/Colorization --name cyclegan_colorization --model cycle_gan
         --load_size 128 --netG unet_128 --direction BtoA
"""
from options.all_options import AllOptions
from data import create_dataset
from models import create_model
from visualization import visualizer
from visualization import visualizer_train
from PyQt5 import QtWidgets

if __name__ == '__main__':
    opt = AllOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations

    app = QtWidgets.QApplication([])
    # visualizer = visualizer.Visualizer(model, framework="PyTorch", opt=opt)
    visualizer = visualizer_train.VisualizerTrain(model, framework="PyTorch", opt=opt)
    visualizer.show_window()
    app.exec_()



