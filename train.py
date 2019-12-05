"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
import pickle
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def compute_losses(cycle_model, opt):
    # GAN loss D_A(G_A(A))
    loss_G_A = cycle_model.criterionGAN(cycle_model.netD_A(cycle_model.fake_B), True)
    # GAN loss D_B(G_B(B))
    loss_G_B = cycle_model.criterionGAN(cycle_model.netD_B(cycle_model.fake_A), True)
    # Forward cycle loss || G_B(G_A(A)) - A||
    loss_cycle_A = cycle_model.criterionCycle(cycle_model.rec_A, cycle_model.real_A) * opt.lambda_A
    # Backward cycle loss || G_A(G_B(B)) - B||
    loss_cycle_B = cycle_model.criterionCycle(cycle_model.rec_B, cycle_model.real_B) * opt.lambda_B

    fake_B = cycle_model.fake_B_pool.query(cycle_model.fake_B)
    loss_D_A = cycle_model.backward_D_basic(cycle_model.netD_A, cycle_model.real_B, fake_B)

    fake_A = cycle_model.fake_A_pool.query(cycle_model.fake_A)
    loss_D_B = cycle_model.backward_D_basic(cycle_model.netD_B, cycle_model.real_A, fake_A)
    return {"G_A":loss_G_A.item(), "G_B":loss_G_B.item(), "cycle_A":loss_cycle_A.item(), "cycle_B":loss_cycle_B.item(), "D_A":loss_D_A.item(), "D_B":loss_D_B.item()}


if __name__ == '__main__':
    sns.set()
    opt = TrainOptions().parse()   # get training options
    opt.name = "unet_128"
    opt.netG = "unet_128"
    opt.direction = "AtoB"
    opt.lambda_identity = 0.0
    opt.batch_size = 1
    opt.isTrain = True
    opt.serial_batches = False
    opt.no_flip = True
    opt.model = "cycle_gan"
    opt.load_size = 128
    opt.dataset_mode = "unaligned"
    opt.input_nc = 3
    opt.output_nc = 1
    # opt.epoch_count = 85
    # opt.continue_train = True
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.

    val_opt = opt
    val_opt.dataroot = "Datasets/Colorization/validation"
    validation_dataset = create_dataset(val_opt)
    print('The number of training images = %d' % dataset_size)


    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    loss_D_A = []
    loss_G_A = []
    loss_cycle_A = []
    loss_D_B = []
    loss_G_B = []
    loss_cycle_B = []
    t_loss_D_A = []
    t_loss_G_A = []
    t_loss_cycle_A = []
    t_loss_D_B = []
    t_loss_G_B = []
    t_loss_cycle_B = []
    iter_axis = [0]

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            #visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()




        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        losses = compute_losses(model, opt)
        loss_D_A.append(losses['D_A'])
        loss_G_A.append(losses['G_A'])
        loss_cycle_A.append(losses['cycle_A'])
        loss_D_B.append(losses['D_B'])
        loss_G_B.append(losses['G_B'])
        loss_cycle_B.append(losses['cycle_B'])
        iter_axis.append(iter_axis[-1] + 1)
        fig_losses = plt.figure(figsize=(20, 9), dpi=80)
        plot = fig_losses.add_subplot(111)
        plt.tight_layout()
        plot.plot(iter_axis[1:], loss_D_A, 'r-.', label='D_A')
        plot.plot(iter_axis[1:], loss_G_A, 'm-.', label='G_A')
        plot.plot(iter_axis[1:], loss_cycle_A, 'y-.', label='Cycle_A')
        plot.plot(iter_axis[1:], loss_D_B, 'b-.', label='D_B')
        plot.plot(iter_axis[1:], loss_G_B, 'g-.', label='G_B')
        plot.plot(iter_axis[1:], loss_cycle_B, 'c-.', label='Cycle_B')
        plt.title("Losses")

        opt.isTrain = False
        selection = np.random.randint(0, len(validation_dataset), size=1)[0]
        for i, data in enumerate(validation_dataset):
            if i == selection:
                model.set_input(data)  # unpack data from data loader
                model.test()  # run inference
                break
        losses = compute_losses(model, opt)
        t_loss_D_A.append(losses['D_A'])
        t_loss_G_A.append(losses['G_A'])
        t_loss_cycle_A.append(losses['cycle_A'])
        t_loss_D_B.append(losses['D_B'])
        t_loss_G_B.append(losses['G_B'])
        t_loss_cycle_B.append(losses['cycle_B'])
        plot.plot(iter_axis[1:], t_loss_D_A, 'r', label='Val_D_A')
        plot.plot(iter_axis[1:], t_loss_G_A, 'm', label='Val_G_A')
        plot.plot(iter_axis[1:], t_loss_cycle_A, 'y', label='Val_Cycle_A')
        plot.plot(iter_axis[1:], t_loss_D_B, 'b', label='Val_D_B')
        plot.plot(iter_axis[1:], t_loss_G_B, 'g', label='Val_G_B')
        plot.plot(iter_axis[1:], t_loss_cycle_B, 'c', label='Val_Cycle_B')
        plot.legend()
        opt.isTrain = True

        with open('checkpoints/%s/losses.pkl' % (opt.name), 'wb') as file:
            pickle.dump([loss_D_A, loss_D_B, loss_G_A, loss_G_B, loss_cycle_A, loss_cycle_B, t_loss_D_A, t_loss_D_B,
                         t_loss_G_A, t_loss_G_B, t_loss_cycle_A, t_loss_cycle_B, iter_axis], file)


        fig_losses.savefig('checkpoints/%s/losses_epoch_%d.jpg' % (opt.name, epoch))
        plt.close(fig=fig_losses)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
