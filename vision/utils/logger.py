import os
import numpy as np
import errno
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from IPython import display
from matplotlib import pyplot as plt
import torch

'''
    TensorBoard Data will be stored in './runs' path
'''


class Logger:

    def __init__(self, comment):
        self.comment = comment
        self.log_dir = self.comment + "_logs"
        # TensorBoard
        self.writer = SummaryWriter(logdir=self.log_dir)

    def log_gan_error(self, d_error, g_error, epoch, n_batch, num_batches):
        d_error = d_error.data.cpu().numpy()
        g_error = g_error.data.cpu().numpy()
        step = Logger._step(epoch, n_batch, num_batches)
        self.writer.add_scalar(
            f'{self.comment}/D_error', d_error, step)
        self.writer.add_scalar(
            f'{self.comment}/G_error', g_error, step)
        
    def display_gan_status(self, d_error, g_error, d_pred_real, d_pred_fake, epoch, num_epochs, n_batch, num_batches):
      d_error = d_error.data.cpu().numpy()
      g_error = g_error.data.cpu().numpy()
      d_pred_real = d_pred_real.data
      d_pred_fake = d_pred_fake.data
        
      print(f'Epoch: [{epoch}/{num_epochs}], Batch Num: [{n_batch}/{num_batches}]')
      print(f'Discriminator Loss: {d_error:.4f}, Generator Loss: {g_error:.4f}')
      print(f'D(x): {d_pred_real.mean():.4f}, D(G(z)): {d_pred_fake.mean():.4f}')
      
    def log_images(self, images, epoch, n_batch, num_batches, format='NCHW', normalize=True):
        '''
        input images are expected in format (NCHW)
        '''
        if type(images) == np.ndarray:
            images = torch.from_numpy(images)
        
        if format=='NHWC':
            images = images.transpose(1,3)
        
        num_images = images.size(0)
        
        step = Logger._step(epoch, n_batch, num_batches)
        img_name = '{}/images{}'.format(self.comment, '')

        # Make horizontal grid from image tensor
        horizontal_grid = vutils.make_grid(
            images, normalize=normalize, scale_each=True)
        # Make vertical grid from image tensor
        nrows = int(np.sqrt(num_images))
        grid = vutils.make_grid(
            images, nrow=nrows, normalize=True, scale_each=True)

        # Add horizontal images to tensorboard
        self.writer.add_image(img_name, horizontal_grid, step)

        # Save plots
        self.save_torch_images(horizontal_grid, grid, epoch, n_batch)

    def save_torch_images(self, horizontal_grid, grid, epoch, n_batch, plot_horizontal=True):
        # Plot and save horizontal
        fig = plt.figure(figsize=(16, 16))
        plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
        plt.axis('off')
        if plot_horizontal:
            display.display(plt.gcf())
        self._save_images(fig, epoch, n_batch, 'hori')
        plt.close()

        # Save squared
        fig = plt.figure()
        plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
        plt.axis('off')
        self._save_images(fig, epoch, n_batch)
        plt.close()

    def _save_images(self, fig, epoch, n_batch, comment=''):
      out_dir = f'{self.log_dir}/images'
      Logger._make_dir(out_dir)
      fig.savefig('{}/{}_epoch_{}_batch_{}.png'.format(out_dir,
                                                         comment, epoch, n_batch))

    def close(self):
        self.writer.close()

    # Private Functionality

    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise