# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 17:52:37 2021

@author: Sina
"""

import matplotlib.pyplot as plt
 
from gan.utils import sample_noise, show_images, deprocess_img, preprocess_img
 
def train(D, G, D_solver, G_solver, discriminator_loss, generator_loss, show_every=250,
              batch_size=128, noise_size=100, num_epochs=10, train_loader=None, device=None):
    """
    Train loop for GAN.

    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    - train_loader: image dataloader
    - device: PyTorch device
    """
    iter_count = 0
    for epoch in range(num_epochs):
        print('EPOCH: ', (epoch+1))
        for x, _ in train_loader:
            _, input_channels, img_size, _ = x.shape

            real_images = preprocess_img(x).to(device)  # normalize

            # Store discriminator loss output, generator loss output, and fake image output
            # in these variables for logging and visualization below
            d_error = None
            g_error = None
            fake_images = None

            D_solver.zero_grad()
            rn_noise = sample_noise(batch_size,noise_size).to(device)
            fake_images = G(rn_noise).reshape(real_images.shape)

            d_error = discriminator_loss(D(real_images),D(fake_images))

            d_error.backward()
            D_solver.step()
            
            #########       END      ###########

            G_solver.zero_grad()
            rn_noise = sample_noise(batch_size,noise_size).to(device)
            fake_images = G(rn_noise).reshape(real_images.shape)

            g_error = generator_loss(D(fake_images))

            g_error.backward(retain_graph=True)
            G_solver.step()
            
            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_error.item(),g_error.item()))
                disp_fake_images = deprocess_img(fake_images.data)  # denormalize
                imgs_numpy = (disp_fake_images).cpu().numpy()
                show_images(imgs_numpy[0:16], color=input_channels!=1)
                plt.show()
                print()
            iter_count += 1