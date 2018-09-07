"""BEGAN implementation that takes images from a folder.
"""

from torch.autograd import Variable
from torch.utils.data import DataLoader

import argparse
import logging
import os
import numpy as np
import math
import random
import torch
import torchvision.datasets as dset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils


def weights_init_normal(m):
    """Custom weights initialization called on netG and netD.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Decoder(nn.Module):

    def __init__(self, num_channels, hidden_size, noise_size, image_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.multiplier = image_size // 4
        self.fc = nn.Linear(noise_size, hidden_size * self.multiplier**2)
        self.first_layer = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.ELU(),
            nn.UpsamplingNearest2d(scale_factor=2)
        )
        self.second_layer = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.ELU(),
            nn.UpsamplingNearest2d(scale_factor=2)
        )
        self.third_layer = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(hidden_size, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, noise):
        out = self.fc(noise)
        out = out.view(out.shape[0], self.hidden_size, self.multiplier, self.multiplier)
        out = self.first_layer(out)
        out = self.second_layer(out)
        out = self.third_layer(out)
        return out


class Encoder(nn.Module):

    def __init__(self, num_channels, hidden_size, noise_size, image_size):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, hidden_size, 3, 1, 1),
            nn.ELU(),

            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(hidden_size, 2*hidden_size, 3, 1, 1),
            nn.ELU(),
            nn.AvgPool2d(2, stride=2),

            nn.Conv2d(2*hidden_size, 2*hidden_size, 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(2*hidden_size, 3*hidden_size, 3, 1, 1),
            nn.ELU(),
            nn.AvgPool2d(2, stride=2),

            nn.Conv2d(3*hidden_size, 3*hidden_size, 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(3*hidden_size, 3*hidden_size, 3, 1, 1),
            nn.ELU(),
        )
        multiplier = image_size // 4
        self.fc = nn.Linear(3*hidden_size * multiplier**2, noise_size)

    def forward(self, img):
        out = self.main(img)
        out = out.view((out.size(0), -1))
        out = self.fc(out)
        return out


class Generator(nn.Module):

    def __init__(self, num_channels, hidden_size, noise_size, image_size):
        super(Generator, self).__init__()
        self.decoder = Decoder(num_channels, hidden_size, noise_size, image_size)

    def forward(self, noise):
        return self.decoder(noise)


class Discriminator(nn.Module):

    def __init__(self, num_channels, hidden_size, noise_size, image_size):
        super(Discriminator, self).__init__()
        self.encoder = Encoder(num_channels, hidden_size, noise_size, image_size)
        self.decoder = Decoder(num_channels, hidden_size, noise_size, image_size)

    def forward(self, image):
        out = self.encoder(image)
        out = self.decoder(out)
        return out


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='cifar10 | folder | mnist')
    parser.add_argument('--dataroot', default='downloads/madhubani/', help='Path to dataset.')
    parser.add_argument('--n-epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--batch-size', type=int, default=128, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--workers', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--image-size', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--noise-size', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--hidden-size', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--sample-interval', type=int, default=400, help='number of image channels')
    parser.add_argument('--generator', default=None, help="path to generator (to continue training)")
    parser.add_argument('--discriminator', default=None, help="path to discriminator (to continue training)")
    parser.add_argument('--outf', default='weights', help='folder to output images and model checkpoints')
    parser.add_argument('--manual-seed', default=1234, type=int, help='manual seed')
    args = parser.parse_args()

    if not os.path.exists(args.outf):
        os.makedirs(args.outf)

    # Config logging
    log_format = '%(levelname)-8s %(message)s'
    logfile = os.path.join(args.outf, 'train.log')
    logging.basicConfig(filename=logfile, level=logging.INFO, format=log_format)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(args.__dict__)

    # Set random seed.
    logging.info("Random Seed: " + str(args.manual_seed))
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    # Housekeeping.
    cuda = True if torch.cuda.is_available() else False
    num_channels = 3

    # Create dataset.
    if args.dataset == 'cifar10':
        dataset = dset.CIFAR10(root=args.dataroot, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(args.image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    elif args.dataset == 'mnist':
        dataset = dset.MNIST(root=args.dataroot, download=True,
                             transform=transforms.Compose([
                                   transforms.Resize(args.image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        num_channels = 1
    elif args.dataset == 'folder':
        dataset = dset.ImageFolder(root=args.dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(args.image_size),
                                       transforms.CenterCrop(args.image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
					     shuffle=True, num_workers=args.workers)

    # Initialize generator.
    generator = Generator(num_channels, args.hidden_size, args.noise_size, args.image_size)
    generator.apply(weights_init_normal)
    if args.generator is not None:
        generator.load_state_dict(torch.load(args.generator))
    if cuda:
        generator.cuda()
    logging.info(generator)

    # Initialize discriminator.
    discriminator = Discriminator(num_channels, args.hidden_size, args.noise_size, args.image_size)
    discriminator.apply(weights_init_normal)
    if args.discriminator is not None:
        discriminator.load_state_dict(torch.load(args.discriminator))
    if cuda:
        discriminator.cuda()
    logging.info(discriminator)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    fixed_noise = Tensor(np.random.normal(0, 1, (args.batch_size, args.noise_size)))

    # BEGAN hyper parameters
    gamma = 0.75
    lambda_k = 0.001
    k = 0.

    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], args.noise_size))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = torch.mean(torch.abs(discriminator(gen_imgs) - gen_imgs))

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            d_real = discriminator(real_imgs)
            d_fake = discriminator(gen_imgs.detach())

            d_loss_real = torch.mean(torch.abs(d_real - real_imgs))
            d_loss_fake = torch.mean(torch.abs(d_fake - gen_imgs.detach()))
            d_loss = d_loss_real - k * d_loss_fake

            d_loss.backward()
            optimizer_D.step()

            #----------------
            # Update weights
            #----------------

            diff = torch.mean(gamma * d_loss_real - d_loss_fake)

            # Update weight term for fake samples
            k = k + lambda_k * diff.item()
            k = min(max(k, 0), 1) # Constraint to interval [0, 1]

            # Update convergence metric
            M = (d_loss_real + torch.abs(diff)).data[0]

            #--------------
            # Log Progress
            #--------------

            logging.info("Epoch: [%d/%d], Iteration: [%d/%d], Loss_D: %.4f, Loss_G: %.4f, M: %f, k: %f" % (
                epoch, args.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), M, k))

            if i % 100 == 0:
                fake = generator(fixed_noise)
                vutils.save_image(fake.detach(),
                        '%s/fake_samples_epoch_%03d.png' % (args.outf, epoch),
                        normalize=True)

        # do checkpointing
        torch.save(generator.state_dict(), '%s/netG_epoch_%d.pth' % (args.outf, epoch))
        torch.save(discriminator.state_dict(), '%s/netD_epoch_%d.pth' % (args.outf, epoch))
