"""DCGAN implementation that takes images from a folder.
"""

from __future__ import print_function

import argparse
import logging
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils


def weights_init(m):
    """Custom weights initialization called on netG and netD.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):

    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()

            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):

    def __init__(self, ngpu, nc, ndf):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(

            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='cifar10 | folder | mnist')
    parser.add_argument('--dataroot', default='downloads/madhubani/', help='Path to dataset.')
    parser.add_argument('--workers', type=int, help='Number of data loading workers.', default=4)
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size')
    parser.add_argument('--image-size', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default=None, help="path to netG (to continue training)")
    parser.add_argument('--netD', default=None, help="path to netD (to continue training)")
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
    cudnn.benchmark = True
    device = torch.device("cuda:0" if args.cuda else "cpu")
    nc = 3

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
        nc = 1
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

    # Create generator.
    netG = Generator(args.ngpu, args.nz, args.ngf, nc).to(device)
    netG.apply(weights_init)
    if args.netG is not None:
        netG.load_state_dict(torch.load(args.netG))
    logging.info(netG)

    # Create discriminator.
    netD = Discriminator(args.ngpu, nc, args.ndf).to(device)
    netD.apply(weights_init)
    if args.netD is not None:
        netD.load_state_dict(torch.load(args.netD))
    logging.info(netD)

    # Setup loss.
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(args.batch_size, args.nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # Setup optimizer.
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    for epoch in range(args.niter):
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with real.
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device)

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with fake.
            noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)

            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake

            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost

            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()

            optimizerG.step()

            logging.info('Epoch: [%d/%d], Iteration: [%d/%d], Loss_D: %.4f, '
                  'Loss_G: %.4f, D(x): %.4f, D(G(z)): %.4f / %.4f'
                  % (epoch, args.niter, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            if i % 100 == 0:
                vutils.save_image(real_cpu,
                        '%s/real_samples.png' % args.outf,
                        normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(),
                        '%s/fake_samples_epoch_%03d.png' % (args.outf, epoch),
                        normalize=True)

        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (args.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (args.outf, epoch))
