# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
from torch import nn, tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import transforms, ToTensor
import torchvision.utils as vutils


device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print("Device:", device)


def load_data(plot=True):
    data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    dataloader = DataLoader(
        data,
        batch_size=64,
        shuffle=True
    )
    for x, y in dataloader:
        print("Shape of x:", x.shape)
        break
    if plot:
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(
            next(iter(dataloader))[0].to(device)[:64],
            padding=2, pad_value=0.5, normalize=True).cpu(), (1, 2, 0)))
        plt.show()
    return dataloader


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        # ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        # https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
        self.main = nn.Sequential(
            nn.Unflatten(1, (16, 1, 1)),
            # latent
            nn.ConvTranspose2d(16, 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            # 16 x 4x4
            nn.ConvTranspose2d(16, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            # 16 x 8x8
            nn.ConvTranspose2d(16, 4, 4, 2, 2, bias=False),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.1, inplace=True),
            # 8 x 14x14
            nn.ConvTranspose2d(4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # 1 x 28x28
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.main = nn.Sequential(
            # 1 x 28x28
            nn.Conv2d(1, 16, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 4 x 14x14
            nn.Conv2d(16, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 x 7x7
            nn.Conv2d(16, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 x 4x4
            nn.Conv2d(16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


def weights_init(m):
    classname = m.__class__.__name__
    if 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    else:
        if 'weight' in m.__dict__ and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if 'bias' in m.__dict__ and m.bias is not None:
            nn.init.normal_(m.bias.data, 0.0, 0.02)


def train_epoch(dataloader, net_g, net_d, loss_fn, optimizer_g, optimizer_d):
    for batch, (x, y) in enumerate(dataloader, 0):

        # add real batches
        net_d.zero_grad()
        real = x.to(device)
        batch_size = real.size(0)
        label = tensor(np.ones((batch_size), dtype=np.float32))
        output = net_d(real).view(-1)
        err_d_real = loss_fn(output, label)
        err_d_real.backward()
        d_x = output.mean().item()

        # add fake batches
        noise = torch.randn(batch_size, 16, device=device)
        fake = net_g(noise)
        label.fill_(0.0)
        output = net_d(fake.detach()).view(-1)
        err_d_fake = loss_fn(output, label)
        err_d_fake.backward()
        d_g_z1 = output.mean().item()
        err_d = err_d_real + err_d_fake
        optimizer_d.step()

        # train generator
        net_g.zero_grad()
        label.fill_(1.0)
        output = net_d(fake).view(-1)
        err_g = loss_fn(output, label)
        err_g.backward()
        d_g_z2 = output.mean().item()
        optimizer_g.step()

        # output stats
        if batch % 50 == 0:
            print('[%d/%d]  Loss_D: %.4f  Loss_G: %.4f  D(x): %.4f  D(G(z)): %.4f / %.4f'
                  % (batch, len(dataloader),
                     err_d.item(), err_g.item(), d_x, d_g_z1, d_g_z2))


def main_train():
    # load data
    dataloader = load_data(False)

    # create generator and discriminator models
    net_g = Generator().to(device)
    net_g.apply(weights_init)
    if os.path.isfile("dcgan-g.pth"):
        net_g.load_state_dict(torch.load("dcgan-g.pth"))
    print(net_g)

    net_d = Discriminator().to(device)
    net_d.apply(weights_init)
    if os.path.isfile("dcgan-d.pth"):
        net_d.load_state_dict(torch.load("dcgan-d.pth"))
    print(net_d)

    if False:  # make sure the model has no error
        z = tensor(np.ones((1, 16), dtype=np.float32))
        print('z', z.shape)
        g = net_g(z)
        print('g', g.shape)
        d = net_d(g)
        print('d', d.shape)
        abort

    # loss function and optimizer
    loss_fn = nn.BCELoss()
    optimizer_g = torch.optim.Adam(net_g.parameters(),
                                   lr=0.001, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(net_d.parameters(),
                                   lr=0.0002, betas=(0.5, 0.999))

    # train with progress
    fixed_noise = torch.randn(64, 16, device=device)
    for epoch in range(1, 10+1):
        # train
        print("Epoch", epoch)
        train_epoch(dataloader, net_g, net_d, loss_fn, optimizer_g, optimizer_d)
        # plot
        generated = net_g(fixed_noise)
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Generated Images")
        plt.imshow(np.transpose(vutils.make_grid(
            generated,
            padding=2, pad_value=0.5, normalize=True).cpu(), (1, 2, 0)))
        plt.savefig("dcgan-output/epoch{:02d}".format(epoch))

    torch.save(net_g.state_dict(), "dcgan-g.pth")
    torch.save(net_d.state_dict(), "dcgan-d.pth")


def main_plot():
    net_g = Generator().to(device)
    net_g.load_state_dict(torch.load("dcgan-g.pth"))

    noise_frames = [torch.randn(64, 16, device=device) for _ in range(10)]

    def noise(t: float):
        nf = len(noise_frames)
        t = (t % 1.0) * nf
        i = int(t)
        j = (i + 1) % nf
        f = t - i
        # f = f*f*(3.-2.*f)
        return noise_frames[i]*(1.0-f)+noise_frames[j]*f

    def images(t: float):
        z = noise(t)
        generated = net_g(z)
        return np.transpose(vutils.make_grid(
            generated,
            padding=2, pad_value=0.5, normalize=True).cpu(), (1, 2, 0))

    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Generated Images")
    im = plt.imshow(images(0.0))

    def init():
        im.set_data(images(0.0))
        return [im]

    def animate(t):
        im.set_data(images(t))
        return [im]

    ani = animation.FuncAnimation(
        fig, animate, np.arange(1, 200)/200,
        init_func=init, interval=50, blit=True)
    plt.show()


if __name__ == "__main__":
    #main_train()
    main_plot()
