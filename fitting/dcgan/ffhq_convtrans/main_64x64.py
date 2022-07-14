# 64x64 GAN trained on FFHQ

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


LATENT = 32
DIMS_G = [32, 64, 32, 16]
DIMS_D = [16, 32, 64, 32]
MODEL_G_PATH = "ffhq-64x64-g.pth"
MODEL_D_PATH = "ffhq-64x64-d.pth"
EXPORT_PATH = "raw-weights-ffhq-64x64"
if False:
    MODEL_G_PATH = "anime-64x64-g.pth"
    MODEL_D_PATH = "anime-64x64-d.pth"
    EXPORT_PATH = "raw-weights-anime-64x64"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


class FFHQDataset(Dataset):

    def __init__(self, transform=None, target_transform=None):
        self.size = 64
        self.path = f"../data/FFHQ/{self.size}x{self.size}/"
        self.raw_bytes = {}
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        #return 10000
        return 70000

    def __getitem__(self, i):
        bd = i - i % 1000
        filepath = self.path + "{:05d}.raw".format(bd)
        if filepath not in self.raw_bytes:
            content = np.fromfile(filepath, dtype=np.uint8)
            content = content.reshape((1000, 3, self.size, self.size))
            self.raw_bytes[filepath] = content
        else:
            content = self.raw_bytes[filepath]
        return content[i%1000].astype(np.float32) / 255.0


class AnimeFaceDataset(Dataset):

    def __init__(self, transform=None, target_transform=None):
        self.size = 64
        self.path = "../data/AnimeFace/64x64.raw"
        raw_bytes = np.fromfile(self.path, dtype=np.uint8)
        self.content = raw_bytes.reshape((60000, 3, 64, 64))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        #return 10000
        return 60000

    def __getitem__(self, i):
        return self.content[i%1000].astype(np.float32) / 255.0


def load_data(dataset, plot=False):
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True
    )
    for x in dataloader:
        print("Shape of x:", x.shape)
        break
    if plot:
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(
            next(iter(dataloader)).to(device)[:64],
            padding=2, pad_value=0.5, normalize=True).cpu(), (1, 2, 0)),
                   interpolation='nearest')
        plt.show()
    return dataloader


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        # ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        # https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
        dims = DIMS_G
        layers = [
            # LATENT
            nn.Linear(LATENT, dims[0]*4*4, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Unflatten(1, (dims[0], 4, 4)),
            # dims[0] x 4x4
            nn.ConvTranspose2d(dims[0], dims[1], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            # dims[1] x 8x8
            nn.ConvTranspose2d(dims[1], dims[2], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            # dims[2] x 16x16
            nn.ConvTranspose2d(dims[2], dims[3], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            # dims[3] x 32x32
            nn.ConvTranspose2d(dims[3], 3, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # 3 x 64x64
        ]
        layers = [layer for layer in layers if layer is not None]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        dims = DIMS_D
        self.main = nn.Sequential(
            # 3 x 64x64
            nn.Conv2d(3, dims[0], 4, 2, 1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, inplace=True),
            # dims[0] x 32x32
            nn.Conv2d(dims[0], dims[1], 4, 2, 1, bias=False),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, inplace=True),
            # dims[1] x 16x16
            nn.Conv2d(dims[1], dims[2], 4, 2, 1, bias=False),
            nn.BatchNorm2d(dims[2]),
            nn.LeakyReLU(0.2, inplace=True),
            # dims[2] x 8x8
            nn.Conv2d(dims[2], dims[3], 4, 2, 1, bias=False),
            nn.BatchNorm2d(dims[3]),
            nn.LeakyReLU(0.2, inplace=True),
            # dims[3] x 4x4
            nn.Conv2d(dims[3], 1, 4, 1, 0, bias=False),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


def weights_init(m):
    classname = m.__class__.__name__
    if 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.normal_(m.bias.data, 0.0, 0.02)
    elif 'Linear' in classname:
        if 'weight' in m.__dict__ and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.1)
        if 'bias' in m.__dict__ and m.bias is not None:
            nn.init.normal_(m.bias.data, 0.0, 0.1)
    else:
        if 'weight' in m.__dict__ and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if 'bias' in m.__dict__ and m.bias is not None:
            nn.init.normal_(m.bias.data, 0.0, 0.02)


def train_epoch(dataloader, net_g, net_d, loss_fn, optimizer_g, optimizer_d):
    for batch, x in enumerate(dataloader, 0):

        # add real batches
        net_d.zero_grad()
        real = x.to(device)
        batch_size = real.size(0)
        label = torch.ones((batch_size), device=device)
        output_d_real = net_d(real).view(-1)
        err_d_real = loss_fn(output_d_real, label)
        err_d_real.backward()

        # add fake batches
        noise = torch.randn(batch_size, LATENT, device=device)
        fake = net_g(noise)
        label.fill_(0.0)
        output_d_fake = net_d(fake.detach()).view(-1)
        err_d_fake = loss_fn(output_d_fake, label)
        err_d_fake.backward()
        optimizer_d.step()

        # train generator
        net_g.zero_grad()
        label.fill_(1.0)
        output_g = net_d(fake).view(-1)
        #err_g = loss_fn(output_g, label)
        err_g = 1.0-loss_fn(1.0-output_g, label)
        err_g.backward()
        optimizer_g.step()

        # output stats
        if (batch+1) % 50 == 0:
            err_d = err_d_real + err_d_fake
            d_x = output_d_real.mean().item()
            d_g_z1 = output_d_fake.mean().item()
            d_g_z2 = output_g.mean().item()
            print('[%d/%d]  Loss_D: %.4f  Loss_G: %.4f  D(x): %.4f  D(G(z)): %.4f / %.4f'
                  % (batch+1, len(dataloader),
                     err_d.item(), err_g.item(), d_x, d_g_z1, d_g_z2))


def count_weights(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main_train():
    # load data
    dataloader = load_data(FFHQDataset())

    # create generator and discriminator models
    net_g = Generator().to(device)
    print("Generator", count_weights(net_g))
    print(net_g)

    net_d = Discriminator().to(device)
    print("Discriminator", count_weights(net_d))
    print(net_d)

    try:
        net_g.load_state_dict(torch.load(MODEL_G_PATH, map_location=device))
        net_d.load_state_dict(torch.load(MODEL_D_PATH, map_location=device))
        print("Model loaded from file.")
    except BaseException as e:
        net_g.apply(weights_init)
        net_d.apply(weights_init)
        print("Model weights initialized.")

    if False:  # make sure the model has no error
        z = tensor(np.ones((1, LATENT), dtype=np.float32))
        print('z', z.shape)
        g = net_g(z)
        print('g', g.shape)
        d = net_d(g)
        print('d', d.shape)
        sys.exit(0)

    # loss function and optimizer
    loss_fn = nn.BCELoss()
    optimizer_g = torch.optim.Adam(net_g.parameters(),
                                   lr=0.001, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(net_d.parameters(),
                                   lr=0.0002, betas=(0.5, 0.999))

    # train with progress
    fixed_noise = torch.randn(64, LATENT, device=device)
    for epoch in range(1, 10+1):
        # train
        print("Epoch", epoch)
        train_epoch(dataloader, net_g, net_d, loss_fn, optimizer_g, optimizer_d)
        # plot
        if epoch % 1 == 0:
            generated = net_g(fixed_noise)
            plt.figure(figsize=(8, 8))
            plt.axis("off")
            plt.title("Generated Images")
            plt.imshow(np.transpose(vutils.make_grid(
                generated,
                padding=2, pad_value=0.5, normalize=True).cpu(), (1, 2, 0)),
                       interpolation='nearest')
            plt.show() # on ipynb

    torch.save(net_g.state_dict(), MODEL_G_PATH)
    torch.save(net_d.state_dict(), MODEL_D_PATH)


def main_plot():
    net_g = Generator().to(device)
    net_g.load_state_dict(torch.load(MODEL_G_PATH, map_location=device))
    print("Model -", count_weights(net_g), "weights")
    print(net_g)

    # export weights
    param_i = 0
    for param in net_g.parameters():
        param_i += 1
        data = param.data
        shape = '_'.join(map(str, data.shape))
        path = EXPORT_PATH+"/w{:02d}_{}.bin".format(param_i, shape)
        data.numpy().astype(np.float32).tofile(path)

    # animation
    nframe = 10
    u1 = np.random.random((nframe, 64, LATENT))
    u2 = np.random.random((nframe, 64, LATENT))

    def noise(t: float):
        t = (t % 1.0) * nframe
        i = int(t)
        j = (i + 1) % nframe
        f = t - i
        v1 = u1[i]*(1.0-f)+u1[j]*f
        v2 = u2[i]*(1.0-f)+u2[j]*f
        return np.sqrt(-2.0*np.log(1.0-v1)) * np.sin(6.283185*v2)

    def images(t: float):
        z = tensor(noise(t).astype(np.float32), device=device)
        generated = net_g(z)
        return np.transpose(vutils.make_grid(
            generated,
            padding=2, pad_value=0.5, normalize=True).cpu(), (1, 2, 0))

    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Generated Images")
    im = plt.imshow(images(0.0), interpolation='nearest')

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
    #load_data(AnimeFaceDataset(), True)
    #main_train()
    main_plot()
