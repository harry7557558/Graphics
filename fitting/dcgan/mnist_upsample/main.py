# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# trained on MNIST, use upsample instead of convtrans

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


# False: larger, visually better model (7832)
# True: smaller model to be exported to GLSL (1700)
SMALL_MODEL = False
MODEL_G_PATH = "dcgan-g-s.pth" if SMALL_MODEL else "dcgan-g.pth"
MODEL_D_PATH = "dcgan-d-s.pth" if SMALL_MODEL else "dcgan-d.pth"
IMAGE_PATH = "dcgan-output-s" if SMALL_MODEL else "dcgan-output"

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
            padding=2, pad_value=0.5, normalize=True).cpu(), (1, 2, 0)),
                   interpolation='nearest')
        plt.show()
    return dataloader


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        # https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
        dims = [4, 8, 4] if SMALL_MODEL else [16, 16, 8]
        layers = [
            nn.Unflatten(1, (16, 1, 1)),
            # 16 x 1x1
            nn.Conv2d(16, dims[0], 4, 1, 3, bias=False),
            None if SMALL_MODEL else nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.1, inplace=True),
            # dims[0] x 4x4
            nn.Upsample(scale_factor=2),
            nn.Conv2d(dims[0], dims[1], 3, 1, 1, bias=False),
            None if SMALL_MODEL else nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.1, inplace=True),
            # dims[1] x 8x8
            nn.Upsample(scale_factor=2),
            nn.Conv2d(dims[1], dims[2], 3, 1, 0, bias=False),
            None if SMALL_MODEL else nn.BatchNorm2d(dims[2]),
            nn.LeakyReLU(0.1, inplace=True),
            # dims[2] x 14x14
            nn.Upsample(scale_factor=2),
            nn.Conv2d(dims[2], 1, 5, 1, 2, bias=False),
            nn.Sigmoid()
            # 1 x 28x28
        ]
        layers = [layer for layer in layers if layer is not None]
        self.main = nn.Sequential(*layers)

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
            # 16 x 14x14
            nn.Conv2d(16, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 x 7x7
            nn.Conv2d(16, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 x 4x4
            nn.Conv2d(16, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


def weights_init(m):
    classname = m.__class__.__name__
    if 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif 'Linear' in classname:
        if 'weight' in m.__dict__ and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 1.0)
        if 'bias' in m.__dict__ and m.bias is not None:
            nn.init.normal_(m.bias.data, 0.0, 1.0)
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
        output_d_real = net_d(real).view(-1)
        err_d_real = loss_fn(output_d_real, label)
        err_d_real.backward()

        # add fake batches
        noise = torch.randn(batch_size, 16, device=device)
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
        err_g = loss_fn(output_g, label)
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
    dataloader = load_data(False)

    # create generator and discriminator models
    net_g = Generator().to(device)
    net_g.apply(weights_init)
    if os.path.isfile(MODEL_G_PATH):
        net_g.load_state_dict(torch.load(MODEL_G_PATH))
    print("Generator", count_weights(net_g))
    print(net_g)

    net_d = Discriminator().to(device)
    net_d.apply(weights_init)
    if os.path.isfile(MODEL_D_PATH):
        net_d.load_state_dict(torch.load(MODEL_D_PATH))
    print("Discriminator", count_weights(net_d))
    print(net_d)

    if False:  # make sure the model has no error
        z = tensor(np.ones((1, 16), dtype=np.float32))
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
            padding=2, pad_value=0.5, normalize=True).cpu(), (1, 2, 0)),
                   interpolation='nearest')
        plt.savefig(IMAGE_PATH+"/epoch{:02d}".format(epoch))

    torch.save(net_g.state_dict(), MODEL_G_PATH)
    torch.save(net_d.state_dict(), MODEL_D_PATH)


def export_weights_glsl(model, names):
    lines = []
    lines.append("// Paste auto-generated weights here\n")
    for (name, layer) in zip(names, model.parameters()):
        k = 1000
        # shape comment
        shape = list(layer.data.shape)
        lines.append(f"// Shape = {shape}")
        # hard-code data
        data = layer.data.flatten().numpy()
        n = len(data)
        data = np.round(k*data).astype(np.int32)
        data = ','.join(map(str, data.tolist()))
        lines.append(f"int {name}[{n}] = int[{n}]({data});")
        # get data
        funname = 'get' + name.capitalize()
        if len(shape) == 4:
            i = f"d+{shape[3]}*(c+{shape[2]}*(b+{shape[1]}*a))"
            lines.append(f"float {funname}(int a, int b, int c, int d)" \
                        + f"{{ return {1/k}*float({name}[{i}]); }}")
        elif len(shape) == 1:
            lines.append(f"float {funname}(int i)" \
                         + f"{{ return {1/k}*float({name}[i]); }}")
        # new line
        lines.append('')
    print('\n'.join(lines))
    #sys.exit(0)


def main_plot():
    net_g = Generator().to(device)
    net_g.load_state_dict(torch.load(MODEL_G_PATH))
    print("Model -", count_weights(net_g), "weights")
    print(net_g)
    export_weights_glsl(net_g, [
        'conv1w', 'bn1g', 'bn1b',
        'conv2w', 'bn2g', 'bn2b',
        'conv3w', 'bn3g', 'bn3b',
        'conv4w'
    ])

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
