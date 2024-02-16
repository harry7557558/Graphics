import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'

def conv3(nin, nout):
    return nn.Conv2d(nin, nout, 3, padding=1, padding_mode='reflect')

def relu(x):
    return F.relu(x, inplace=True)


class ResidualBlock(nn.Module):
    def __init__(self, n):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3(n, n)
        self.conv2 = conv3(n, n)
    
    def forward(self, x):
        return self.conv2(relu(self.conv1(x))) + x


class Modelx4(torch.nn.Module):
    def __init__(self, n_in, n_out, n_hidden, n_residual):
        super(Modelx4, self).__init__()

        self.conv_i = conv3(n_in, n_hidden)
        self.residual = nn.Sequential(
            *[ResidualBlock(n_hidden) for _ in range(n_residual)]
        )
        self.conv_m = conv3(n_hidden, n_hidden)
        self.upscale = nn.Sequential(
            conv3(n_hidden, 4*n_hidden),
            nn.ReLU(),
            nn.PixelShuffle(2),
            conv3(n_hidden, 4*n_hidden),
            nn.ReLU(),
            nn.PixelShuffle(2)
        )
        # self.conv_o = nn.Conv2d(n_hidden, n_out, 7, padding=3)
        self.conv_o = conv3(n_hidden, n_out)
    
    def forward(self, x):
        x_i = relu(self.conv_i(x))
        x_r = self.residual(x_i)
        x_m = self.conv_m(x_r) + x_i
        x_u = self.upscale(x_m)
        x_o = torch.sigmoid(self.conv_o(x_u))
        return x_o


model = Modelx4(3, 3, 32, 10).to(device)
model = torch.load("model01_x4.pth").to(device)

model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

from PIL import Image
filename = "test/full-00747.png"
x = Image.open(filename).convert("RGB")
#x = x.resize((x.size[0]//2, x.size[1]//2), Image.Resampling.BICUBIC)
x = np.array(x, dtype=np.float32) / 255.0
x = np.transpose(x, (2, 0, 1))

x = torch.tensor(x, device=device).unsqueeze(0)
print(x.shape)
y = torch.zeros((1, x.shape[1], 4*x.shape[2], 4*x.shape[3]),
                dtype=torch.float32, device=device)

tile = 512
with torch.no_grad():
    for i in range(0, x.shape[2], tile):
        for j in range(0, x.shape[3], tile):
            di = min(tile, x.shape[2]-i)
            dj = min(tile, x.shape[3]-j)
            block = x[:, :, i:i+di, j:j+dj]
            upscaled = model(block)
            y[:, :, 4*i:4*(i+di), 4*j:4*(j+dj)] = upscaled
print(y.shape)

y = y[0].cpu().numpy()
y = np.transpose(y, (1, 2, 0))
y = (y*255).astype(np.uint8)
Image.fromarray(y).save("test/x4.jpg")
