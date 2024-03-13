import torch
import numpy as np

from oidn.tza import Reader
from oidn.model import UNet

import imageio.v3 as iio


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model():
    model = UNet(3, 3)
    weights = Reader("oidn/rt_alb.tza")

    for key in weights._table:
        tensor, layout = weights[key]
        layer, wb = key.split('.')
        weight = torch.from_numpy(np.array(tensor)).float()
        weight = torch.nn.Parameter(weight)
        setattr(getattr(model, layer), wb, weight)

    model = model.to(device)
    return model


def load_image(filename):
    ext = filename.split('.')[-1]
    if ext.lower() in ['exr', 'hdr']:
        img = iio.imread(filename) ** (1.0/2.2)
    else:
        img = iio.imread(filename)[:, :, :3].astype(np.float32) / 255.0
    w, h = img.shape[:2]
    if w%16 != 0 or h%16 != 0:
        print(f"Warning: Image `{filename}` clipped because its dimension is not a multiple of 16.")
        img = img[:(w//16)*16, :(h//16)*16, :]
    img = img.reshape(1, *img.shape)
    return torch.tensor(img, device=device).permute(0, 3, 1, 2)


def write_image(filename, img):
    img = img[0].detach().cpu().permute(1, 2, 0).numpy()
    ext = filename.split('.')[-1]
    if ext.lower() in ['exr', 'hdr']:
        img = np.maximum(img, 0.0)
        iio.imwrite(filename, img**2.2)
    else:
        img = np.clip(img, 0.0, 1.0)
        iio.imwrite(filename, (img*255).astype(np.uint8))
