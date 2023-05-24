# Process the FFHQ dataset to raw file of different sizes
# Images are manually downloaded from Google Drive and unzipped to `/raw`

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
from torch import tensor
import torchvision.utils as vutils


def plot_images(images):
    images = images[:40]
    plt.figure(figsize=(8, 5))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(
            tensor(images/255.0),
            padding=2, pad_value=1.0, normalize=True).cpu(), (1, 2, 0)),
               interpolation='nearest')
    plt.show()


for batch in range(0, 70000, 1000):
    bd = "{:05d}".format(batch)
    print(bd)

    # loade images
    images = []
    for i in range(batch, batch+1000):
        path = "raw/{}/{:05d}.png".format(bd, i)
        img = np.array(Image.open(path))
        assert img.shape == (128, 128, 3)
        img = np.einsum('ijk->kij', img)
        images.append(img)
    images = np.array(images, dtype=np.uint8)

    # save 128x128 raw file
    path = f"128x128/{bd}.raw"
    images.tofile(path)

    # save 64x64 raw file
    images = images.reshape((len(images), 3, 64, 2, 64, 2))
    images = images.mean((3, 5))
    path = f"64x64/{bd}.raw"
    (images+0.5).astype(np.uint8).tofile(path)

    # save 32x32 raw file
    images = images.reshape((len(images), 3, 32, 2, 32, 2))
    images = images.mean((3, 5))
    #plot_images(images)
    path = f"32x32/{bd}.raw"
    (images+0.5).astype(np.uint8).tofile(path)

    #break
