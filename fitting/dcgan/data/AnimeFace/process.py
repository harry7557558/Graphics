# Process data downloaded from
# https://www.kaggle.com/datasets/splcher/animefacedataset

# Generate 64x64 and 32x32 raw files


import os
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
from torch import tensor
import torchvision.utils as vutils


def load_images(image_n: int):
    path = "images/"
    files = [path+fn for fn in os.listdir(path)]
    raw_images = [None] * len(files)
    for i in range(len(files)):
        with open(files[i], "rb") as fp:
            image = Image.open(fp)
            raw_images[i] = (
                image.size[0]*image.size[1],
                np.array(image, dtype=np.uint8)
            )
        if (i+1)%1000 == 0:
            print(i+1)
    raw_images.sort(key=lambda a: -a[0])
    raw_images = raw_images[::-1]  # why?
    images = []
    for i in range(min(len(raw_images), image_n)):
        image = Image.fromarray(raw_images[i][1])
        image = image.resize((64, 64))
        image = np.array(image, dtype=np.uint8)
        image = np.einsum('ijc->cij', image)
        images.append(image)
    return np.array(images, dtype=np.uint8)


def plot_images(images):
    images = images[:40]
    plt.figure(figsize=(8, 5))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(
            tensor(images/255.0),
            padding=2, pad_value=1.0, normalize=True).cpu(), (1, 2, 0)),
               interpolation='nearest')
    plt.show()


if __name__ == "__main__":
    images = load_images(60000)
    #plot_images(images)

    # save 64x64 raw file
    images.tofile("64x64.raw")

    # save 32x32 raw file
    images = images.reshape((len(images), 3, 32, 2, 32, 2))
    images = images.mean((3, 5))
    (images+0.5).astype(np.uint8).tofile("32x32.raw")
