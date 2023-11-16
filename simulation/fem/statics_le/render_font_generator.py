# Generate pixelized fonts for rendering the GUI

import numpy as np
from PIL import Image


def load_font(filename: str):
    image = Image.open(filename)
    pixels = np.asarray(image).astype(np.int32)
    pixels = pixels[64:, 32:]
    assert pixels.shape == (4096, 4096)

    font = []
    for code in range(65536):
        row = 16*(code//256)
        col = 16*(code%256)
        subimg = pixels[row:row+16, col:col+16]
        font.append(1-subimg)

    return font


if __name__ == "__main__":
    font = load_font("unifont-15.0.01.bmp")

    chars = "0123456789.e+-στuFxyzvmtc"

    for ci in range(len(chars)):
        char = chars[ci]
        block = font[ord(char)][:16, :8]
        ds = []
        for i in range(4):
            x = 0
            bits = np.concatenate(block[4*i:4*i+4][::-1])
            for b in bits:
                x = (x << 1) | b
            ds.append(str(x))
        print(f'if (i=={ci}) c=ivec4(' + ','.join(ds) + '); // ' + char)
