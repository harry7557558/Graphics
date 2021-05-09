
from PIL import Image


def load_image(filename):
    # load pixels
    img = Image.open(filename)
    img = img.crop(img.getbbox())
    w, h = img.size
    pixels = img.load()
    # take color the average on vertical direction
    vals = []
    for i in range(w):
        s = (0, 0, 0)
        for j in range(h):
            c = pixels[i, j]
            s = (s[0]+c[0], s[1]+c[1], s[2]+c[2])
        vals.append((s[0]/(255.*h), s[1]/(255.*h), s[2]/(255.*h)))
    # return colors as a list of rgb tuples
    return vals
