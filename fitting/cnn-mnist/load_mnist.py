import numpy as np
from keras.datasets import mnist

(train_x, train_y), (test_x, test_y) = mnist.load_data()

assert train_x.shape == (60000, 28, 28)
assert test_x.shape == (10000, 28, 28)
assert train_x.dtype == np.ubyte
assert test_x.dtype == np.ubyte

all_x = np.concatenate((train_x, test_x))
all_y = np.concatenate((train_y, test_y))

indices = np.arange(len(all_x))
np.random.shuffle(indices)
all_x = all_x[indices]
all_y = all_y[indices]

train_x.tofile("bin/train_x.bin")
test_x.tofile("bin/test_x.bin")
train_y.astype(np.uint8).tofile("bin/train_y.bin")
test_y.astype(np.uint8).tofile("bin/test_y.bin")
all_x.astype(np.uint8).tofile("bin/all_x.bin")
all_y.astype(np.uint8).tofile("bin/all_y.bin")

open("bin/.gitignore", 'w').write("*")
