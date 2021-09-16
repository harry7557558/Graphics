import numpy as np
import tensorflow as tf
from model_export_code import *


LAYER_SIZE = 80


class RBFLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        self.units = LAYER_SIZE
        self.initializer = tf.keras.initializers.RandomUniform
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.center = self.add_weight(
            name='center',
            shape=(self.units, input_shape[1]),
            initializer=self.initializer,
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        center = tf.expand_dims(self.center, -1)
        dx = tf.transpose(center - tf.transpose(inputs))
        return tf.sqrt(tf.math.reduce_sum(dx**2, axis=1))


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(3,)),
        RBFLayer(),
        tf.keras.layers.Dense(units=1, activation='linear'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['accuracy'],
    )
    return model


def save_weights(model):
    w0 = str(model.layers[0].get_weights()).replace('\n', '').replace(' ', '')
    w1 = str(model.layers[1].get_weights()).replace('\n', '').replace(' ', '')
    fp = open("bunny_weights.txt", "w")
    fp.write(w0+'\n')
    fp.write(w1+'\n')
    fp.close()


def load_weights(model):
    fr = open("bunny_weights.txt", "r").read().strip().split('\n')
    w0 = eval(fr[0].replace('array', 'np.array').replace('float', 'np.float'))
    w1 = eval(fr[1].replace('array', 'np.array').replace('float', 'np.float'))
    model.layers[0].set_weights(w0)
    model.layers[1].set_weights(w1)
    return model


def export_model(model):
    s = num2str(model.layers[1].weights[1].numpy()[0])
    for i in range(LAYER_SIZE):
        k = num2str(model.layers[1].weights[0].numpy()[i][0], signed=True)
        center = vec2str(model.layers[0].weights[0].numpy()[i])
        s += "{}*r{}".format(k, center.lstrip('vec3'))
    return s


if __name__ == "__main__":
    tf.random.set_seed(1)

    data = np.fromfile('bunny_train.raw', dtype=np.float32)
    data = data.reshape((len(data)//4, 4))
    X = data[:, 0:3]
    Y = data[:, 3:4]

    model = create_model()
    try:
        model = load_weights(model)
    except BaseException as e:
        print("Failed to load weights:", e)

    N = 100
    for i in range(N):
        batch_size = len(data)//100
        model.fit(X, Y, epochs=16, batch_size=batch_size, verbose=0)
        err = model.evaluate(X, Y, batch_size=batch_size, verbose=0)[0]
        print(f"{i+1}/{N}", np.sqrt(err))

    save_weights(model)
    s = export_model(model)
    print(s)
