import tensorflow as tf
import numpy as np


TRAIN_SIZE = 1000
LAYER_SIZE = 12


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
        self.smoothness = self.add_weight(
            name='smoothness',
            shape=(self.units, input_shape[1]),
            initializer=self.initializer,
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        center = tf.expand_dims(self.center, -1)
        smoothness = tf.expand_dims(self.smoothness, -1)
        dx = tf.transpose(center - tf.transpose(inputs))
        dy = tf.transpose(smoothness)
        return 1.0/tf.sqrt(tf.math.reduce_sum(dx**2+dy**2, axis=1))


def train_data():
    # generate x and y data for training
    x = np.linspace(-5.0, 5.0, TRAIN_SIZE).reshape((TRAIN_SIZE, 1))
    y = 0.1*np.log(6.0*np.exp(0.1*x**3)+np.exp(10.0*np.cos(np.pi*x)-1.2*x))
    #y = 0.1*x*x
    return (x, y)


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(1,)),
        RBFLayer(),
        tf.keras.layers.Dense(units=1, activation='linear'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['accuracy'],
    )
    return model


def export_model(model):
    s = str(model.layers[1].weights[1].numpy()[0])
    for i in range(LAYER_SIZE):
        k = str(model.layers[1].weights[0].numpy()[i][0])
        center = str(model.layers[0].weights[0].numpy()[i][0])
        smoothness = str(model.layers[0].weights[1].numpy()[i][0]).lstrip('-')
        s += "+({})((x-({}))^{{2}}+{}^{{2}})^{{-0.5}}".format(k, center, smoothness)
    return s


if __name__ == "__main__":
    tf.random.set_seed(1)

    X, Y = train_data()
    model = create_model()

    N = 200
    for i in range(N):
        batch_size = TRAIN_SIZE//100
        model.fit(X, Y, epochs=16, batch_size=batch_size, verbose=0)
        err = model.evaluate(X, Y, batch_size=batch_size, verbose=0)[0]
        print(f"{i+1}/{N}", np.sqrt(err))

    s = export_model(model)
    print(s)
