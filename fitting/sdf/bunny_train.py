import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import model_export_code

data = np.fromfile('bunny_train.raw', dtype=np.float32)
data = data.reshape((len(data)//4, 4))
X = data[:, 0:3]
Y = data[:, 3:4]


def plot_data(data):
    plt.clf()
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=data[:, 3])
    plt.show()


def create_model():
    act = 'relu'
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, activation=act, input_shape=(3,)),
        tf.keras.layers.Dense(16, activation=act),
        tf.keras.layers.Dense(16, activation=act),
        tf.keras.layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.03, momentum=0.9),
        loss='mean_squared_error', metrics=['accuracy'])
    return model


def fit_model(model, X, Y):
    model.fit(X, Y, epochs=10, batch_size=len(X)//100, verbose=0)


def export_model_code(model):
    layers = model.layers
    weights = [[a.tolist() for a in layer.get_weights()] for layer in layers]
    for i in range(len(weights)):
        w, b = weights[i]
        print(f'const float layer{i+1}_weights[{len(w)}][{len(w[0])}] =',
              model_export_code.arr_to_str_2d(w, '{', '}') + ';')
        print(f'const float layer{i+1}_biases[{len(b)}] =',
              model_export_code.arr_to_str_1d(b, '{', '}') + ';')


if __name__ == '__main__':

    model = create_model()

    for i in range(100):
        fit_model(model, X, Y)
        print(model.evaluate(X, Y, batch_size=len(X)//100)[0])

    export_model_code(model)
