import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import model_export_code


def load_data(file, shape):
    sz = np.prod(shape)
    arr = np.fromfile(file, dtype=np.uint8)[:sz].reshape(shape)
    arr = arr / 255.0
    z = np.linspace(-1.0, 1.0, shape[0]).repeat(shape[1]*shape[2])
    y = np.array([np.linspace(-1.0, 1.0, shape[1]).repeat(shape[2]),]*shape[0]).reshape((sz,))
    x = np.array([np.linspace(-1.0, 1.0, shape[2]),]*(shape[0]*shape[1])).reshape((sz,))
    X = np.array([x, y, z]).T.astype(np.float32)
    Y = arr.reshape(sz).astype(np.float32)
    return (X, Y)


def plot_data(X, Y):
    indices = np.random.randint(0, len(Y), 3000)
    plt.clf()
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(X[indices,0], X[indices,1], X[indices,2], c=Y[indices])
    plt.show()


def create_model():
    act = tf.sin
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, activation=act, input_shape=(3,)),
        tf.keras.layers.Dense(16, activation=act),
        tf.keras.layers.Dense(16, activation=act),
        tf.keras.layers.Dense(16, activation=act),
        tf.keras.layers.Dense(1)
    ])
    weights = open('tooth_weights.txt').read().strip()
    if weights != "":
        weights = weights.replace('{','[').replace('}',']').replace(';', '')
        weights = [w[w.find('=')+1:] for w in weights.split('\n')]
        weights = [np.array(eval(w)) for w in weights]
        for i in range(len(weights)//2):
            model.layers[i].set_weights((weights[2*i], weights[2*i+1]))
    model.compile(
        #optimizer=tf.keras.optimizers.SGD(learning_rate=0.03, momentum=0.9),
        optimizer='adam',
        loss='mean_squared_error', metrics=['accuracy'])
    return model


def export_model_code(model):
    layers = model.layers
    weights = [[a.tolist() for a in layer.get_weights()] for layer in layers]
    output = ""
    for i in range(len(weights)):
        w, b = weights[i]
        output += f'const float layer{i+1}_weights[{len(w)}][{len(w[0])}] =' + \
              model_export_code.arr_to_str_2d(w, '{', '}', 6) + ';\n'
        output += f'const float layer{i+1}_biases[{len(b)}] =' + \
              model_export_code.arr_to_str_1d(b, '{', '}', 6) + ';\n'
    print(output)
    print(model_export_code.export_glsl(weights))


if __name__ == '__main__':

    X, Y = load_data("tooth_100x94x160_uint8.raw", (160, 94, 100))
    #plot_data(X, Y)

    model = create_model()

    N = 200
    for i in range(N):
        model.fit(X, Y, epochs=10, batch_size=len(Y)//10, verbose=0)
        print(f"{i+1}/{N}", model.evaluate(X, Y, batch_size=len(X)//100, verbose=0)[0])

    export_model_code(model)
