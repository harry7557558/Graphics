Inspired by neural bunnies on Shadertoy, the motivation of this is to find "better" fits to binary images and implicitly defined 3D models.

Instead of fitting the SDF, I use `tanh` for the output activation function. It is like a binary classification task, where `-1.0` is inside and `1.0` is outside, and the represented object is the zero-isosurface of the output.

Timeline:

[e_1d_cg.py](e_1d_cg.py): Starts from a Desmos graph that takes forever to minimize a function with 4 scalar parameters. Input is one-dimensional. Defines two loss functions where one is easier to minimize on startup and another is more accurate. Minimized using `scipy.optimize.minimize` with either `CG` or `BFGS` that takes less than one second.

[e_2d_binimg.py](e_2d_binimg.py): Fits two dimensional non-antialiased binary images. The model is the weighted sum of a list of basis functions. Compares polynomial and trigonometric basis functions. Minimizes the loss function using the BFGS algorithm implemented in SciPy.

[e_3d_bunny.py](e_3d_bunny.py): Fits the "Stanford Bunny" with polynomial or trigonometric basis functions. Takes about 10 minutes to fit.

[rbf_2d_binimg.py](rbf_2d_binimg.py): Experiments with radial basis functions in regression. Fits the SDF without activation function before fitting the model. Performs poorly due to local minima. With an attempt for determining the initial guess using simulated annealing.

[rbf_3d_bunny.py](rbf_3d_bunny.py): Fits the bunny using RBF in 3D. Takes more than one hour to fit. Produces a loss lower than using trigonometric basis functions with a similar number of scalar weights, but I suspect it can do better when optimized using using genetic algorithm + mini-batch gradient descent.

[sgd_test_1.py](sgd_test_1.py): Experiments with techniques to optimize the weights in fitting a bunny using trigonometric basis functions. SGD algorithms perform well on startup but stuck in a "valley." Stochastic CG and Newton-Raphson are unstable. This experiment isn't successful, but the Adam algorithm may be used for future tasks.

[bfgs_test_1.py](bfgs_test_1.py): Copy SciPy BFGS code out and add some modifications so I can see the progress without using callback.

[trig_3d_bunny.py](trig_3d_bunny.py) and [trig_3d_bunny.cpp](trig_3d_bunny.cpp): Trigonometric basis functions with variable frequencies. Try to see if C++ is faster than Python. (Answer: slightly.) The C++ BFGS function seems to have a bug because it takes twice as many iterations to converge as the SciPy one.

[bunny_neural_1.cpp](bunny_neural_1.cpp): Neural network with 1 hidden layer. The hidden layer has an activation function. Tried three different activation functions, `sin` best, then `tanh`, ReLU is very fast but very terrible.

[bunny_neural_2.cpp](bunny_neural_1.cpp): Neural network with 2 hidden layers. The model has an unexpectedly well performance with the `sin` activation function, but the `tanh` activation function doesn't have a significant improvement compared to using one hidden layer.

[tooth_neural_3.cpp](tooth_neural_3.cpp), [tooth_neural_3.glsl](tooth_neural_3.glsl): Volume of a tooth encoded in a neural network with 3 hidden layers (sin, sin, tanh) that outputs a scalar. Trained 4 models with different number of weights in each hidden layer.
