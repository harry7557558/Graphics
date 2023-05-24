Numerical algorithms used in computer graphics.

Used by code in this repository. Add `Graphics` folder in the compiler additional include directory settings.

Avoid object orientation for simplicity and performance.

Files in the `debug` directory are used for debugging header files.

--------

Some of the following headers include other headers.

[rootfinding.h](rootfinding.h): find the roots of functions;

[optimization.h](optimization.h): find the minimum of functions; numerical differentiation;

[geometry.h](geometry.h): 2d/3d vector, 3d matrix; misc macros/functions;

[random.h](random.h): floating-point random number generator;

[interpolation.h](interpolation.h): generate interpolated or fitted functions on discrete sample points;

[linearsystem.h](linearsystem.h): matrixes as 1d arrays of length N×N; Gaussian elimination;

[eigensystem.h](eigensystem.h): find the eigenvalues/eigenpairs of real symmetric matrices;

[integral.h](integral.h): numerical integration of functions;

[ode.h](ode.h): numerical ordinary differential equation solutions;

To-do: fft.h, complex.h

