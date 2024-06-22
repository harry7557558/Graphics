## bezier_fitting_test

0: fit image contour, optimize sum of closest distances, recursive split + merge close pieces
- 0.1, 0.2: getting things working
- 0.3: attempt to optimize split location for fixed number of pieces

1: fit noise-free parametric continuous curves, optimize integrated squared error
- 1.1: generate dense spline from points, problem formulation, L-BFGS optimization
- 1.2: optimize using a least squares solver, struggles with split locations
