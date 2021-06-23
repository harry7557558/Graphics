import numpy as np


def hermite_spline_matrix():
    """
        c(t) = [t³ t² t 1] ⋅ A ⋅ [p₀ p₁ p'₀ p'₁]ᵀ
        Cubic spline determined by starting point, ending point, starting tangent, ending tangent
    """
    M = np.array([
        [0, 0, 0, 1],
        [1, 1, 1, 1],
        [0, 0, 1, 0],
        [3, 2, 1, 0]
    ])
    return np.linalg.inv(M)


def catmull_rom_spline_matrix():
    """
        c(t) = [t³ t² t 1] ⋅ A ⋅ [p₀ p₁ p₂ p₃]ᵀ
        Hermite spline from p₁ to p₂, with c'(0)=(p₂-p₀)/2 and c'(1)=(p₃-p₁)/2
    """
    A_hermite = hermite_spline_matrix()
    M = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [-1/2, 0, 1/2, 0],
        [0, -1/2, 0, 1/2]
    ])
    return np.matmul(A_hermite, M)


def quintic_hermite_spline_matrix():
    """
        c(t) = [t⁵ t⁴ t³ t² t 1] ⋅ A ⋅ [p₀ p₁ p'₀ p'₁ p"₀ p"₁]ᵀ
        Quintic spline determined by starting/ending points and up to second order derivatives
    """
    M = np.array([
        [0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 0],
        [5, 4, 3, 2, 1, 0],
        [0, 0, 0, 2, 0, 0],
        [20, 12, 6, 2, 0, 0]
    ])
    return np.linalg.inv(M)


if __name__ == "__main__":
    print(quintic_hermite_spline_matrix())
