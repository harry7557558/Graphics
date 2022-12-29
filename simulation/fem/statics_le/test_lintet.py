# Test a linear tetrahedral element

import numpy as np
np.random.seed(0)
np.set_printoptions(precision=2, linewidth=150)

# setup element
X = np.array([[-1, -1, -1], [1, 1, -1], [-1, 1, 1], [1, -1, 1]])  # regular
X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 90-45-45
#X = np.random.normal(size=(4, 3))  # random
invX = np.linalg.inv([X[1]-X[0], X[2]-X[0], X[3]-X[0]])
print(invX)


# numerical gradient

def delU(U):
    U = U.reshape((4, 3))
    dUdX = invX @ [U[1]-U[0], U[2]-U[0], U[3]-U[0]]
    return dUdX.reshape(9)
    #return dUdX.T.reshape(9)

def epsilon(U):
    U = U.reshape((4, 3))
    dUdX = invX @ [U[1]-U[0], U[2]-U[0], U[3]-U[0]]
    e = 0.5 * (dUdX + dUdX.T)
    return np.array([e[0][0], e[1][1], e[2][2],
                     2*e[1][2], 2*e[0][2], 2*e[0][1]])

U = np.random.random(12)
dGdU = np.zeros((9, 12))
dEdU = np.zeros((6, 12))
for i in range(12):
    h = 0.0001
    e = np.zeros(12)
    e[i] = h
    dGdU[:, i] = (delU(U+e)-delU(U-e))/(2*h)
    dEdU[:, i] = (epsilon(U+e)-epsilon(U-e))/(2*h)

print(dGdU)
print(dEdU)


# stiffness matrix

E, nu = 2e5, 0.33
C = np.array([
    [1-nu, nu, nu, 0, 0, 0],
    [nu, 1-nu, nu, 0, 0, 0],
    [nu, nu, 1-nu, 0, 0, 0],
    [0, 0, 0, 0.5-nu, 0, 0],
    [0, 0, 0, 0, 0.5-nu, 0],
    [0, 0, 0, 0, 0, 0.5-nu]
]) * E / ((1+nu)*(1-2*nu))
V = 1.0 / abs(6.0*np.linalg.det(invX))
print(np.linalg.eigvalsh(C))
#C = np.eye(6)

dFdU = V * dEdU.T @ C @ dEdU
print(dFdU)
eigs = np.sort(np.linalg.eigvalsh(dFdU))
print(eigs)  # 6 zeros

print(eigs[-1]/eigs[6])
# regular tetrahedron: 3.911764705882357
# 90-45-45: 9.105020506433476
# random: mostly 50-300, some >10000


# preconditioning

def cond(A):
    eigs = np.sort(np.linalg.eigvalsh(A))
    cond1 = eigs[-1] / eigs[0]
    cond2 = np.linalg.norm(A)*np.linalg.norm(np.linalg.inv(A))
    return (cond1, cond2)

# no preconditioning
i = (1, 2, 3, 4, 5, 7)
dFdU = np.array([[dFdU[a][b] for a in i] for b in i])
print(dFdU)
print(cond(dFdU))

# diagonal preconditioning, no much improvement
diag = np.diag(np.diag(dFdU)**-0.5)
precond = diag @ dFdU @ diag
print(precond)
print(cond(precond))

# block diagonal preconditioning, good for bad meshes
precond = np.zeros((6, 6))
precond[0:3, 0:3] = np.linalg.cholesky(dFdU[0:3, 0:3])
precond[3:6, 3:6] = np.linalg.cholesky(dFdU[3:6, 3:6])
precond = np.linalg.inv(precond)
precond = precond @ dFdU @ precond.T
print(precond)
print(cond(precond))
