**Previous truss solver notes using the stiffness method:** https://github.com/harry7557558/DMOJ-Render_Main/blob/master/Moana/truss/notes.md (as a competitive programming problem that isn't intended to be published)

**For readers:** I use markdown notes like using scrap paper. You may find my thought process in my notes. You *may* not expect seeing concepts presented in a formal and organized way.

 - Side note: Some LaTeX may be broken on GitHub markdown preview. They should show up properly in vscode markdown preview because it's what I use to make notes.


# Stress/Strain/Force/Deflection

https://en.wikipedia.org/wiki/Hooke%27s_law#Isotropic_materials

Young's modulus $E$, Poisson ratio $\nu$. Write a stress tensor as $[\sigma_{11}\ \sigma_{22}\ \sigma_{33}\ \sigma_{23}\ \sigma_{13}\ \sigma_{12}]^T$ and a strain tensor as $[\epsilon_{11}\ \epsilon_{22}\ \epsilon_{33}\ 2\epsilon_{23}\ 2\epsilon_{13}\ 2\epsilon_{12}]^T$.

Strain to stress: $$\begin{bmatrix}\sigma_{11}\\\sigma_{22}\\\sigma_{33}\\\sigma_{23}\\\sigma_{13}\\\sigma_{12}\end{bmatrix} = \dfrac{E}{(1+\nu)(1-2\nu)}\begin{bmatrix}1-\nu&\nu&\nu&0&0&0\\\nu&1-\nu&\nu&0&0&0\\\nu&\nu&1-\nu&0&0&0\\0&0&0&\frac{1-2\nu}{2}&0&0\\0&0&0&0&\frac{1-2\nu}{2}&0\\0&0&0&0&0&\frac{1-2\nu}{2}\end{bmatrix} \begin{bmatrix}\epsilon_{11}\\\epsilon_{22}\\\epsilon_{33}\\2\epsilon_{23}\\2\epsilon_{13}\\2\epsilon_{12}\end{bmatrix}$$

$\sigma=C\epsilon$, $C$ is positive definite when $0<\nu<0.5$. (A higher strain won't result in a lower stress.)

Energy density: $U=\dfrac{1}{2}\epsilon^TC\epsilon$, $\dfrac{\partial U}{\partial \epsilon}=C\epsilon=\sigma$.

Force density: $F^T=\nabla U=\dfrac{\partial U}{\partial \epsilon}\dfrac{\partial\epsilon}{\partial x}=\sigma^T\dfrac{\partial\epsilon}{\partial x}$.
 - https://physics.stackexchange.com/a/525756 ? $F=-\nabla\cdot\sigma$

https://github.com/harry7557558/Graphics/blob/master/simulation/mass_spring/xpbd_cloth/notes-elasticity.ipynb

Green strain tensor: $\epsilon=\dfrac{1}{2}(\nabla u^T+\nabla u+\nabla u^T\nabla u)$; Linearize by dropping the last term for small $\nabla u$.

 - Independence of orientation: $u=Rx-x+u_0$, $\nabla u=R-I+\nabla u_0$, $\dfrac{1}{2}(\nabla u^T+\nabla u+\nabla u^T\nabla u)=\dfrac{1}{2}(R^T-I+\nabla u_0^T+R-I+\nabla u_0+R^TR+I+\nabla u_0^T\nabla u_0-R^T-R-\nabla u_0-\nabla u_0^T+R^T\nabla u_0+\nabla u_0^TR)=\dfrac{1}{2}\left((R\nabla u_0)^T+R\nabla u_0+(R\nabla u_0)^T(R\nabla u_0)\right)$

Objective: find stiffness $\dfrac{\partial F}{\partial u}=\dfrac{\partial F}{\partial\sigma}\dfrac{\partial\sigma}{\partial\epsilon}\dfrac{\partial\epsilon}{\partial u}$.

Analogous to a truss:
 - $u$ is truss deflection
 - $\epsilon$ is $\Delta L$
 - $\sigma$ is member force, $C$ is axial stiffness
 - $F$ is applied load

# Linear tetrahedral element

FEM elements: https://www.ccg.msm.cam.ac.uk/system/files/documents/FEMOR_Lecture_2.pdf `[lt1]`, https://academic.csuohio.edu/duffy_s/CVE_512_12.pdf `[lt2]`, https://www.valuedes.co.uk/blog/tet-vs-hex-handbags-at-dawn `[lt3]`

Vertices of a tetrahedron: $x_0,x_1,x_2,x_3$; After deformation: $x_i'=x_i+u_i$.

FEM: better solve for $u$ instead of $x+u$ because $u$ is much smaller compared to $x$ and we want to reduce rounding error.

Find $\nabla u$: solve $\cdot(\nabla u+I)\cdot(x_{i\ne0}-x_0)=(x_{i\ne0}'-x'_0)$

 - $x$ are row vectors; $x_{i\ne0}-x_0=[x_1-x_0;\ x_2-x_0;\ x_3-x_0]$

 - Let $X=x_{i\ne0}-x_0$, $\nabla u=X^{-1}((x_{i\ne0}-x_0)+(u_{i\ne0}-u_0))-I=X^{-1}(X+(u_{i\ne0}-u_0))-I=X^{-1}(u_{i\ne0}-u_0)$

   - Same as solving $\cdot\nabla u\cdot(x_{i\ne0}-x_0)=(u_{i\ne0}-u_0)$. I found the one starting from $x'$ intuitively makes more sense.

 - $\dfrac{\partial\nabla u}{\partial u}$: $\nabla u$ is 9-dimensional ($_{11}$, $_{12}$, $_{13}$, $_{21}$... $_{33}$), $u$ is 12-dimensional

   - Transform from 9-component vector $u_{i\ne0}-u_0$ to 9-component vector $\nabla u$: a 3x3 grid of 3x3 diagonal matrices, each corresponds to the corresponding element of $X^{-1}$ multiplied by $I_3$.

   - $\dfrac{\partial\nabla u}{\partial u_i}$: (0-indexed) column $3i$ to $3i+2$ for the 9x9 matrix

   - $\dfrac{\partial\nabla u}{\partial u_0} = \displaystyle -\left(\sum_{i=1}^{3}\frac{\partial\nabla u}{\partial u_i}\right)$

 - Is $\dfrac{\partial\nabla u}{\partial u}$ consistent when the order of vertices is changed? It would be a pain to manually verify it by expanding the matrix and finding the inverse. Might verify it numerically later.

 - $\dfrac{\partial\nabla u^T}{\partial u}$: swap the rows of $\dfrac{\partial\nabla u}{\partial u}$, row $3i+j$ becomes row $3j+i$.

Linearized strain $\epsilon=\dfrac{1}{2}(\nabla u+\nabla u^T)$

 - Let $D=\dfrac{\partial\nabla u}{\partial u}$. 0-indexing for $D,u$ and 1-indexing for $\epsilon$.

 - $\epsilon_{11}=D_0$, $\epsilon_{22}=D_4$, $\epsilon_{33}=D_8$

 - $2\epsilon_{23}=D_5+D_7$, $2\epsilon_{13}=D_2+D_6$, $2\epsilon_{12}=D_1+D_3$

 - $S=\begin{bmatrix}1&&&&&&&&\\&&&&1&&&&\\&&&&&&&&1\\&&&&&1&&1&\\&&1&&&&1&&\\&1&&1&&&&&\end{bmatrix}$: convert 9-component $\dfrac{\partial\nabla u}{\partial x}$ to 6-component $\epsilon$.

Stress-strain relationship: $\sigma=C\epsilon$.

$\dfrac{\partial F}{\partial\sigma}$

 - Use $S^T$ to convert a 6-component $\sigma$ back to a 9-component $\sigma$. (coincidence? I guess not.)

 - $F=-\nabla\cdot\sigma=-\left(\dfrac{\partial\sigma_1}{\partial x_1}+\dfrac{\partial\sigma_2}{\partial x_2}+\dfrac{\partial\sigma_3}{\partial x_3}\right)$

 - `[lt2]` states $F=D^T\sigma$?

   - Seems like $F$ is the force (not force per unit volume) distributed on the node, exerted by either the element or surface traction. ($X^{-T}$ involves dividing by volume)

 - Think how to calculate $\sigma$ from $F$? Doesn't work because divergence can't uniquely determine a function.

 - The force density equation derived from energy is $F=\left(\dfrac{\partial\epsilon}{\partial x}\right)^T\sigma$. $\dfrac{\partial\epsilon}{\partial x}$ is $SD$.
 
 - Multiply the force density by the volume of the tetrahedron to get forces at the joints

Result stiffness matrix: $F=VD^TS^TCSDu$

 - $V$: the volume of the tetrahedral element, a scalar, in $\mathrm{mm}^3$

 - $F$: the force (not force density) acting at joints, a 4x3=12-component vector, in $\mathrm{N}$

 - $u$: deflections, a 4x3=12-component vector, in $\mathrm{mm}$

 - $D=\dfrac{\partial\nabla u}{\partial u}$: a 9x12 matrix, in $\mathrm{mm}^{-1}$

 - $S$: 6x9, transforms a 9-component strain tensor to a 6-component strain; $S^T$ transforms a 6-component stress to a 9-component stress tensor

 - $C$: a positive definite matrix, transforming strain to stress, in $\mathrm{MPa}=\mathrm{N/mm^2}$

 - 9-component strain: $SDu$, dimensionless
 - 9-component stress: $S^TCSDu$, in $\mathrm{MPa}$

## Thoughts

Nonlinear (Green) strain tensor? The equation is no longer linear.

 - Nonlinear conjugate gradient? How to precondition it?

 - First Google search result for "nonlinear stiffness matrix:" https://www.hindawi.com/journals/jam/2014/932314/ (haven't read it yet)

 - In real-world: large deformation -> material is no longer linearly elastic?

   - Materials like concrete are already not linearly elastic for small deformations

   - Steel yields at $\epsilon=0.002$, $\epsilon^2=4\times10^{-6}$ is neglectable

   - ~~Counterclaim: It is about $\nabla u$, not $\epsilon$. $\nabla u$ can still be large when $\epsilon$ is small (ex. at the end of a cantilever with a high orientation change)~~
     - ~~$u_i-u_0$ remains consistent, intuition is wrong~~
       - Under orientation change, $\nabla u$ becomes "closer to" skew-symmetric, $\nabla u^T+\nabla u$ becomes smaller and $\nabla u^T\nabla u$ becomes relatively larger

   - Need to empirically test how big is the effect of $\nabla u^T\nabla u$ on $\epsilon$

Brick/prism/cone elements: splitting into tetrahedra and adding matrices up?

 - `[lt3]` shows brick elements perform better than tetrahedral elements, so it might not be that simple.

   - Might because of brick mesh generation produces a "smoother" result than a tetrahedral one? Does it only work better if the bricks are aligned in the principle components of the stress/strain tensors?
     - If this is the case, prism and cone elements aren't better than tetrahedral elements.

 - Calculate $\frac{\partial\nabla u}{\partial u}$: Finding different ways to split it and averaging the results? Least squares? Finding weights to minimize the error term in series expansion?

The stiffness matrix may be singular in the truss one. Intuitively, as long as not all fixed nodes are colinear and all nodes are connected by element faces, a force can't result in an "infinite" deformation and therefore the matrix is nonsigular. An element must have a positive volume, like a truss member must have a positive length.


# Quadratic tetrahedral element

Why better:
 - Force is the gradient of energy; Captures energy more accurately?
 - A quadratic element approximates much more linear elements?
 - Better captures nonlinearity?
 - Lower error order in series expansion? Might need to empirically test how the error changes as the size of elements change.

Interpolate and integrate the stiffness matrix across the volume? Difficult to integrate an expresion with matrix inverse analytically. Gaussian quadrature?

Interpolation: parameters $r,s,t$ to coordinates $x,y,z$.

 - Tri-quadratic interpolation?

 - At the vertices, $[r,s,t]$ have values $[0,0,0],[1,0,0],[0,1,0],[0,0,1]$. At the midpoints, $[r,s,t]$ have values $[0.5,0,0],[0,0.5,0],[0,0,0.5],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]$.

 - Quadrature: weighted sum of stiffness matrices at sample points?

https://help.febio.org/FEBio/FEBio_tm_2_7/FEBio_tm_2-7-Subsection-4.1.4.html

 - Parameters $t_0=1-t_1-t_2-t_3$, $t_1$, $t_2$, $t_3$

 - $x=[t_0(2t_0-1)\ \ t_1(2t_1-1)\ \ t_2(2t_2-1)\ \ t_3(2t_3-1)\ \ 4t_0t_1\ \ 4t_0t_2\ \ 4t_0t_3\ \ 4t_1t_2\ \ 4t_1t_3\ \ 4t_2t_3][x_{0}\ \ x_{1}\ \ x_{2}\ \ x_{3}\ \ x_{01}\ \ x_{02}\ \ x_{03}\ \ x_{12}\ \ x_{13}\ \ x_{23}]^T = t^TX$

 - For the three-vector matrix in calculating $\nabla u$: take the derivative to $t_1,t_2,t_3$.

   - Let $\nabla u=(WX)^{-1}Wu$, where $W$ is 3x10, $X$ and $u$ are 10x3

   - $(WX)^{-1}W$ is 3x10

   - Flatten $u$ into a 30-component vector, $D$ is $(WX)^{-1}W\otimes I_3$

 - Quadrature: choose the 11-point Gauss-Lobatto rule because it is independent of the order of vertices

   - Weights add up to $1/6$? Normalize it to $1$.

   - $(WX)^{-1}W$ is a sub-grid of $\otimes I_3$, the overall stiffness matrix isn't.

   - Volume: $\dfrac{1}{6}\det(WX)$

Calculating $W$:

```py
from sympy import *

t1, t2, t3 = symbols('t1 t2 t3')
t0 = 1 - (t1+t2+t3)
T = [t0*(2*t0-1), t1*(2*t1-1), t2*(2*t2-1), t3*(2*t3-1),
     4*t0*t1, 4*t0*t2, 4*t0*t3, 4*t1*t2, 4*t1*t3, 4*t2*t3]
W = []
for t in [t1, t2, t3]:
    W.append([diff(ti, t) for ti in T])

GL = [
    (0, 0, 0, 1/60),
    (1, 0, 0, 1/60),
    (0, 1, 0, 1/60),
    (0, 0, 1, 1/60),
    (0.5, 0, 0, 1/15),
    (0.5, 0.5, 0, 1/15),
    (0, 0.5, 0, 1/15),
    (0, 0, 0.5, 1/15),
    (0.5, 0, 0.5, 1/15),
    (0, 0.5, 0.5, 1/15),
    (0.25, 0.25, 0.25, 8/15)
]
assert abs(sum([p[3] for p in GL])-1) < 1e-12
print('{'+','.join([str(p[3]) for p in GL])+'}')
for t1, t2, t3, w in GL:
    s = []
    for wi in W:
        wi = [float(_.subs({'t1': t1, 't2': t2, 't3': t3})) for _ in wi]
        assert abs(sum(wi)) < 1e-12
        s.append('{'+','.join(["{:.12g}".format(_) for _ in wi])+'}')
    print('{'+', '.join(s)+'},')
```

```cpp
{0.016666666666666666,0.016666666666666666,0.016666666666666666,0.016666666666666666,0.06666666666666667,0.06666666666666667,0.06666666666666667,0.06666666666666667,0.06666666666666667,0.06666666666666667,0.5333333333333333}
{{-3,-1,0,0,4,0,0,0,0,0}, {-3,0,-1,0,0,4,0,0,0,0}, {-3,0,0,-1,0,0,4,0,0,0}},
{{1,3,0,0,-4,0,0,0,0,0}, {1,0,-1,0,-4,0,0,4,0,0}, {1,0,0,-1,-4,0,0,0,4,0}},
{{1,-1,0,0,0,-4,0,4,0,0}, {1,0,3,0,0,-4,0,0,0,0}, {1,0,0,-1,0,-4,0,0,0,4}},
{{1,-1,0,0,0,0,-4,0,4,0}, {1,0,-1,0,0,0,-4,0,0,4}, {1,0,0,3,0,0,-4,0,0,0}},
{{-1,1,0,0,0,0,0,0,0,0}, {-1,0,-1,0,-2,2,0,2,0,0}, {-1,0,0,-1,-2,0,2,0,2,0}},
{{1,1,0,0,-2,-2,0,2,0,0}, {1,0,1,0,-2,-2,0,2,0,0}, {1,0,0,-1,-2,-2,0,0,2,2}},
{{-1,-1,0,0,2,-2,0,2,0,0}, {-1,0,1,0,0,0,0,0,0,0}, {-1,0,0,-1,0,-2,2,0,0,2}},
{{-1,-1,0,0,2,0,-2,0,2,0}, {-1,0,-1,0,0,2,-2,0,0,2}, {-1,0,0,1,0,0,0,0,0,0}},
{{1,1,0,0,-2,0,-2,0,2,0}, {1,0,-1,0,-2,0,-2,2,0,2}, {1,0,0,1,-2,0,-2,0,2,0}},
{{1,-1,0,0,0,-2,-2,2,2,0}, {1,0,1,0,0,-2,-2,0,0,2}, {1,0,0,1,0,-2,-2,0,0,2}},
{{0,0,0,0,0,-1,-1,1,1,0}, {0,0,0,0,-1,0,-1,1,0,1}, {0,0,0,0,-1,-1,0,0,1,1}},
```


## Linear brick element (8 nodes)

https://help.febio.org/FEBio/FEBio_tm_2_7/FEBio_tm_2-7-Subsection-4.1.1.html

 - $[x_{000},x_{100},x_{110},x_{010},x_{001},x_{101},x_{111},x_{011}]^T$

 - Parameters: $r,s,t$, from $-1$ to $1$, $x_{\frac{1+r}{2},\frac{1+s}{2},\frac{1+t}{2}}$

 - Volume: $\dfrac{\partial X}{\partial uvw}\dfrac{\partial uvw}{\partial xyz}=8\cdot\dfrac{\partial X}{\partial uvw}$

```py
from sympy import *

r, s, t = symbols('r s t')
T = [(1-r)*(1-s)*(1-t)/8,
     (1+r)*(1-s)*(1-t)/8,
     (1+r)*(1+s)*(1-t)/8,
     (1-r)*(1+s)*(1-t)/8,
     (1-r)*(1-s)*(1+t)/8,
     (1+r)*(1-s)*(1+t)/8,
     (1+r)*(1+s)*(1+t)/8,
     (1-r)*(1+s)*(1+t)/8]
assert sum(T).equals(1)
W = []
for t in [r, s, t]:
    W.append([diff(ti, t) for ti in T])

c = sqrt(3)/3
GL = [
    (-c, -c, -c, 1/8),
    (c, -c, -c, 1/8),
    (c, c, -c, 1/8),
    (-c, c, -c, 1/8),
    (-c, -c, c, 1/8),
    (c, -c, c, 1/8),
    (c, c, c, 1/8),
    (-c, c, c, 1/8)
]
assert abs(sum([p[3] for p in GL])-1) < 1e-12
print('{'+','.join([str(p[3]) for p in GL])+'}')
for r, s, t, w in GL:
    c = []
    for wi in W:
        wi = [float(_.subs({'r': r, 's': s, 't': t})) for _ in wi]
        assert abs(sum(wi)) < 1e-12
        c.append('{'+','.join([str(_) for _ in wi])+'}')
    print('{'+', '.join(c)+'},')
```

Won't include the program output here because it's pretty long.


## Quadratic brick element (20 nodes)

The last slide of `[lt2]`.

https://www.fidelisfea.com/post/what-are-shape-functions-in-fea-and-how-are-they-derived


# Element force to joint force

Linear triangle: $F/3$
 - Splitting a rectangle into two triangles gives uneven force distribution on vertices?!
   - A "good" mesh matters in FEM?

Linear tetrahedron: $F/4$

Quadratic triangle (assume linear initially)
 - Each vertex node is $(F/4)/3=F/12$
 - Each edge node is $F/12\times3=F/4$
 - Check: $F/12\times3+F/4\times3=F$

Quadratic tetrahedron (assume linear initially)
 - Each vertex node is $(F/8)/4=F/32$
 - After cutting four "vertices", the remaining octahedron has a force of $F/2$, distributed equally on $6$ vertices ($F/12$)
 - Each edge node: $F/32\times2+F/12=7F/48$
 - Check: $1/32\times4+7/48\times6=1$

Linear quadrilateral
 - Force isn't distributed equally
 - CCW vertices $x_0,x_1,x_2,x_3$
 - Bilinear interpolation, split edges in the middle, split the patch into $4$ faces
 - Estimate the area of each patch in the middle: $||x_u'\times x_v'||$
   - Middle might not be the best location? Too lazy to do an analysis using series expansion.
   - Intuitively, the cross product isn't zero for non-colinear $x$ and $u,v\notin\{0,1\}$
 - Weight forces by areas

```py
from sympy import *

u, v = symbols('u v')
T = [(1-u)*(1-v), u*(1-v), u*v, (1-u)*v]
assert sum(T).equals(1)

Tu = [diff(t, u) for t in T]
Tv = [diff(t, v) for t in T]

uv = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]
cu, cv = [], []
for u, v in uv:
    tu = [float(t.subs({'u': u, 'v': v})) for t in Tu]
    tv = [float(t.subs({'u': u, 'v': v})) for t in Tv]
    cu.append('{'+','.join(["{:.12g}".format(_) for _ in tu])+'}')
    cv.append('{'+','.join(["{:.12g}".format(_) for _ in tv])+'}')
print('{'+', '.join(cu)+'},')
print('{'+', '.join(cv)+'},')
```

Linear brick: similar to linear quadrilteral, split into $8$ volumes and use the determinant of three vectors instead of the cross product norm.

```py
from sympy import *

u, v, w = symbols('u v w')
T = [(1-u)*(1-v)*(1-w), u*(1-v)*(1-w), u*v*(1-w), (1-u)*v*(1-w),
     (1-u)*(1-v)*w, u*(1-v)*w, u*v*w, (1-u)*v*w]
assert sum(T).equals(1)

Tu = [diff(t, u) for t in T]
Tv = [diff(t, v) for t in T]
Tw = [diff(t, w) for t in T]

uvw = [(0.25, 0.25, 0.25), (0.75, 0.25, 0.25),
       (0.75, 0.75, 0.25), (0.25, 0.75, 0.25),
       (0.25, 0.25, 0.75), (0.75, 0.25, 0.75),
       (0.75, 0.75, 0.75), (0.25, 0.75, 0.25)]
cu, cv, cw = [], [], []
for u, v, w in uvw:
    sub = {'u': u, 'v': v, 'w': w}
    tu = [float(t.subs(sub)) for t in Tu]
    tv = [float(t.subs(sub)) for t in Tv]
    tw = [float(t.subs(sub)) for t in Tw]
    cu.append('{'+','.join(["{:.12g}".format(_) for _ in tu])+'}')
    cv.append('{'+','.join(["{:.12g}".format(_) for _ in tv])+'}')
    cw.append('{'+','.join(["{:.12g}".format(_) for _ in tw])+'}')
print('{'+', '.join(cu)+'},')
print('{'+', '.join(cv)+'},')
print('{'+', '.join(cw)+'},')
```


# Tetrahedral mesh generation

https://graphics.stanford.edu/papers/meshing-sig03/meshing.pdf
 - Marching crystaline lattice + subdivide to level of details + compress to fit the boundary
 - Suggests Delaunay is bad for 3D??
 - Drawbacks: results in not-so-regular tetrahedral elements; no guarantee to fit the boundary exactly

https://arxiv.org/ftp/arxiv/papers/0911/0911.3884.pdf
 - Marching cubes + advancing front + Delaunay refinement + optimize (Laplacian smoothing, etc.)

A table of index of mesh generation techniques: https://people.eecs.berkeley.edu/~jrs/mesh/

https://people.eecs.berkeley.edu/~jrs/papers/LabellePhD.pdf

 - Mesh quality
   - Interpolation error depends on $l_{max}$, regular tetrahedra are the best
   - ~0 degrees is not so bad, ~180 degrees is very bad because it makes the matrix near-singular

 - Marching tetrahedra/cubes + isosurface stuffing + Delaunay refinement

**Goal 1: generate a coarse tetrahedral volume mesh from a polygon surface mesh.**
 - Eliminate the idea of marching + stuffing because it doesn't represent the boundary exactly
 - Initial mesh: uniform element size, choose the element size based on the size of the structure and the "thinnest" part

Idea 1:
 - Generate "uniform" edge and surface points
 - BFS-like advancing front; need a data structure to quickly find neighbors and a way to determine the "best" place for the next point
 - Delaunay
 - Optimize (Laplacian smoothing, etc.)
   - Optimization for vertex positions may be done before Delaunay

Idea 2:
 - Generate a surface mesh, starting from edges and then advance front in 2D
 - Advancing front in the interior of the mesh; Add both vertices and elements; Need a data structure for fast collision detection
   - Need to guarantee no overlapping at where fronts join together
 - Optimize (edge flip, Laplacian smoothing, reducing a loss function, etc.)

**Goal 2: refine the mesh based on the FEM result on the coarse mesh**
 - Decrease element size at locations with high variations of stress/strain
   - Reduce approximation error to a certain amount
 - Idea: subdivide the mesh and then optimize it

