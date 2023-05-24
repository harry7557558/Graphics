# Generate 3D model from transparent PNG

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from PIL import Image
import matplotlib.pyplot as plt

# https://josan-coba.github.io/svg-to-png/
name = "fire"  # filename of the 3D model
filename = "fire.png"  # image
res = 64  # mesh resolution
zscale = 0.8  # lower -> flatter

# load image
img0 = Image.open(filename)
img = img0.resize((res, res))
img = np.array(img) / 255.0
if img.shape[2] != 4:
    print('error: transparent png required')


# mesh

print('generating 2d mesh...')

xys = []
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        c = img[img.shape[1]-j-1][i]
        if c[3] > 0.0:
            xys.append((i, j))
xyss = dict(zip(xys, range(len(xys))))
trigs = []
for i in range(img.shape[0]-1):
    for j in range(img.shape[1]-1):
        v = [(i, j), (i, j+1), (i+1, j), (i+1, j+1)]
        if (i+j)&1:
            v = [v[2], v[0], v[3], v[1]]
        ts = [
            (v[0], v[3], v[1]),
            (v[0], v[2], v[3])
        ]
        for a, b, c in ts:
            try:
                trigs.append([
                    xyss[a], xyss[b], xyss[c]
                ])
            except KeyError:
                pass

# remove all boundary trigs
edges = {}
for t in trigs:
    for e in zip(t, [t[1], t[2], t[0]]):
        e = frozenset(e)
        if e not in edges:
            edges[e] = 0
        edges[e] += 1
isb = [0]*len(xys)
for e, c in edges.items():
    if c != 2:
        for i in e:
            isb[i] = 1
trigs = [t for t in trigs if
         not (isb[t[0]] and isb[t[1]] and isb[t[2]])]

# remove unused vertices
vmap = [-1]*len(xys)
for t in trigs:
    for i in t:
        vmap[i] = 1
count = 0
for i in range(len(vmap)):
    if vmap[i] == 1:
        vmap[i] = count
        count += 1
xys = [xys[i] for i in range(len(xys)) if vmap[i] != -1]
for i in range(len(trigs)):
    for j in range(3):
        trigs[i][j] = vmap[trigs[i][j]]

# edges and boundary
edges = {}
for t in trigs:
    for e in zip(t, [t[1], t[2], t[0]]):
        e = frozenset(e)
        if e not in edges:
            edges[e] = 0
        edges[e] += 1
xys = np.array(xys)
uvs = np.divide(xys+0.5, [img.shape[1], img.shape[0]])
xys = np.divide(xys+0.5, [img.shape[1], img.shape[0]])*2.0-1.0
xys[:, 1] *= img0.size[1] / img0.size[0]
bmap = [-1]*len(xys)
for e, v in edges.items():
    if v == 1:
        for i in e:
            bmap[i] = 1
    else:
        assert v == 2
count = 0
for i in range(len(bmap)):
    if bmap[i] == -1:
        bmap[i] = count
        count += 1
    else:
        bmap[i] = -1
nbn = count  # non boundary count
edges = [list(e) for e in edges]

# smooth mesh
for step in [0.5, -0.2] * 20:
    lap = 0.0*xys
    counts = [0]*len(xys)
    for i, j in edges:
        d = xys[j]-xys[i]
        if not (bmap[i] == -1 and bmap[j] != -1):
            lap[i] += d
            counts[i] += 1
        if not (bmap[j] == -1 and bmap[i] != -1):
            lap[j] -= d
            counts[j] += 1
    lap /= np.transpose([counts, counts])
    xys += step*lap


# FEM
# https://alecjacobson.com/weblog/media/notes-on-inflating-curves-2009-baran.pdf

print('inflating...')

A = []
for t0 in trigs:
    t = t0[:]
    # https://graphics.stanford.edu/courses/cs468-13-spring/assets/lecture12-lu.pdf
    def cot(v0, v1, v2):
        cos = np.dot(v1-v0, v2-v0) / (np.linalg.norm(v1-v0) * np.linalg.norm(v2-v0))
        return cos / (1-cos**2)**0.5
    for _ in range(3):
        i, j1, j2 = t[_], t[(_+1)%3], t[(_+2)%3]
        if bmap[i] == -1:
            continue
        v0, v1, v2 = xys[i], xys[j1], xys[j2]
        alpha = cot(v1, v0, v2)
        beta = cot(v2, v0, v1)
        w = (alpha+beta) / (np.linalg.det([v1-v0,v2-v0])*6.0)
        i, j1, j2 = bmap[i], bmap[j1], bmap[j2]
        A += [(i, i, 2*w)]
        if j1 != -1:
            A += [
                (i, j1, -w),
                (j1, i, -w),
                (j1, j1, w)
            ]
        if j2 != -1:
            A += [
                (i, j2, -w),
                (j2, i, -w),
                (j2, j2, w)
            ]
A = scipy.sparse.csr_matrix(
        ([a[2] for a in A], ([a[0] for a in A], [a[1] for a in A])),
        shape=(nbn, nbn))
sol = scipy.sparse.linalg.cg(A, 4.0*np.ones(nbn).T)
if sol[1] != 0:
    print('error')
#print(sol[0])

ans = [0.0]*len(xys)
for i in range(len(xys)):
    if bmap[i] != -1:
        ans[i] = zscale * sol[0][bmap[i]] ** 0.5

#for e in edges: plt.plot([xys[e[0]][0], xys[e[1]][0]], [xys[e[0]][1], xys[e[1]][1]])
#plt.scatter(xys[:, 0], xys[:, 1], c=ans)
#plt.axis('equal'); plt.show()


# generate 3d model

print('generating 3d model...')

# two pieces
n0 = len(bmap)
points = [[xys[i][0], xys[i][1], ans[i]] for i in range(n0)]
for i in range(len(bmap)):
    if bmap[i] == -1:
        continue
    points.append([points[i][0], points[i][1], -points[i][2]])
uvis = trigs[:]
for ti in range(len(trigs)):
    t = trigs[ti][:]
    t[0], t[1] = t[1], t[0]
    uvis.append(t[:])
    for i in range(3):
        if bmap[t[i]] != -1:
            t[i] = n0 + bmap[t[i]]
    trigs.append(t)
points = np.array(points)

# smoothing
edges = set()
for t in trigs:
    for e in zip(t, [t[1], t[2], t[0]]):
        edges.add(frozenset(e))
edges = [list(e) for e in edges]
for step in [0.2] * 10:
    lap = 0.0*points
    counts = [0]*len(points)
    for i, j in edges:
        d = points[j]-points[i]
        lap[i] += d
        counts[i] += 1
        lap[j] -= d
        counts[j] += 1
    lap /= np.transpose([counts, counts, counts])
    points += step * lap

# normals
normals = 0.0*points
for t in trigs:
    v = [points[i] for i in t]
    n = np.cross(v[1]-v[0], v[2]-v[0])
    n /= np.linalg.norm(n)
    for i in t:
        normals[i] += n
for i in range(len(points)):
    normals[i] /= np.linalg.norm(normals[i])

# save OBJ
lines = [
    f"mtllib {name}.mtl",
    f"o {name}",
]
for i in range(len(points)):
    lines.append("v {:.6f} {:.6f} {:.6f}".format(
        points[i][0], points[i][1], points[i][2]))
for i in range(len(uvs)):
    lines.append("vt {:.6f} {:.6f}".format(
        uvs[i][0], uvs[i][1]))
for i in range(len(normals)):
    lines.append("vn {:.6f} {:.6f} {:.6f}".format(
        normals[i][0], normals[i][1], normals[i][2]))
lines += [
    "usemtl Default_OBJ",
    "s 1"
]
for (t, uvi) in zip(trigs, uvis):
    lines.append("f {}/{}/{} {}/{}/{} {}/{}/{}".format(
        t[0]+1, uvi[0]+1, t[0]+1,
        t[1]+1, uvi[1]+1, t[1]+1,
        t[2]+1, uvi[2]+1, t[2]+1))
open(f"{name}.obj", "w").write('\n'.join(lines))

lines = [
    "newmtl Default_OBJ",
    "Ns 200.000000",
    "Ka 1.0 1.0 1.0",
    "Kd 0.8 0.8 0.8",
    "Ks 0.5 0.5 0.5",
    "Ni 1.45",
    "d 1.0",
    "illum 2",
    f"map_Kd {name}_texture.png",
    ""
]
open(f"{name}.mtl", "w").write('\n'.join(lines))


# generate texture, fix transparent pixels

print('generating texture...')

pixels = np.array(img0, dtype=np.uint8)
pixels, alphas = pixels[:, :, 0:3], pixels[:, :, 3]//192
visited = []
w, h = alphas.shape
for i in range(w):
    for j in range(h):
        if alphas[i][j] == 1:
            visited.append((i, j))
i0, j0 = 0, len(visited)
while i0 < j0:
    def test(i, j, pixel):
        if i < 0 or j < 0 or i >= w or j >= h:
            return
        if alphas[i][j] == 1:
            return
        visited.append((i, j))
        alphas[i][j] = 1
        pixels[i][j] = pixel
    for i, j in visited[i0:j0]:
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                test(i+di, j+dj, pixels[i][j])
    i0, j0 = j0, len(visited)
Image.fromarray(pixels).save(f"{name}_texture.png")

print('done.')
