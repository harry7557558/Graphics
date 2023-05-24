# Generate marching tetrahedra lookup table
import numpy as np
from itertools import combinations


Edges = list(map(frozenset, [
    (0,), (1,), (2,), (3,),
    (0, 1), (0, 2), (0, 3),
    (1, 2), (1, 3), (2, 3)
]))
EdgesI = dict(zip(Edges, range(10)))
Vertices = [
    (-2, -2, -2), (2, 2, -2), (-2, 2, 2), (2, -2, 2),
    (0, 0, -2), (-2, 0, 0), (0, -2, 0),
    (0, 2, 0), (2, 0, 0), (0, 0, 2)
]
VerticesI = dict(zip(Vertices, range(10)))
Faces = [
    (1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)
]
Planes = [
    (1, 2, 3, 7, 9, 8),
    (0, 3, 2, 6, 9, 5),
    (0, 1, 3, 4, 8, 6),
    (0, 2, 1, 5, 7, 4),
]


def tetrahedron_volume(vs):
    vs = [np.array(Vertices[i]) for i in vs]
    return round(np.linalg.det([
        vs[1]-vs[0], vs[2]-vs[0], vs[3]-vs[0]
    ]))


def is_tets_overlap(t1, t2):
    """Check if two tets overlap
        Both tets must follow the vertice order convention"""
    assert tetrahedron_volume(t1) > 0 and tetrahedron_volume(t2) > 0
    t1 = [np.array(Vertices[i]) for i in t1]
    t2 = [np.array(Vertices[i]) for i in t2]

    def is_overlap(t1, t2):
        for fi in range(len(Faces)):
            f = [t1[Faces[fi][_]] for _ in range(3)]
            n = np.cross(f[1]-f[0], f[2]-f[0])
            assert np.dot(n, t1[fi]-f[0]) < 0
            has_negative = False
            for v in t2:
                d = np.dot(n, v-f[0])
                if d < 0:
                    has_negative = True
                    break
            if not has_negative:
                return False
        return True

    #return is_overlap(t1, t2)
    return is_overlap(t1, t2) and is_overlap(t2, t1)
                


def dfs_combinations(tets, f2ti, is_overlap, fcount, current, found, checked):
    """Called by `find_tetrahedra_combinations`
        @fcount: the number of times a face is used
        @current: the current (incomplete) set[int] of tets
        @found: a list of found combinations
        @checked: dp cache
    """
    fcurrent = frozenset(current)
    if fcurrent in checked:
        return
    checked.add(fcurrent)
    # check recursively while figuring out whether its complete
    complete = True
    for cti in current:
        tc = tets[cti]
        for fc in combinations(tc, 3):
            fc = frozenset(fc)
            if fcount[fc] not in [1, 2]:
                print(fc)
                for c in current:
                    print(c, tets[c])
                assert False
            if fcount[fc] == 2:
                continue
            for sti in f2ti[fc]:
                if sti in current:
                    continue
                # must not overlap with existing tets
                non_overlap = True
                for tti in current:
                    if is_overlap[sti][tti]:
                        non_overlap = False
                        break
                if not non_overlap:
                    continue
                # add the tet to the list
                complete = False
                ts = tets[sti]
                current.add(sti)
                for fs in combinations(ts, 3):
                    fcount[frozenset(fs)] += 1
                # recursion
                dfs_combinations(tets, f2ti, is_overlap,
                                 fcount, current, found, checked)
                # remove the tet from the list
                current.remove(sti)
                for fs in combinations(ts, 3):
                    fcount[frozenset(fs)] -= 1
    # if complete - add to the found list
    if complete:
        found.append(fcurrent)


def find_tetrahedra_combinations(es):
    """Find all ways to fill a mesh of a subdivided tet
        @bs: a set of subdivided edge indices
    """
    es = set(es).union(set([0, 1, 2, 3]))

    # find all possible tetrahedra
    tets = []
    for c in combinations(es, 4):
        v = tetrahedron_volume(c)
        if v < 0:
            c = (c[0], c[1], c[3], c[2])
        # must not be coplanar
        if v == 0:
            if True:
                continue
            good = True
            for plane in Planes:
                if len(set(c).difference(set(plane))) == 0:
                    good = False
                    break
            if not good:
                continue
        # edges must not overlap
        good = True
        for v1, v2 in combinations(c, 2):
            if v1 < 4 and v2 < 4:
                key = frozenset((v1, v2))
                if EdgesI[key] in es:
                    good = False
                    break
        if not good:
            continue
        # add to the list
        tets.append(c)

    # generate a map from faces to tetrahedra list
    f2ti = {}
    for i in range(len(tets)):
        for c in combinations(tets[i], 3):
            c = frozenset(c)
            if c not in f2ti:
                f2ti[c] = []
            f2ti[c].append(i)

    # generate a table of whether tets overlap
    is_overlap = []
    for i in range(len(tets)):
        is_overlap.append([
            is_tets_overlap(tets[i], tets[j])
            for j in range(len(tets))])

    #print(np.array(is_overlap).astype(np.int32))
    #print(tets)
    #print(f2ti)

    # DFS
    founds = set()
    checked = set()
    for i in range(len(tets)):
        fcount = dict(zip(f2ti.keys(), [0]*len(f2ti)))
        for fs in combinations(tets[i], 3):
            fcount[frozenset(fs)] += 1
        current = set([i])
        found = []
        dfs_combinations(tets, f2ti, is_overlap,
                         fcount, current, found, checked)
        for f in found:
            founds.add(f)
    V0 = tetrahedron_volume((0, 1, 2, 3))
    res = []
    for f in founds:
        ts = [tets[i] for i in f]
        #print(' '.join(map(str, ts)))
        assert sum([tetrahedron_volume(t) for t in ts]) == V0
        res.append(ts)
    #print(len(founds))
    return res


def find_tetrahedra_combinations_ss(ss):
    """Based on vertex signs for marching tetrahedra
        @ss: a set of vertices with negative signs"""
    ss = set(ss)
    # generate edge list
    es = []
    for i in range(len(Edges)):
        e = tuple(Edges[i])
        if len(e) > 1 and ((e[0] in ss) ^ (e[1] in ss)):
            es.append(i)
    # filter tets within the shape
    tets = set()
    for ts in find_tetrahedra_combinations(es):
        # must not contain a positive vertex
        ts1 = []
        for t in ts:
            is_out = False
            for vi in t:
                if vi < 4 and vi not in ss:
                    is_out = True
                    break
            if not is_out:
                ts1.append(t)
        if len(ts1) == 0:
            continue
        if frozenset(ts1) in tets:
            continue
        # must contain isosurfaces
        cts = []
        for t in ts1:
            for f in combinations(t, 3):
                if f[0] < 4 or f[1] < 4 or f[2] < 4:
                    continue
                assert f[0] in es and f[1] in es and f[2] in es
                cts.append(f)
        assert (len(es), len(cts)) in [(3, 1), (4, 2), (0, 0)]
        # add to the list
        assert len(ts1) in [1, 3]
        if len(ts1) != 0:
            tets.add(frozenset(ts1))
    tets = [list(ts) for ts in tets]
    assert len(tets) in [0, 1, 6]
    #print(tets)
    return tets


def list_faces(tets):
    faces = set()
    for tet in tets:
        for f in Faces:
            f = [tet[fi] for fi in f]
            found = False
            for plane in Planes:
                if len(set(f).difference(set(plane))) == 0:
                    found = True
                    break
            if not found:
                continue
            f = [list(plane).index(vi) for vi in f]
            i = np.argmin(f)
            f = (f[i], f[(i+1)%3], f[(i+2)%3])
            faces.add(f)
    return faces


def generate_march_lut():
    """Print LUT for marching tetrahedra"""
    lut = []
    for idx in range(2**4):
        ss = [i for i in range(4) if (idx>>i)&1]
        combs, combss = [], []
        for tets in find_tetrahedra_combinations_ss(ss):
            combs.append(tets)
            tets = sum(map(list, tets), [])
            if len(tets) < 12:
                tets.append(-1)
            tets = '{' + ','.join(map(str, tets)) + '}'
            combss.append(tets)
        lut.append(combs)
        if len(combss) < 6:
            combss.append('{-1}')
        combss = '{ ' + ', '.join(combss) + ' },'
        print(combss)
    all_tets = set(sum([sum(c, []) for c in lut], []))
    print(all_tets)
    faces = list_faces(all_tets)
    print(faces)
    print(len(faces))


def generate_march_lut_edge():
    """Print LUT for edges associated with vertice signs"""
    edges = []
    for idx in range(2**4):
        ss = [(idx>>i)&1 for i in range(4)]
        es = [0]*6
        for i, j in combinations(range(4), 2):
            if ss[i] ^ ss[j]:
                es[EdgesI[frozenset([i, j])]-4] = 1
        es = '{' + ','.join(map(str, es)) + '}'
        if idx % 4 == 0:
            es = '\n' + es
        edges.append(es)
    print(', '.join(edges).strip())


def tetrahedron_goodness_0(vs):
    """always negative, higher is better"""
    vc = sum(vs)/4
    d = [v-vc for v in vs]
    m = 1.0 / np.linalg.norm(d)
    vs = [m*v for v in vs]
    v1, v2, v3 = vs[1]-vs[0], vs[2]-vs[0], vs[3]-vs[0]
    det = np.linalg.det([v1, v2, v3])
    if det <= 0.0:
        return -float('inf')
    return np.log(det)


def tetrahedron_goodness(vs):
    vs = [np.array(Vertices[i]) for i in vs]
    return tetrahedron_goodness_0(vs)


def tetrahedron_goodness_slow(vs):
    global tgs_cache
    if 'tgs_cache' not in globals():
        tgs_cache = {}
    if frozenset(vs) in tgs_cache:
        return tgs_cache[frozenset(vs)]

    def points(v):
        v = sorted(Edges[v])
        if len(v) == 1:
            return [np.array(Vertices[v[0]])]
        a = np.array(Vertices[v[0]])
        b = np.array(Vertices[v[1]])
        n = 8  # n-1 points
        ts = np.arange(1, n) / n
        return [(1.0-t)*a+t*b for t in ts]

    worst, mean, count = 0.0, 0.0, 0
    for v0 in points(vs[0]):
        for v1 in points(vs[1]):
            for v2 in points(vs[2]):
                for v3 in points(vs[3]):
                    g = tetrahedron_goodness_0([v0, v1, v2, v3])
                    worst = min(worst, g)
                    if 0.0*g == 0.0:
                        mean += g
                        count += 1
    if 0.0*worst == 0.0:
        worst = round(worst, 6)
        mean = round(mean/count, 3)
        worst += 1e-6*np.tanh(0.1*mean)
        worst = round(worst, 9)
    tgs_cache[frozenset(vs)] = worst
    return worst


def tetrahedra_goodness(tets):
    tg = tetrahedron_goodness
    tg = tetrahedron_goodness_slow
    return min([tg(t) for t in tets])


def generate_split_lut_edge():
    """Print LUT for tetrahedra after edge split"""
    res = []
    resf = []
    for idx in range(2**6):
        edges = [4+i for i in range(6) if (idx>>i)&1]
        cs0 = find_tetrahedra_combinations(edges[:])
        gs = [tetrahedra_goodness(ts) for ts in cs0]
        best = max(gs)
        cs = [ts for ts in cs0 if tetrahedra_goodness(ts)==best]
        print(edges, len(cs0), len(cs), len(set(gs)), best)
        # print(cs)
        assert len(cs) <= 8
        ts = []
        fs = []
        for c in cs:
            assert len(c) <= 8
            c = [list(ci) for ci in c]
            # tets
            ts.append('{' + ','.join(map(str, (sum(c, []) + [-1])[:8*4])) + '}')
            # faces
            for i in range(len(c)):
                ci = c[i][:]
                for _ in range(4):
                    f = set([ci[f] for f in Faces[_]])
                    for fi in [0, 1, 2, 3, -1]:
                        if fi == -1:
                            break
                        if f.difference(set(Planes[fi])) == set():
                            break
                    c[i][_] = fi
            fs.append('{' + ','.join(map(str, sum(c, []))) + '}')
        ts = (ts + ['{-1}'])[:8]
        res.append('{ ' + ', '.join(ts) + ' }')
        resf.append('{ ' + ', '.join(fs) + ' }')
    print(',\n'.join(res).strip())
    print()
    print(',\n'.join(resf).strip())


if __name__ == "__main__":

    # generate_march_lut_edge()
    # generate_march_lut()

    generate_split_lut_edge()

