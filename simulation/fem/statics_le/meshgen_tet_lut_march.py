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
    set((0, 1, 2, 4, 5, 7)),
    set((0, 2, 3, 5, 6, 9)),
    set((0, 1, 3, 4, 6, 8)),
    set((1, 2, 3, 7, 8, 9))
]


def tetrahedron_volume(vs):
    vs = [np.array(Vertices[i]) for i in vs]
    return round(np.linalg.det([
        vs[1]-vs[0], vs[2]-vs[0], vs[3]-vs[0]
    ]))


def assert_no_coplanar():
    """Make sure the vertice has no coplanar tets
        Not necessary? Might just skip that case."""
    for vs in combinations(range(10), 4):
        # faces must not overlap
        in_planes = False
        for plane in Planes:
            if len(set(vs).difference(plane)) == 0:
                in_planes = True
        if in_planes:
            continue
        # edges must not overlap
        good = True
        for v1, v2 in combinations(vs, 2):
            if v1 < 4 and v2 < 4:
                key = frozenset((v1, v2))
                if EdgesI[key] in vs:
                    good = False
                    break
        if not good:
            continue
        # assert
        if tetrahedron_volume(vs) == 0:
            print(vs)


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
                if len(set(c).difference(plane)) == 0:
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


def generate_march_lut():
    """Print LUT for marching tetrahedra"""
    for idx in range(2**4):
        ss = [i for i in range(4) if (idx>>i)&1]
        combs = []
        for tets in find_tetrahedra_combinations_ss(ss):
            tets = sum(map(list, tets), [])
            if len(tets) < 12:
                tets.append(-1)
            tets = '{' + ','.join(map(str, tets)) + '}'
            combs.append(tets)
        if len(combs) < 6:
            combs.append('{-1}')
        combs = '{ ' + ', '.join(combs) + ' },'
        print(combs)


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


if __name__ == "__main__":

    #assert_no_coplanar()
    #find_tetrahedra_combinations([4, 5, 6, 7, 8, 9])

    #generate_march_lut()
    generate_march_lut_edge()

