#pragma once

#include <stdio.h>
#include "meshgen_misc.h"

#if SUPPRESS_ASSERT
#undef assert
#define assert(x) 0
#endif

#define MESHGEN_TRIG_IMPLICIT_NS_START namespace MeshgenTrigImplicit {
#define MESHGEN_TRIG_IMPLICIT_NS_END }

MESHGEN_TRIG_IMPLICIT_NS_START

using namespace MeshgenMisc;

namespace MeshgenTrigLoss {
#include "meshgen_trig_loss.h"
}


void generateInitialMeshOld(
    ScalarFieldFBatch Fs, vec2 b0, vec2 b1, ivec2 bn,
    std::vector<vec2> &vertices, std::vector<ivec3> &trigs,
    std::vector<int> &constraintI, std::vector<vec2> &constraintN
) {
    assert(bn.x >= 1 && bn.y >= 1);
    assert(bn.x <= 2000 && bn.y <= 2000); // prevent overflow

    // verts
    auto vci = [=](int i, int j) {  // vertex index
        return j * (bn[0] + 1) + i;
    };
    const int vcn = (bn[0] + 1) * (bn[1] + 1);
    auto vbi = [=](int i, int j) {  // body index
        return vcn + j * bn[0] + i;
    };
    int vn = vcn + bn[0] * bn[1];
    std::vector<vec2> verts(vn);
    constraintI.clear(), constraintN.clear();
    for (int j = 0; j <= bn[1]; j++)
        for (int i = 0; i <= bn[0]; i++)
            verts[vci(i, j)] = mix(b0, b1, vec2(i, j) / vec2(bn.x, bn.y));
    for (int j = 0; j < bn[1]; j++)
        for (int i = 0; i < bn[0]; i++)
            verts[vbi(i, j)] = mix(b0, b1, (vec2(i, j) + vec2(0.5)) / vec2(bn.x, bn.y));
    std::vector<float> vals(vn);
    Fs(verts.size(), &verts[0], &vals[0]);

    // constraints
    std::vector<bool> isOnEdgeX(vn, false), isOnEdgeY(vn, false);
    std::vector<bool> toConstraintX(vn, false), toConstraintY(vn, false);
    std::vector<bool> *isOnEdge[2] = {
        &isOnEdgeX, &isOnEdgeY
    };
    std::vector<bool> *toConstraint[2] = {
        &toConstraintX, &toConstraintY
    };
    for (int j = 0; j <= bn[1]; j++)
        for (int i = 0; i <= bn[0]; i++) {
            int _ = vci(i, j);
            if (i == 0 || i == bn[0])
                isOnEdgeX[_] = true;
            if (j == 0 || j == bn[1])
                isOnEdgeY[_] = true;
        }
    for (int dim = 0; dim < 2; dim++)
        for (int i = 0; i < vn; i++)
            if (vals[i] < 0.0 && isOnEdge[dim]->at(i))
                toConstraint[dim]->at(i) = true;
    for (int j = 0; j <= bn[1]; j++)
        for (int i = 0; i <= bn[0]; i++) {
            int _ = vci(i, j);
            if (toConstraintX[_])
                constraintI.push_back(_), constraintN.push_back(vec2(1, 0));
            if (toConstraintY[_])
                constraintI.push_back(_), constraintN.push_back(vec2(0, 1));
        }

    // triangles
    trigs.clear();
    auto calcTrig = [&](int t1, int t2, int t3) {
        ivec3 t(t1, t2, t3);
        int inCount = 0, cCount = 0;
        for (int _ = 0; _ < 3; _++) {
            inCount += int(vals[t[_]] < 0.0);
            cCount += int(toConstraintX[t[_]] || toConstraintY[t[_]]);
        }
        return ivec2(inCount, cCount);
    };
    auto testTrig = [&](ivec2 p) {
        return p.x - p.y >= 1;
    };
    auto addTrig = [&](int t1, int t2, int t3) {
        ivec2 p = calcTrig(t1, t2, t3);
        if (testTrig(p))
            trigs.push_back(ivec3(t1, t2, t3));
    };
    for (int j = 0; j < bn[1]; j++)
        for (int i = 0; i < bn[0]; i++) {
            int c = vbi(i, j);
            addTrig(vci(i, j), vci(i+1, j), c);
            addTrig(vci(i+1, j), vci(i+1, j+1), c);
            addTrig(vci(i+1, j+1), vci(i, j+1), c);
            addTrig(vci(i, j+1), vci(i, j), c);
            // addTrig(vci(i, j), vci(i+1, j), vci(i+1, j+1));
            // addTrig(vci(i+1, j+1), vci(i, j+1), vci(i, j));
        }

    // remove unused vertices
    std::vector<int> vmap(vn, 0);
    for (ivec3 t : trigs)
        for (int _ = 0; _ < 3; _++)
            vmap[t[_]] = 1;
    vertices.clear();
    for (int i = 0; i < vn; i++) {
        if (vmap[i]) {
            vmap[i] = (int)vertices.size();
            vertices.push_back(verts[i]);
        }
        else vmap[i] = -1;
    }
    for (int i = 0; i < (int)trigs.size(); i++) {
        for (int _ = 0; _ < 3; _++) {
            trigs[i][_] = vmap[trigs[i][_]];
            assert(trigs[i][_] >= 0);
        }
    }
    int j = 0;
    for (int i = 0; i < (int)constraintI.size(); i++) {
        constraintI[j] = vmap[constraintI[i]];
        if (constraintI[j] != -1) j++;
    }
    constraintI.resize(j);
    constraintN.resize(j);
}


void generateInitialMesh(
    ScalarFieldFBatch Fs, vec2 b0, vec2 b1, ivec2 bn, int nd,
    std::vector<vec2> &vertices, std::vector<ivec3> &trigs,
    std::vector<int> &constraintI, std::vector<vec2> &constraintN
) {
    assert(bn.x >= 1 && bn.y >= 1);
    nd++;
    ivec2 bnd(bn.x << nd, bn.y << nd);
    assert(bn.x <= 2000 && bn.y <= 2000); // prevent overflow

    std::unordered_map<int, int> vmap;
    std::vector<float> vals;
    std::vector<int> reqValIdx;

    auto getIdx = [&](int i, int j) {
        return (i << 16) | j;
    };
    auto idxToIj = [&](int idx) {
        int i = idx >> 16;
        int j = idx - (i << 16);
        return ivec2(i, j);
    };
    auto idxToPos = [&](int idx) {
        ivec2 ij = idxToIj(idx);
        float tx = ij.x / (float)bnd.x, ty = ij.y / (float)bnd.y;
        return b0 + (b1 - b0) * vec2(tx, ty);
    };
    auto reqVal = [&](int i, int j) {
        int idx = getIdx(i, j);
        if (vmap.find(idx) == vmap.end()) {
            reqValIdx.push_back(getIdx(i, j));
            vmap[idx] = 0.0;
        }
    };
    auto batchVal = [&]() {
        std::vector<vec2> ps;
        ps.reserve(reqValIdx.size());
        for (int i = 0; i < (int)reqValIdx.size(); i++) {
            int idx = reqValIdx[i];
            vmap[idx] = (int)vals.size() + i;
            ps.push_back(idxToPos(idx));
        }
        vals.resize(vals.size() + ps.size());
        Fs(ps.size(), &ps[0], &vals[vals.size() - ps.size()]);
        reqValIdx.clear();
    };
    auto getVal = [&](int i, int j) {
        int idx = getIdx(i, j);
        assert(vmap.find(idx) != vmap.end());
        return vals[vmap[idx]];
    };
    auto isConstrainedX = [&](int idx) {
        ivec2 ij = idxToIj(idx);
        return ij.x == 0 || ij.x == bnd.x;
    };
    auto isConstrainedY = [&](int idx) {
        ivec2 ij = idxToIj(idx);
        return ij.y == 0 || ij.y == bnd.y;
    };

    // triangles
    trigs.clear();
    auto calcTrig = [&](int t1, int t2, int t3) {
        ivec3 t(t1, t2, t3);
        int inCount = 0, cCount = 0;
        for (int _ = 0; _ < 3; _++) {
            inCount += int(vals[vmap[t[_]]] < 0.0);
            cCount += (int)(isConstrainedX(t[_]) || isConstrainedY(t[_]));
        }
        return ivec2(inCount, cCount);
    };
    auto testTrig = [&](ivec2 p) {
        return p.x - p.y >= 1;
    };
    auto addTrig = [&](int t1, int t2, int t3) {
        ivec2 p = calcTrig(t1, t2, t3);
        if (testTrig(p))
            trigs.push_back(ivec3(t1, t2, t3));
    };
    std::vector<ivec2> squaresToAdd;
    std::vector<int> squareSizes;
    auto addSquare = [&](int i, int j, int step) {
        squaresToAdd.push_back(ivec2(i, j));
        squareSizes.push_back(step);
    };

    // verts
    int step = 1 << nd;
    for (int j = 0; j <= bn[1]; j++)
        for (int i = 0; i <= bn[0]; i++)
            reqVal(i * step, j * step);
    for (int j = 0; j < bn[1]; j++)
        for (int i = 0; i < bn[0]; i++)
            reqVal(i * step + step / 2, j * step + step / 2);
    batchVal();

    // next level
    std::vector<ivec2> squares;
    for (int j = 0; j < bn[1]; j++)
        for (int i = 0; i < bn[0]; i++)
            squares.push_back(ivec2(i*step, j*step));
    
    for (int _ = 1; _ < nd; _++) {
        std::unordered_map<int, int> squaresmap;
        for (int i = 0; i < (int)squares.size(); i++) {
            ivec2 sq = squares[i];
            squaresmap[getIdx(sq.x, sq.y)] = i;
        }

        int step1 = step >> 1;

        // todiv criterion by sign
        std::vector<bool> toDiv0(squares.size(), false);
        for (int i = 0; i < (int)squares.size(); i++) {
            ivec2 sq = squares[i];
            int totsign =
                (int)(getVal(sq.x+0*step, sq.y+0*step) > 0.0) +
                (int)(getVal(sq.x+0*step, sq.y+1*step) > 0.0) +
                (int)(getVal(sq.x+1*step, sq.y+0*step) > 0.0) +
                (int)(getVal(sq.x+1*step, sq.y+1*step) > 0.0) +
                (int)(getVal(sq.x+step/2, sq.y+step/2) > 0.0);
            if (totsign % 5 != 0) {
                toDiv0[i] = true;
            }
        }
        // spread todiv
        std::vector<bool> toDiv = toDiv0;
        if (_ + 1 != nd) {
            for (int i = 0; i < (int)squares.size(); i++) {
                if (!toDiv[i]) continue;
                ivec2 sq = squares[i];
                for (int u = -2; u <= 2; u++)
                    for (int v = -2; v <= 2; v++) {
                        if (u * u  + v * v <= 2) {
                            ivec2 sq1(sq.x + u * step, sq.y + v * step);
                            int idx = getIdx(sq1.x, sq1.y);
                            if (squaresmap.find(idx) != squaresmap.end()) {
                                toDiv0[squaresmap[idx]] = true;
                            }
                        }
                    }
            }
            toDiv = toDiv0;
        }
        // todiv: neighbors must in squares
        for (int i = 0; i < (int)squares.size(); i++) {
            if (!toDiv[i]) continue;
            ivec2 sq = squares[i];
            int sqidx = getIdx(sq.x, sq.y);
            for (int u = -1; u <= 1; u++)
                for (int v = -1; v <= 1; v++) {
                    if (abs(u) + abs(v) == 1) {
                        ivec2 sq1(sq.x + u * step, sq.y + v * step);
                        int idx = getIdx(sq1.x, sq1.y);
                        if (squaresmap.find(idx) == squaresmap.end() && !(
                            sq1.x < 0 || sq1.y < 0 || sq1.x >= bnd.x || sq1.y >= bnd.y
                        ))
                            toDiv0[squaresmap[sqidx]] = false;
                    }
                }
        }
        toDiv = toDiv0;

        // add squares
        step = step1;
        for (int i = 0; i < (int)squares.size(); i++) {
            if (!toDiv[i]) continue;
            int s = step / 2;
            int ic = squares[i].x + 2 * s;
            int jc = squares[i].y + 2 * s;
            // reqVal(ic, jc);
            reqVal(ic - 2 * s, jc);
            reqVal(ic + 2 * s, jc);
            reqVal(ic, jc - 2 * s);
            reqVal(ic, jc + 2 * s);
            reqVal(ic - s, jc - s);
            reqVal(ic + s, jc - s);
            reqVal(ic - s, jc + s);
            reqVal(ic + s, jc + s);
        }
        batchVal();
        for (int i = 0; i < (int)squares.size(); i++) {
            ivec2 sq = squares[i];
            if (!toDiv[i]) {
                addSquare(sq.x, sq.y, step * 2);
            }
        }

        // next iteration
        std::vector<ivec2> squares1;
        for (int i = 0; i < (int)squares.size(); i++) {
            if (!toDiv[i]) continue;
            ivec2 sq = squares[i];
            for (int u = 0; u < 2; u++)
                for (int v = 0; v < 2; v++)
                    squares1.push_back(ivec2(
                        sq.x + u * step,
                        sq.y + v * step));
        }
        squares = squares1;
    }
    for (ivec2 sq : squares) {
        addSquare(sq.x, sq.y, step);
    }

    // add squares
    auto addT2 = [&](int idx1, int idx2, int idx3, int idx0, int s) {
        if (s > 1 && vmap.find(idx0) != vmap.end()) {
            addTrig(idx1, idx0, idx3);
            addTrig(idx2, idx3, idx0);
        }
        else addTrig(idx1, idx2, idx3);
    };
    for (int i = 0; i < (int)squaresToAdd.size(); i++) {
        ivec2 sq = squaresToAdd[i];
        int s = squareSizes[i];
        if (s == 1) continue;
        int i0 = sq.x, i1 = i0 + s / 2, i2 = i0 + s;
        int j0 = sq.y, j1 = j0 + s / 2, j2 = j0 + s;
        int idxc = getIdx(i1, j1);
        addT2(getIdx(i0, j0), getIdx(i2, j0), idxc, getIdx(i1, j0), s);
        addT2(getIdx(i2, j0), getIdx(i2, j2), idxc, getIdx(i2, j1), s);
        addT2(getIdx(i2, j2), getIdx(i0, j2), idxc, getIdx(i1, j2), s);
        addT2(getIdx(i0, j2), getIdx(i0, j0), idxc, getIdx(i0, j1), s);
    }

    // remove unused vertices
    std::vector<int> vpsa(vmap.size(), 0);
    for (ivec3 t : trigs)
        for (int _ = 0; _ < 3; _++)
            vpsa[vmap[t[_]]] = 1;
    vertices.clear();
    for (int i = 0; i < (int)vpsa.size(); i++) {
        if (vpsa[i]) {
            vpsa[i] = (int)vertices.size();
            vertices.push_back(vec2());
        }
        else vpsa[i] = -1;
    }
    for (std::pair<int, int> ii : vmap) {
        int i = vpsa[ii.second];
        if (i != -1) {
            int idx = ii.first;
            vertices[i] = idxToPos(idx);
            if (isConstrainedX(idx)) {
                constraintI.push_back(i);
                constraintN.push_back(vec2(1, 0));
            }
            if (isConstrainedY(idx)) {
                constraintI.push_back(i);
                constraintN.push_back(vec2(0, 1));
            }
        }
    }
    for (int i = 0; i < (int)trigs.size(); i++) {
        for (int _ = 0; _ < 3; _++) {
            trigs[i][_] = vpsa[vmap[trigs[i][_]]];
            assert(trigs[i][_] >= 0);
        }
    }

}



/* Mesh Optimizer */


void assertAreaEqual(
    const std::vector<vec2>& verts,
    const std::vector<ivec3>& trigs
) {
    // sum of trig values
    float At = 0.0;
    for (ivec3 t : trigs) {
        float dA = determinant(mat2(
            verts[t[1]] - verts[t[0]],
            verts[t[2]] - verts[t[0]]
        )) / 2.0;
        assert(dA > 0.0);
        At += dA;
    }

    // edges
    std::set<uint64_t> edges;
    for (ivec3 t : trigs) {
        for (int i = 0; i < 3; i++) {
            ivec2 e;
            for (int _ = 0; _ < 2; _++)
                e[_] = t[(i+_)%3];
            uint64_t ei = ((uint64_t)e.x << 32) | (uint64_t)e.y;
            assert(edges.find(ei) == edges.end());
            uint64_t eo = ((uint64_t)e.y << 32) | (uint64_t)e.x;
            if (edges.find(eo) != edges.end())
                edges.erase(eo);
            else edges.insert(ei);
        }
    }

    // area from boundary
    float As = 0.0;
    for (uint64_t ei : edges) {
        ivec2 e = ivec2((int)(ei>>32), (int)ei);
        float dA = determinant(mat2(
            verts[e.x], verts[e.y]
        )) / 2.0;
        As += dA;
    }
    assert(As > 0);

    // compare
    printf("At=%f As=%f\n", At, As);
    assert(abs(As / At - 1.0) < 1e-3);
}


// Refine the mesh, requires positive volumes for all trigs
void smoothMesh(
    std::vector<vec2>& verts,
    std::vector<ivec3>& trigs,
    int nsteps,
    ScalarFieldFBatch F = nullptr,
    std::function<vec2(vec2)> constraint = nullptr,  // add this to bring point back
    std::vector<int> constraintI = std::vector<int>(),
    std::vector<vec2> constraintN = std::vector<vec2>()
) {
    printf("%d %d\n", (int)constraintI.size(), (int)constraintN.size());
    int vn = (int)verts.size(), svn = 0;  // # of vertices; # on boundary
    int tn = (int)trigs.size(), stn = 0;  // # of trigs; # on boundary
    for (int i : constraintI)
        assert(i >= 0 && i < vn);

    // boundary
    std::set<uint64_t> boundary;
    for (ivec3 t : trigs) {
        for (int i = 0; i < 3; i++) {
            ivec2 e;
            for (int _ = 0; _ < 2; _++)
                e[_] = t[(i+_)%3];
            uint64_t ei = ((uint64_t)e.x << 32) | (uint64_t)e.y;
            assert(boundary.find(ei) == boundary.end());
            uint64_t eo = ((uint64_t)e.y << 32) | (uint64_t)e.x;
            if (boundary.find(eo) != boundary.end())
                boundary.erase(eo);
            else boundary.insert(ei);
        }
    }

    // geometry
    std::vector<int> compressedIndex(vn, -1);  // [vn] global -> near boundary
    std::vector<int> fullIndex;  // [svn] near boundary -> global
    std::vector<int> boundaryTrigs;  // [stn] indices of trigs near boundary
    std::vector<bool> isConstrained[3];  // [vn] constrained on domain boundary? (any, x, y, z)
    std::vector<bool> applyBoundaryConstraints(vn, false);  // [vn] constrained on isoboundary?
    auto isOnBoundary = [&](int i) {
        return isConstrained[0][i] || applyBoundaryConstraints[i];
    };
    // geometry and values
    std::vector<vec2> boundaryVertPs;  // [svn] positions
    std::vector<float> boundaryVertVals;  // [svn] function values
    std::vector<vec2> boundaryVertGrads, boundaryTrigGrads;  // [svn, stn] gradients
    std::vector<float> boundaryVertGradWeights;  // [svn] used to project trig gradients to verts
    // smoothing
    std::vector<vec2> grads(vn);
    std::vector<float> maxFactor(vn), maxMovement(vn);

    // vertices near boundary
    for (int _ = 0; _ < 3; _++)
        isConstrained[_] = std::vector<bool>(vn, false);
    for (int ci = 0; ci < (int)constraintI.size(); ci++) {
        int i = constraintI[ci];
        isConstrained[0][i] = true;
        vec2 t = constraintN[ci];
        if (t == vec2(1, 0))
            isConstrained[1][i] = true;
        else if (t == vec2(0, 1))
            isConstrained[2][i] = true;
        else assert(false);
    }
    if (F) {
        // on boundary
        for (uint64_t f_ : boundary) {
            ivec2 f = ivec2((int)(f_>>32), (int)f_);
            bool isC = isConstrained[0][f[0]]
                && isConstrained[0][f[1]];
            for (int _ = 0; _ < 2; _++) {
                if (compressedIndex[f[_]] == -1) {
                    compressedIndex[f[_]] = svn;
                    fullIndex.push_back(f[_]);
                    svn++;
                }
                if (!isC)
                    applyBoundaryConstraints[f[_]] = true;
            }
        }
        // one layer
        int i0 = 0, i1 = svn;
        for (int ti = 0; ti < tn; ti++) {
            ivec3 t = trigs[ti];
            int onboundary = 0;
            for (int _ = 0; _ < 3; _++)
                if (compressedIndex[t[_]] >= i0 &&
                    compressedIndex[t[_]] < i1)
                    onboundary += 1;
            if (onboundary == 0)
                continue;
            boundaryTrigs.push_back(ti);
            for (int _ = 0; _ < 3; _++)
                if (compressedIndex[t[_]] == -1) {
                    compressedIndex[t[_]] = svn;
                    fullIndex.push_back(t[_]);
                    svn++;
                }
            stn++;
        }
        // make sure isOnBoundary() works
        std::vector<bool> onBoundary(vn, false);
        for (uint64_t f : boundary)
            for (int _ = 0; _ < 2; _++)
                onBoundary[((int*)&f)[_]] = true;
        for (int i = 0; i < vn; i++)
            assert(onBoundary[i] == isOnBoundary(i));
    }
    boundaryVertPs.resize(svn);
    boundaryVertVals.resize(svn);
    boundaryVertGrads.resize(svn);
    boundaryVertGradWeights.resize(svn);
    boundaryTrigGrads.resize(stn);

    for (int stepi = 0; stepi < nsteps; stepi++) {

        /* Smoothing */

        // accumulate gradient
        for (int i = 0; i < vn; i++)
            grads[i] = vec2(0.0);
        for (ivec3 trig : trigs) {
            vec2 v[3], g[3];
            for (int _ = 0; _ < 3; _++)
                v[_] = verts[trig[_]];
            const float* vd = (const float*)&v[0];
            float val, size2;
            float* res[3] = { &val, (float*)g, &size2 };
            MeshgenTrigLoss::meshgen_trig_loss(&vd, res, nullptr, nullptr, 0);
            for (int _ = 0; _ < 3; _++)
                grads[trig[_]] -= 0.005f * g[_] * size2;
        }

        // force the mesh on the boundary
        if (F) {
            // evaluate
            for (int i = 0; i < svn; i++) {
                int j = fullIndex[i];
                if (j != -1)
                    boundaryVertPs[i] = verts[j] + grads[j];
            }
            F(svn, &boundaryVertPs[0], &boundaryVertVals[0]);
            // gradient on trigs
            for (int i = 0; i < stn; i++) {
                vec2 x[3]; float v[3];
                for (int _ = 0; _ < 3; _++) {
                    int j = compressedIndex[trigs[boundaryTrigs[i]][_]];
                    assert(j >= 0 && j < svn);
                    x[_] = boundaryVertPs[j];
                    v[_] = boundaryVertVals[j];
                }
                mat2 m(x[1]-x[0], x[2]-x[0]);
                vec2 b(v[1]-v[0], v[2]-v[0]);
                boundaryTrigGrads[i] = inverse(transpose(m)) * b;
            }
            // gradient on verts
            for (int i = 0; i < svn; i++) {
                boundaryVertGradWeights[i] = 0.0;
                boundaryVertGrads[i] = vec2(0.0);
            }
            for (int i = 0; i < stn; i++) {
                for (int _ = 0; _ < 3; _++) {
                    int j = compressedIndex[trigs[boundaryTrigs[i]][_]];
                    boundaryVertGrads[j] += boundaryTrigGrads[i];
                    boundaryVertGradWeights[j] += 1.0;
                }
            }
            for (int i = 0; i < svn; i++) {
                if (boundaryVertGradWeights[i] <= 0.0) printf("%d %lf\n", i, boundaryVertGradWeights[i]);
                assert(boundaryVertGradWeights[i] > 0.0);
                boundaryVertGrads[i] /= boundaryVertGradWeights[i];
            }
            // boundary constraints
            for (int ci = 0; ci < (int)constraintI.size(); ci++) {
                int i = compressedIndex[constraintI[ci]];
                if (i == -1) continue;
                vec2 n = constraintN[ci];
                boundaryVertGrads[i] -= dot(boundaryVertGrads[i], n) * n;
            }
            // move the vertex to the boundary
            for (int i = 0; i < svn; i++) {
                if (fullIndex[i] == -1)
                    continue;
                if (!applyBoundaryConstraints[fullIndex[i]])
                    continue;
                float v = boundaryVertVals[i];
                vec2 g = boundaryVertGrads[i];
                grads[fullIndex[i]] -= v * g / dot(g, g);
            }
        }

        // apply boundary constraints
        for (int i = 0; i < vn; i++) {
            for (int _ = 0; _ < 2; _++)
                if (isConstrained[_ + 1][i])
                    ((float*)&grads[i])[_] = 0.0;
        }
        if (constraint) {
            for (int i = 0; i < vn; i++)
                grads[i] += constraint(verts[i] + grads[i]);
        }

        // calculate maximum allowed vertex movement factor
        for (int i = 0; i < vn; i++)
            maxFactor[i] = 1.0, maxMovement[i] = 0.0;
        for (ivec3 trig : trigs) {
            // prevent going negative by passing through a face
            // check boundary
            vec2 v[3], g[3];
            float mf[3] = { 1.0, 1.0, 1.0 };
            for (int i = 0; i < 3; i++) {
                for (int _ = 0; _ < 3; _++) {
                    int j = trig[(i+_)%3];
                    v[_] = verts[j], g[_] = grads[j];
                }
                // plane normal and distance to the vertex
                vec2 n = v[1] - v[0]; n = vec2(-n.y, n.x);
                float d = dot(n, v[2] - v[0]);
                if (!(d > 0.0)) printf("error: d = %f\n", d);
                // how far you need to go to make it negative
                float d3 = fmax(-dot(n, g[2]), 0.0f);
                float k[4] = { 1, 1, 1 };
                for (int _ = 0; _ < 2; _++) {
                    float d_ = fmax(dot(n, g[_]), 0.0f);
                    float ds = fmax(d_ + d3, 1e-6);
                    if (ds == 0.0) continue;
                    k[_] = fmin(k[_], d / ds);
                }
                k[3] = fmin(fmin(k[0], k[1]), k[2]);
                for (int _ = 0; _ < 3; _++)
                    mf[(i+_)%3] = fmin(mf[(i+_)%3], k[_]);
            }
            for (int _ = 0; _ < 3; _++)
                maxFactor[trig[_]] = fmin(maxFactor[trig[_]],
                    mf[_] > 0.0f ? mf[_] : 1.0f);
            // prevent going crazy
            float sl = sqrt(abs(determinant(mat2(v[1] - v[0], v[2] - v[0])) / 2.0f));
            for (int _ = 0; _ < 3; _++)
                maxMovement[trig[_]] = fmax(maxMovement[trig[_]], sl);
        }

        // displacements
        for (int i = 0; i < vn; i++) {
            vec2 g = 0.5f * maxFactor[i] * grads[i];
            float gl = length(g);
            if (gl != 0.0f) {
                float a = maxMovement[i];
                g *= a * tanh(gl / a) / gl;
                if (std::isnan(g.x)) {
                    printf("warning: nan displacement %d %f %f %f\n", i, a, gl, maxFactor[i]);
                    continue;
                }
            }
            grads[i] = g;
        }

        // expect this to drop to 0.1x after 20 iterations
        // if not, adjust step size
        float meanDisp = 0.0;
        for (int i = 0; i < vn; i++)
            meanDisp += length(grads[i]) / vn;
        printf("%f\n", meanDisp);

        // reduce displacement if negative area occurs
        std::vector<bool> reduce(vn, true);
        for (int iter = 0; iter < 4; iter++) {
            // update vertex position
            const float r = 0.8;
            float k = (iter == 0 ? 1.0f : (r - 1.0f) * pow(r, iter - 1.0f));
            for (int i = 0; i < vn; i++) if (reduce[i])
                verts[i] += k * grads[i];
            // check if negative area occurs
            reduce = std::vector<bool>(vn, false);
            bool found = false;
            for (ivec3 trig : trigs) {
                vec2 v[3] = {
                    verts[trig[0]], verts[trig[1]], verts[trig[2]]
                };
                if (determinant(mat2(v[1] - v[0], v[2] - v[0])) < 0.0) {
                    reduce[trig[0]] = reduce[trig[1]] =
                        reduce[trig[2]] = reduce[trig[3]] = true;
                    found = true;
                    printf("%d\n", iter);
                }
            }
            if (!found) break;
        }

    }

}


MESHGEN_TRIG_IMPLICIT_NS_END
