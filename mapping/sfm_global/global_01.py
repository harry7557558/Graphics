import cv2 as cv
import numpy as np
import scipy.sparse
import scipy.optimize
import scipy.sparse.linalg
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree

import networkx as nx

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from typing import Union, Optional, Dict, Tuple, List

import os
import sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path

# from sfm_calibrated.ba_solver import ba_solver
# from sfm_calibrated.lc_solver import lc_solver
from ba_solver import ba_solver

import warnings
warnings.filterwarnings('ignore')


# Feature extraction

def filter_features(kps, descs=None, keeps=None, r=20, k_max=20):
    """ https://stackoverflow.com/a/57390077
    Use kd-tree to perform local non-maximum suppression of key-points
    kps - key points obtained by one of openCVs 2d features detectors (SIFT, SURF, AKAZE etc..)
    r - the radius of points to query for removal
    k_max - maximum points retreived in single query
    """

    # sort by score to keep highest score features in each locality
    N = len(kps)
    if keeps is not None:
        keeps_set = set(keeps)
        neg_responses = [-np.inf if i in keeps_set else -kp.response
                         for i,kp in enumerate(kps)]
    else:
        neg_responses = [-kp.response for kp in kps]
    order = np.argsort(neg_responses)
    kps = np.array(kps)[order].tolist()
    if keeps is not None:
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(N)
        keeps = inv_order[np.array(keeps)]

    # create kd-tree for quick NN queries
    data = np.array([list(kp.pt) for kp in kps])
    kd_tree = KDTree(data)

    # perform NMS using kd-tree, by querying points by score order, 
    # and removing neighbors from future queries
    removed = set()
    for i in range(N):
        if i in removed:
            continue
        dist, inds = kd_tree.query(data[i,:],k=k_max,distance_upper_bound=r)
        for j in inds:
            if j>i and (keeps is None or j not in keeps):
                removed.add(j)

    kp_filtered = [kp for i,kp in enumerate(kps) if i not in removed]
    descs_filtered = None
    if descs is not None:
        descs = descs[order]
        descs_filtered = np.array([desc for i,desc in enumerate(descs) if i not in removed])
    print('filtered', len(kp_filtered), 'of', N, 'features')

    if keeps is not None:
        index_filtered = [order[i] for i in range(N) if i not in removed]
        index_map = -np.ones(N, dtype=np.int32)
        index_map[(index_filtered)] = np.arange(len(index_filtered))
        return kp_filtered, descs_filtered, index_map
    return kp_filtered, descs_filtered

def extract_features(img, num_features=8192, filter=False, detector=[]):
    if len(detector) == 0:
        detector.append(cv.ORB_create(num_features))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    keypoints, descriptors = detector[0].detectAndCompute(gray, None)
    keypoints = [(*p.pt, p.size, p.angle, p.response) for p in keypoints]
    return (img, np.array(keypoints), descriptors)


# Video loading

class Frame:
    def __init__(self, image):
        self.img = image
        _, self.keypoints, self.descriptors = extract_features(image, 1536, False)
        # print(self.keypoints.shape, self.keypoints.dtype, self.descriptors.shape, self.descriptors.dtype)

    @property
    def shape(self):
        return self.img.shape

    def plot(self):
        img = np.empty_like(self.img)
        img[:, :self.img.shape[1], :] = self.img
        keypoints = [cv.KeyPoint(*p) for p in self.keypoints]
        cv.drawKeypoints(img, keypoints, img, (0, 255, 0))
        cv.imshow("Matches", img)

class FrameSequence(list):
    cache_path = "cache/frames.npy"

    def __init__(self, video_filename: str, max_frames=200, skip=5):
        super().__init__()

        try:
            self.load_cache()
            print(len(self), "frames loaded")
            if False:
                for f in self:
                    f.plot()
                    cv.waitKey(25)
                cv.destroyAllWindows()
        except FileNotFoundError:
            pass
        except BaseException as e:
            print(e)
        if len(self) > 0:
            return

        cap = cv.VideoCapture(video_filename)
        assert cap.isOpened()

        image_shape = [960, 540]
        target_img_size = np.prod(image_shape)**0.5
        image_shape = None

        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % skip != 0:
                count += 1
                continue
            count += 1

            if image_shape is None:
                img_size = np.prod(frame.shape[:2])**0.5
                sc = target_img_size / img_size
                image_shape = np.round([sc*frame.shape[1], sc*frame.shape[0]])
                print("Image shape:", image_shape)
            frame = cv.resize(frame, image_shape.astype(np.int32))

            frame = Frame(frame)
            frame.plot()
            cv.waitKey(25)
            self.append(frame)

            if len(self) >= max_frames:
                break

        cap.release()
        cv.destroyAllWindows()

        print(len(self), "frames loaded")

        self.save_cache()
    
    def save_cache(self):
        frames = np.array([f for f in self])
        print(len(frames))
        np.save(self.cache_path, frames)
    
    def load_cache(self):
        self.clear()
        for frame in np.load(self.cache_path, allow_pickle=True):
            self.append(frame)



# Feature matching

def draw_matches(frame1: Frame, frame2: Frame, matches, inliner_mask=None):
    if inliner_mask is None:
        inliner_mask = [True] * len(matches)
    img_matches = np.empty((frame1.shape[0], frame1.shape[1]+frame2.shape[1], 3), dtype=np.uint8)
    img_matches[:, :frame1.shape[1], :] = frame1.img
    img_matches[:, frame1.shape[1]:, :] = frame2.img
    for match, inliner in zip(matches, inliner_mask):
        if not inliner:
            continue
        kp1 = frame1.keypoints[match.queryIdx][:2]
        kp2 = frame2.keypoints[match.trainIdx][:2]
        pt1 = (int(kp1[0]), int(kp1[1]))
        pt2 = (int(kp2[0]) + frame1.shape[1], int(kp2[1]))
        cv.line(img_matches, pt1, pt2, (0, 255, 0), 1)
    
    # Display the matches
    cv.imshow("Matches", img_matches)
    cv.waitKey(0)
    cv.destroyAllWindows()

def draw_matches_flow(frame1: Frame, frame2: Frame, matches, inliner_mask=None):
    if inliner_mask is None:
        inliner_mask = [True] * len(matches)
    img_matches = np.empty((frame1.shape[0], frame1.shape[1], 3), dtype=np.uint8)
    img_matches[:, :frame1.shape[1], :] = frame1.img
    for match, inliner in zip(matches, inliner_mask):
        if not inliner:
            continue
        kp1 = frame1.keypoints[match.queryIdx][:2]
        kp2 = frame2.keypoints[match.trainIdx][:2]
        pt1 = (int(kp1[0]), int(kp1[1]))
        pt2 = (int(kp2[0]), int(kp2[1]))
        cv.line(img_matches, pt1, pt2, (0, 255, 0), 1)
    
    # Display the matches
    cv.imshow("Matches", img_matches)
    cv.waitKey(0)
    cv.destroyAllWindows()

class FramePair:
    matcher = cv.BFMatcher(cv.NORM_HAMMING)

    def __init__(self, frame0: Frame, frame1: Frame, intrins: List[float]):

        # feature matching
        matches = self.matcher.match(frame0.descriptors, frame1.descriptors)

        # filter good matches
        if True:
            dist_th = np.hypot(*frame0.shape[:2]) / 20
            good_matches = []
            min_dist = min([match.distance for match in matches])
            for match in matches:
                if match.distance <= max(2 * min_dist, dist_th):
                    good_matches.append(match)
            matches = good_matches

        # draw_matches_flow(frame1, frame2, matches)
        # plt.figure()
        # plt.hist([match.distance for match in matches])
        # plt.show()

        # get matched points
        points0 = frame0.keypoints[:, :2]
        points1 = frame1.keypoints[:, :2]
        if intrins is not None and len(intrins) > 4:
            dist_coeffs = intrins[4:]
            points0 = cv.undistortImagePoints(points0, K, dist_coeffs)[:,0,:]
            points1 = cv.undistortImagePoints(points1, K, dist_coeffs)[:,0,:]

        idx0 = [m.queryIdx for m in matches]
        idx1 = [m.trainIdx for m in matches]
        mpoints0, mpoints1 = points0[idx0], points1[idx1]

        if False:
            plt.plot(mpoints0[:,0], mpoints0[:,1], '.')
            plt.plot(mpoints1[:,0], mpoints1[:,1], '.')
            for p1, p2 in zip(mpoints0, mpoints1):
                plt.plot([p1[0],p2[0]], [p1[1],p2[1]], 'k-')
            plt.show()

        # fundamental matrix
        mat_f, inliner_mask_f = cv.findFundamentalMat(mpoints0, mpoints1)
        # print(inliner_mask_f.sum(), '/', len(inliner_mask_f))
        inliners = np.where(inliner_mask_f.flatten())
        fpoints0, fpoints1 = mpoints0[inliners], mpoints1[inliners]
        self.mat_f = mat_f
        self.idx0 = np.array(idx0)[inliners]
        self.idx1 = np.array(idx1)[inliners]

        # essential matrix
        fx, fy, cx, cy = intrins[:4]
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        mat_e, inliner_mask_e = cv.findEssentialMat(fpoints0, fpoints1, cameraMatrix=K, method=0)
        # print(inliner_mask_e.sum(), '/', len(inliner_mask_e))
        n, R, t, _ = cv.recoverPose(mat_e, fpoints0, fpoints1, cameraMatrix=K)
        # print(R); print(t)
        self.mat_e = mat_e
        self.R, self.t = R, t

        # update
        self.confidence = inliner_mask_e.sum() / len(matches)
        # print(inliner_mask_e.sum(), '/', len(inliner_mask_e))
        inliners = np.where(inliner_mask_e.flatten())
        self.idx0s = self.idx0[inliners]
        self.idx1s = self.idx1[inliners]



# Structure from motion

class DisjointSet:

    def __init__(self, n):
        self.n = n
        self.parent = list(range(n))
        self.size = [1] * n
        self._union_queue = []

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        irep = self.find(i)
        jrep = self.find(j)
        if irep == jrep:
            return
        isize = self.size[irep]
        jsize = self.size[jrep]
        if isize < jsize:
            self.parent[irep] = jrep
            self.size[jrep] += self.size[irep]
        else:
            self.parent[jrep] = irep
            self.size[irep] += self.size[jrep]

    def queue_union(self, i, j):
        self._union_queue.append((i, j))

    def union_queue(self):
        for i, j in self._union_queue:
            self.union(i, j)
        self._union_queue.clear()

    def union_queue_clique3(self):
        """Union that is slightly more robust to outliers"""
        adj_list = [[] for _ in self.size]
        for i, j in self._union_queue:
            adj_list[i].append(j)
            adj_list[j].append(i)
        adj_set = [set(l) for l in adj_list]
        for i in range(len(adj_list)):
            if len(adj_list[i]) == 0:
                continue
            # no need to worry about outliers in this case
            if len(adj_list[i]) == 1:
                j = adj_list[i][0]
                if len(adj_list[j]) == 1 or True:
                    # assert adjacency[j][0] == i
                    self.union(i, j)
                continue
            # union cliques with size 3
            for j in adj_list[i]:
                if self.find(i) == self.find(j):
                    continue
                for k in adj_list[j]:
                    if k != i and k in adj_set[i]:
                        self.union(i, j)
                        self.union(i, k)
        self._union_queue.clear()


class FilteredDisjointSet:

    def __init__(self, n: int, edges: List[Tuple[int, int]]):

        # perform usual union find
        djs = DisjointSet(n)
        for i, j in edges:
            djs.queue_union(i, j)
        # djs.union_queue()
        djs.union_queue_clique3()

        # find disjoint sets
        groups = {}
        for i, j in edges:
            repi = djs.find(i)
            repj = djs.find(j)
            if repi != repj:
                continue
            if repi not in groups:
                groups[repi] = []
            groups[repi].append((i, j))
        groups = groups.values()
        groups = sorted(groups, key=lambda s: -len(s))

        # clean up each disjoint set
        self.edges = []
        for group in groups:
            group = self.filter_djs_group(group)
            self.edges.extend(group)

        # create final disjoint set
        djs = DisjointSet(n)
        for i, j in self.edges:
            djs.queue_union(i, j)
        djs.union_queue()
        self.djs = djs

    @staticmethod
    def filter_djs_group(edges: List[Tuple[int, int]]):
        if len(edges) <= 5:
            return edges
        print(len(edges))
        # if len(edges) > 100:
        #     return edges
        # print(edges)

        G = nx.Graph()
        G.add_edges_from(edges)
        G = FilteredDisjointSet.hcs(G)
        # G, _ = FilteredDisjointSet.remove_branch_edges(G)
        # nx.draw_networkx(G)
        # plt.show()

        return G.edges

    @staticmethod
    def hcs(graph: nx.Graph) -> nx.Graph:
        """https://en.wikipedia.org/wiki/HCS_clustering_algorithm"""

        if len(graph.nodes) <= 5:
            return graph

        graph, branches = FilteredDisjointSet.remove_branch_edges(graph)

        if len(graph.nodes) <= 5:
            graph.add_edges_from(branches)
            return graph

        # assert nx.is_connected(graph)

        if False:
            edges = nx.algorithms.connectivity.cuts.minimum_edge_cut(graph)
        else:
            # fiedler_vector = nx.linalg.fiedler_vector(graph, weight=1, normalized=True)
            fiedler_vector = nx.linalg.fiedler_vector(graph, weight=1, normalized=True, method="lobpcg", tol=1e-4)

            # sorted_nodes = sorted(graph.nodes, key=lambda node: fiedler_vector[node])
            sorted_nodes = np.array(graph.nodes)[np.argsort(fiedler_vector)].tolist()

            if False:
                best_cut = None
                best_cut_weight = float('inf')
                for i in range(1, len(sorted_nodes)):
                    S = set(sorted_nodes[:i])
                    T = set(sorted_nodes[i:])
                    cut_weight = sum(1 for u, v in nx.edge_boundary(graph, S, T))
                    if cut_weight < best_cut_weight:
                        best_cut_weight = cut_weight
                        best_cut = nx.edge_boundary(graph, S, T)

            else:
                boundary_edges = set()
                best_cut_weight = float('inf')
                best_cut = None
                S = set()
                T = set(graph.nodes)
                for u in sorted_nodes:
                    T.remove(u)
                    S.add(u)
                    for v in graph.neighbors(u):
                        edge = (u, v) if u < v else (v, u)
                        if v in T:
                            boundary_edges.add(edge)
                        else:
                            boundary_edges.discard(edge)
                    # assert set(map(frozenset, boundary_edges)) == set(map(frozenset, nx.edge_boundary(graph, S, T)))
                    cut_weight = len(boundary_edges)
                    if 0 < cut_weight < best_cut_weight:
                        best_cut_weight = cut_weight
                        best_cut = boundary_edges.copy()

            edges = set(best_cut)

        is_highly_connected = len(edges) > len(graph.nodes) / 2

        if not is_highly_connected:
            for e in edges:
                graph.remove_edge(*e)

            subgraphs = [graph.subgraph(sg).copy() for sg in nx.connected_components(graph)]
            # print([len(g.nodes) for g in subgraphs])

            if len(subgraphs) == 2:
                g0, g1 = subgraphs
                if len(g0.nodes) > 1:
                    g0 = FilteredDisjointSet.hcs(g0)
                if len(g1.nodes) > 1:
                    g1 = FilteredDisjointSet.hcs(g1)
                graph = nx.compose(g0, g1)

                if len(g0.nodes) == 1 or len(g1.nodes) == 1:
                    nodes0 = set(g0.nodes)
                    nodes1 = set(g1.nodes)
                    for i, j in edges:
                        if i in nodes0 and j in nodes1:
                            graph.add_edge(i, j)
                        if i in nodes1 and j in nodes0:
                            graph.add_edge(i, j)

        graph.add_edges_from(branches)
        return graph

    @staticmethod
    def remove_branch_edges(graph: nx.Graph):
        pruned_graph = graph.copy()
        removed_edges = []

        while True:
            branch_edges = []

            # Identify branch edges
            for edge in pruned_graph.edges():
                u, v = edge

                # Check if either node is a leaf or part of a branch structure
                if pruned_graph.degree[u] == 1 or pruned_graph.degree[v] == 1:
                    branch_edges.append(edge)

            # If no branch edges are found, break the loop
            if not branch_edges:
                break

            # Remove branch edges from the graph
            pruned_graph.remove_edges_from(branch_edges)
            removed_edges.extend(branch_edges)

        # Remove isolated nodes
        isolated_nodes = list(nx.isolates(pruned_graph))
        pruned_graph.remove_nodes_from(isolated_nodes)

        return pruned_graph, removed_edges


class MatchedFrameSequence:
    match_cache_path = "cache/matches.npy"

    def __init__(self, frames: FrameSequence, max_sw=10):
        self.frames = frames
        self.matches = {}  # type: Dict[Tuple[int, int], FramePair]
        self.intrins = self._intrins_init()

        self.match_features(max_sw)
        print()

        # All rotations and translations are world to camera
        self.rotation_averaging()
        # self.translation_averaging()
        print()

        # Bundle adjustment
        self.setup_ba()
        print()

        print("==== Bundle Adjustment ====")
        # intrins, poses, points, _ = bundle_adjustment_3(self.intrins[1:], [*zip(self.poses_R, self.poses_t)], self.points, self.observations)
        # intrins, poses, points, _ = bundle_adjustment_ceres_3(intrins, poses, points, self.observations)
        intrins, poses, points = bundle_adjustment_ceres_3(self.intrins[1:], [*zip(self.poses_R, self.poses_t)], self.points, self.observations)

        if True:
            ax = plt.subplot(projection="3d")
            plot_cameras(ax, [_[0] for _ in poses], [_[1] for _ in poses])
            ax.plot(points[:,0], points[:,2], points[:,1], '.', alpha=0.5)
            set_axes_equal(ax)
            plt.show()

    def _intrins_init(self):
        frame = self.frames[0]
        h, w = frame.shape[:2]
        f = 0.4 * np.hypot(h, w)
        return [f, f, w/2, h/2]

    def match_features(self, max_sw: int):
        try:
            self.matches = np.load(self.match_cache_path, allow_pickle=True)
            self.matches = self.matches.tolist()
        except FileNotFoundError:
            pass
        if len(self.matches.keys()) > 0:
            return

        print("==== Feature Matching ====")
        for i in range(len(self.frames)):
            num_matches = 0
            for j in range(i+1, min(i+max_sw+1, len(frames))):
                match = FramePair(self.frames[i], self.frames[j], self.intrins)
                # print(i, j,  match.confidence)
                if match.confidence > 0.2 or j <= i+2:
                    self.matches[(i, j)] = match
                    num_matches += 1
                else:
                    break
            print(f"{i+1}/{len(frames)} - {num_matches} image matches")
        np.save(self.match_cache_path, self.matches)

    def rotation_averaging(self):
        print("==== Rotation Averaging ====")

        # Initialization
        rotations = [np.eye(3)*1.0]
        translations = [np.zeros((3,1))*1.0]
        for j in range(1, len(frames)):
            i = j-1
            R0, t0 = rotations[i], translations[i]
            match = self.matches[(i, j)]
            R, t = match.R, match.t
            R1, t1 = R0@R, R0@t+t0
            rotations.append(R1)
            translations.append(t1)
            # matches = [cv.DMatch(p[0], p[1], 1.0) for p in zip(match.idx0s, match.idx1s)]
            # draw_matches_flow(self.frames[i], self.frames[j], matches)

        # Optimize poses to refine initial estimate
        terms = [(*key, match.R, match.confidence) for (key, match) in self.matches.items()]

        def objective(x):
            """Objective function to minimize"""
            residuals = []

            for i, j, R_ij, w in terms:
                Ri = x[3*i:3*i+3]
                Rj = x[3*j:3*j+3]
                Ri_mat = Rotation.from_rotvec(Ri).as_matrix()
                Rj_mat = Rotation.from_rotvec(Rj).as_matrix()

                # Rj = Ri @ Rij -> Rij = Ri.T @ Rj
                Rij_est = Ri_mat.T @ Rj_mat

                # Rotation residual
                R_res = Rotation.from_matrix(Rij_est @ R_ij.T).as_rotvec()
                residuals.append(w * R_res)

            return np.concatenate(residuals)

        # Jacobian sparsity
        sp = scipy.sparse.lil_matrix(
            (3*len(terms), 3*len(frames)), dtype=int)
        for ti, (i, j, R_ij, w) in enumerate(terms):
            sp[3*ti:3*ti+3, 3*i:3*i+3] = 1
            sp[3*ti:3*ti+3, 3*j:3*j+3] = 1

        # Run optimization
        x0 = []
        for i in range(len(self.frames)):
            rotvec = Rotation.from_matrix(rotations[i]).as_rotvec()
            x0.append(rotvec)
        x0 = np.concatenate(x0)
        result = scipy.optimize.least_squares(
            objective, x0, jac_sparsity=sp,
            verbose=2, x_scale='jac', ftol=1e-4, method='trf')
        x_opt = result.x

        # Extract optimized rotations
        for i in range(len(self.frames)):
            rotvec = x_opt[3*i:3*i+3]
            rotations[i] = Rotation.from_rotvec(rotvec).as_matrix()
        rotations = rotations[0].T @ rotations
        self.poses_R = np.array(rotations)

        # Do this for translations btw
        translations = np.array(translations)
        translations -= translations.mean(0)
        translations /= 2.0*np.std(translations)
        self.poses_t = translations

        if False:
            ax = plt.subplot(projection="3d")
            plot_cameras(ax, rotations, translations)
            set_axes_equal(ax)
            plt.show()

    def translation_averaging(self):
        print("==== Translation Averaging ====")

        # Initialization
        rotations = self.poses_R
        translations = [np.zeros((3,1))*1.0]
        for j in range(1, len(frames)):
            i = j-1
            R0, t0 = rotations[i], translations[i]
            match = self.matches[(i, j)]
            R, t = match.R, match.t
            R1, t1 = R0@R, R0@t+t0
            translations.append(t1)

        # Optimize poses to refine initial estimate
        terms = [(*key, match.t.flatten(), match.confidence) for (key, match) in self.matches.items()]

        def objective(x):
            """Objective function to minimize"""
            residuals = []
            x_scales = x[len(frames):]
            # residuals.append([np.mean(x_scales)])

            for (i, j, t_ij, w), sc in zip(terms, x_scales):
                Ri = rotations[i]
                ti = x[3*i:3*i+3].reshape((3, 1))
                Rj = rotations[j]
                tj = x[3*j:3*j+3].reshape((3, 1))

                # tj = Ri @ tij + ti -> tij = Ri.T @ (tj - ti)
                tij_est = Ri.T @ (tj - ti)
                tij_est_normalized = tij_est.flatten() / np.linalg.norm(tij_est)

                # t_res = w * np.cross(tij_est_normalized, t_ij)
                t_res = w * (tij_est_normalized - np.exp(sc) * t_ij.flatten())
                # t_res = w * (tij_est.flatten() * np.exp(sc) - t_ij)
                residuals.append(t_res)
                # residuals.append([-w * np.log(np.dot(tij_est_normalized, t_ij))])

            return np.concatenate(residuals)

        # Jacobian sparsity
        sp = scipy.sparse.lil_matrix(
            (3*len(terms), 3*len(frames)+len(terms)), dtype=int)
        for ti, (i, j, R_ij, w) in enumerate(terms):
            sp[3*ti:3*ti+3, 3*i:3*i+3] = 1
            sp[3*ti:3*ti+3, 3*j:3*j+3] = 1
            sp[3*ti:3*ti+3, 3*len(frames)+ti] = 1

        # Run optimization
        x0 = np.concatenate(translations).flatten()
        x0 = np.concatenate((x0, np.zeros(len(terms))*0.0))
        result = scipy.optimize.least_squares(
            objective, x0, jac_sparsity=sp,
            verbose=2, x_scale='jac', ftol=1e-3, method='trf')
        x_opt = result.x

        # Extract optimized translations
        for i in range(len(self.frames)):
            translations[i] = x_opt[3*i:3*i+3]
        translations = np.array(translations)
        translations -= translations.mean(0)
        translations /= 2.0*np.std(translations)
        self.poses_t = np.array(translations)

        if True:
            ax = plt.subplot(projection="3d")
            plot_cameras(ax, rotations, translations)
            set_axes_equal(ax)
            plt.show()

    def setup_ba(self):
        """Create list of points and observations"""

        num_points = np.cumsum([len(frame.keypoints) for frame in self.frames])
        num_points, points_psa = num_points[-1], num_points-num_points[0]
        points_djs = DisjointSet(num_points)
        points_count = [0]*num_points
        # print(num_points, points_psa)
        
        edges = []
        for (i, j), m in self.matches.items():
            idx0 = points_psa[i] + m.idx0s
            idx1 = points_psa[j] + m.idx1s
            for i0, i1 in zip(idx0, idx1):
                points_djs.queue_union(i0, i1)
                edges.append((i0, i1))
                points_count[i0] += 1
                points_count[i1] += 1
        if True:
            # points_djs.union_queue()
            points_djs.union_queue_clique3()
        else:
            points_djs = FilteredDisjointSet(num_points, edges).djs
        for i in range(num_points):
            points_djs.find(i)
        # print([_ for _  in points_djs.size if _ > 1])
        # print(sorted(points_djs.size, reverse=True))
        # assert False

        is_remain = np.array([int(
            points_djs.find(i) == i and points_count[i] >= 3)
              for i in range(num_points)])
        remain_idx = np.arange(num_points)[np.where(is_remain)]
        remain_map = -np.ones(num_points, dtype=np.int32)
        remain_map[remain_idx] = np.cumsum(is_remain)[remain_idx]-1
        print(len(remain_idx), 'points')

        observations = []
        for i, frame in enumerate(self.frames):
            for j, kp in enumerate(frame.keypoints):
                pi = remain_map[points_djs.find(points_psa[i]+j)]
                if pi < 0:
                    continue
                observations.append((i, pi, kp[:2]))
        print(len(observations), 'observations')

        self.points = -1.0+2.0*np.random.random((len(remain_idx), 3))
        self.observations = observations
        # self.poses_t = -1.0+2.0*np.random.random(self.poses_t.shape)



# Bundle adjustment

def exp_so3t(T):
    phi = T[0:3]
    t = T[3:6]
    R, _ = cv.Rodrigues(phi)
    return R, t

def log_so3t(R, t):
    assert np.linalg.norm(R@R.T-np.eye(3)) < 1e-6
    phi, _ = cv.Rodrigues(R)
    return np.concatenate((phi.flatten(), t.flatten()))

BA_OUTLIER_Z = 3

def bundle_adjustment_3(camera_params, poses, points, observations):
    """points + poses + camera intrinsics (f,cx,cy)"""

    # params
    n_poses = len(poses)
    n_points = len(points)
    n_obs = len(observations)
    poses = np.array([log_so3t(*pose) for pose in poses])
    points2d = np.array([uv for (pose_i, point_i, uv) in observations])
    dist_scales = np.zeros(len(observations))
    params_init = np.concatenate((camera_params, poses.flatten(), points.flatten(), dist_scales))
    poses_i = np.array([o[0] for o in observations])
    points_i = np.array([o[1] for o in observations])

    # function
    def fun(params):
        f, cx, cy = params[:3]
        # print(f, cx, cy)
        f, cx, cy = camera_params
        so3ts = params[3:6*n_poses+3].reshape((n_poses, 6))
        poses = [exp_so3t(so3t) for so3t in so3ts]
        points3d = params[6*n_poses+3:6*n_poses+3*n_points+3].reshape((n_points, 3))
        dists = params[6*n_poses+3*n_points+3:]
        dists = np.exp(dists)
        # dists = (dists+1)**2

        R = np.array([p[0] for p in poses])
        t = np.array([p[1] for p in poses])
        vdirs = (points2d - [[cx, cy]]) / f
        vdirs = np.concatenate((vdirs, np.ones_like(vdirs[:, :1])), axis=-1)
        points_r = np.einsum('kij,kj->ki', R[poses_i], points3d[points_i]) + t[poses_i]
        residual = points_r * dists[:, None] - vdirs
        residual = residual.flatten()
        # print(np.mean(residual**2)**0.5)

        # return residual
        # return np.arcsinh(residual)
        delta = 2 / np.hypot(960,540)
        qr_residual = np.sign(residual) * np.sqrt(delta*np.fmax(2.0*np.abs(residual)-delta, 0.0))
        return residual + (qr_residual-residual) * (np.abs(residual) > delta)

    residuals = fun(params_init)
    rmse = np.mean(residuals**2)**0.5
    print('rmse before:', rmse)

    # jacobian sparsity
    sp = scipy.sparse.lil_matrix(
        (3*n_obs, 3+6*n_poses+3*n_points+n_obs), dtype=int)
    for i, (pose_i, point_i, uv) in enumerate(observations):
        p0 = 3+6*pose_i
        sp[3*i:3*i+3, p0:p0+6] = 1  # to pose
        p0 = 3+6*n_poses+3*point_i
        sp[3*i:3*i+3, p0:p0+3] = 1  # to point
        p0 = 3+6*n_poses+3*n_points+i
        sp[3*i:3*i+3, p0:p0+1] = 1  # to distance scale
    sp[:, :3] = 1

    # optimization
    res = scipy.optimize.least_squares(
        fun, params_init, jac_sparsity=sp,
        verbose=2, x_scale='jac', ftol=1e-2, method='trf')
    print('(nfev, njev):', res.nfev, res.njev)
    print('rmse after:', np.mean(res.fun**2)**0.5)
    # print(res)
    
    # residual = np.linalg.norm(res.fun.reshape((-1,2)), axis=1)
    # outliers = np.where(residual > OUTLIER_Z*np.mean(residual))
    residual = np.abs(res.fun.reshape((-1, 3)))
    mask = residual > BA_OUTLIER_Z*np.mean(residual)
    outliers = np.where(mask[:,0] | mask[:,1])

    params = res.x
    intrins = params[:3]
    so3ts = params[3:3+6*n_poses].reshape((n_poses, 6))
    poses = [exp_so3t(so3t) for so3t in so3ts]
    points_3d = params[3+6*n_poses:3+6*n_poses+3*n_points].reshape((n_points, 3))
    dist_scales = params[3+6*n_poses+3*n_points:]
    print(intrins)
    return intrins, poses, points_3d, outliers

def bundle_adjustment_ceres_3(camera_params, poses, points, observations):
    """points + poses + camera intrinsics (fx,fy,cx,cy,*dist_coeffs)"""

    camera_params = np.array(camera_params)
    poses = np.array([log_so3t(*pose) for pose in poses])
    points2d = np.array([uv for (pose_i, point_i, uv) in observations])
    poses_i = np.array([o[0] for o in observations])
    points_i = np.array([o[1] for o in observations])
    dist_scales = np.zeros(len(observations), dtype=np.float64)

    for iter in range(5):
        n_pose = len(poses)
        n_point = len(points)
        n_obs = len(points2d)
        print("iter", iter, '-', n_pose, 'poses,', n_point, 'points,', n_obs, 'obs')

        # optimization
        poses = np.asfortranarray(poses.astype(np.float64))
        points = np.array(points, dtype=np.float64, order='F')
        fixed_intrinsic = iter < 3 or True
        residuals = ba_solver.solve_ba_3(
            camera_params, poses, points, dist_scales,
            poses_i, points_i, points2d,
            fixed_intrinsic, True
        )
        print('intrins:', camera_params)

        # identify outliers
        residuals = np.abs(residuals)
        mask = residuals < BA_OUTLIER_Z * np.mean(residuals, axis=0, keepdims=True)
        mask = mask[:,0] & mask[:,1] & mask[:,2]
        inliners = np.array(np.where(mask)).flatten()

        # filter outliers
        points2d = points2d[inliners]
        poses_i = poses_i[inliners]
        points_i = points_i[inliners]
        dist_scales = dist_scales[inliners]
        # remove points with fewer than 3 observations
        point_idx = np.zeros(len(points))
        for i in points_i:
            point_idx[i] += 1
        mask_p = point_idx >= 3
        inliners_p = np.array(np.where(mask_p)).flatten()
        points = points[inliners_p]
        # clean up observations after point removal
        mask_e = mask_p[points_i]
        inliners_e = np.array(np.where(mask_e)).flatten()
        points2d = points2d[inliners_e]
        poses_i = poses_i[inliners_e]
        points_i = np.cumsum(mask_p)[points_i[inliners_e]]-1
        dist_scales = dist_scales[inliners_e]


    poses = [exp_so3t(so3t) for so3t in poses]
    return camera_params, poses, points



# plotting

def plot_points(ax, points, colors):
    if colors is None:
        ax.scatter(points[:,0], points[:,2], points[:,1], zorder=4)
        return
    ax.scatter(points[:,0], points[:,2], points[:,1], c=colors, marker='.', zorder=4)

def plot_camera(ax, R, t, sc=1.0):
    IMG_SHAPE = [960, 540]
    K = [[500, 0, 480], [0, 500, 270], [0, 0, 1]]
    points = np.array([
        (0, 0, 0),
        (0, 0, 1),
        (IMG_SHAPE[0], 0, 1),
        (IMG_SHAPE[0], IMG_SHAPE[1], 1),
        (0, IMG_SHAPE[1], 1)
    ]).T * sc
    points_3d = R.T @ (np.linalg.inv(K) @ points - t.reshape((3, 1)))
    idx = [0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1]
    vertices = points_3d[:,idx]
    ax.plot(vertices[0], vertices[2], vertices[1], '-')

def plot_cameras(ax, rotations, translations, sc=1.0):
    sc *= np.linalg.det(np.cov(np.array(translations).reshape(-1, 3).T))**(1/6)
    sc /= len(translations)**0.5
    for R, t in zip(rotations, translations):
        plot_camera(ax, R, t, sc)

def set_axes_equal(ax):
    # https://stackoverflow.com/a/31364297
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    ax.invert_zaxis()



if __name__ == "__main__":
    from time import perf_counter

    np.random.seed(42)
    cv.setRNGSeed(42)

    os.makedirs('cache', exist_ok=True)

    import sfm_calibrated.img.videos as videos
    video_filename = videos.videos[4]
    frames = FrameSequence(video_filename, max_frames=200, skip=5)

    MatchedFrameSequence(frames)

