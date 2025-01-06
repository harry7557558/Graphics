import cv2 as cv
import numpy as np
import scipy.sparse
import scipy.optimize
import scipy.sparse.linalg
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema
import bisect
import networkx as nx

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from typing import Union, Optional, Dict, Tuple, List, Callable

import os
import sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path

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
        _, self.keypoints, self.descriptors = extract_features(image, 1024, False)
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
            print(len(self), "frames loaded", end='\n\n')
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

        print("==== Feature Extraction ====")

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
        print()

        self.save_cache()
    
    def save_cache(self):
        frames = np.array([f for f in self])
        np.save(self.cache_path, frames)
    
    def load_cache(self):
        self.clear()
        for frame in np.load(self.cache_path, allow_pickle=True):
            self.append(frame)



# Feature matching

def pack_pose(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:] = t.reshape((3, 1))
    return T

def unpack_pose(T):
    return T[:3, :3], T[:3, 3:]

def exp_so3t(T):
    phi = T[0:3]
    t = T[3:6]
    R, _ = cv.Rodrigues(phi)
    return pack_pose(R, t)

def log_so3t(T):
    R, t = unpack_pose(T)
    assert np.linalg.norm(R@R.T-np.eye(3)) < 1e-6
    phi, _ = cv.Rodrigues(R)
    return np.concatenate((phi.flatten(), t.flatten()))


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
        f, cx, cy = intrins[:4]
        K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
        mat_e, inliner_mask_e = cv.findEssentialMat(fpoints0, fpoints1, cameraMatrix=K, method=0)
        # print(inliner_mask_e.sum(), '/', len(inliner_mask_e))
        n, R, t, _ = cv.recoverPose(mat_e, fpoints0, fpoints1, cameraMatrix=K)
        # print(R); print(t)
        self.mat_e = mat_e
        self.R, self.t = R, t
        self.T = pack_pose(self.R, self.t)

        # update
        self.confidence = inliner_mask_e.sum() / len(matches)
        # print(inliner_mask_e.sum(), '/', len(inliner_mask_e))
        inliners = np.where(inliner_mask_e.flatten())
        self.idx0s = self.idx0[inliners]
        self.idx1s = self.idx1[inliners]



# Structure from motion

def inverse_index_map(mp, n, assert_unique=False):
    if assert_unique:
        assert len(set(mp)) == len(mp)
    imp = -np.ones(n, dtype=np.int32)
    imp[mp] = np.arange(len(mp))
    return imp


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

    def __init__(self, n: int, edges: List[Tuple[int, int]], max_nodes: int):

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
            group = self.filter_djs_group(group, max_nodes)
            self.edges.extend(group)

        # create final disjoint set
        djs = DisjointSet(n)
        for i, j in self.edges:
            djs.queue_union(i, j)
        djs.union_queue()
        self.djs = djs

    @staticmethod
    def filter_djs_group(edges: List[Tuple[int, int]], max_nodes: int):
        if len(edges) <= max_nodes:
            return edges

        G = nx.Graph()
        G.add_edges_from(edges)
        G = FilteredDisjointSet.partition_graph(G, max_nodes)
        # nx.draw_networkx(G)
        # plt.show()

        return G.edges

    @staticmethod
    def partition_graph(graph: nx.Graph, max_nodes: int) -> nx.Graph:

        if len(graph.nodes) <= max_nodes:
            return graph

        # fiedler_vector = nx.linalg.fiedler_vector(graph, weight=1, normalized=True)
        fvec = nx.linalg.fiedler_vector(graph, weight=1, normalized=True, method="lobpcg", tol=1e-4)

        fidx = np.argsort(fvec)
        fidx_inv = np.empty_like(fidx)
        fidx_inv[fidx] = np.arange(len(fidx))
        fvec = fvec[fidx]

        def split_fvec(fvec):
            if len(fvec) <= max_nodes:
                return [fvec]
            si = np.argmax(fvec[1:]-fvec[:-1])
            return split_fvec(fvec[:si+1]) + split_fvec(fvec[si+1:])

        fvec_split = split_fvec(fvec)
        vmap = []
        for i, fv in enumerate(fvec_split):
            vmap.append([len(vmap)]*len(fv))
        vmap = np.concatenate(vmap)

        # plt.plot(fvec, 'k.')
        # plt.plot(fvec[1:]-fvec[:-1], 'k.')
        # plt.show()

        nmap = { node: vmap[fidx_inv[i]] for i, node in enumerate(graph.nodes()) }
        discarded_edges = []
        for u, v in graph.edges():
            if nmap[u] != nmap[v]:
                discarded_edges.append((u, v))
        graph.remove_edges_from(discarded_edges)
        return graph


def scaled_procrustes(pnts1, pnts2, weights):
    pnts1, pnts2 = np.array(pnts1), np.array(pnts2)
    weights = np.array(weights).reshape((-1, 1))
    weights /= weights.sum()
    mean1 = (pnts1 * weights).sum(axis=0)
    mean2 = (pnts2 * weights).sum(axis=0)

    centered1 = pnts1 - mean1
    centered2 = pnts2 - mean2

    H = (centered2 * weights).T @ centered1
    U, S, Vh = np.linalg.svd(H)

    S = np.eye(3)
    S[2, 2] = np.linalg.det(U @ Vh)
    R = U @ S @ Vh

    # TODO: is this correct?
    s = np.sqrt((centered2**2 * weights).sum() / (centered1**2 * weights).sum())

    t = mean2 - s * R @ mean1

    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3, 3] = t/s
    mat[3, 3] = 1/s
    return mat


class PoseGraph:
    min_obs: int = 3
    """Minimum number of observations per point"""
    min_obs_ba: int = 3
    """Minimum number of observations per point, for filtering after BA"""

    def __init__(self, frames, matches, frame_id, w2cs):
        assert len(frame_id) == len(w2cs)
        self.frame_id = frame_id  # type: List[int]
        self.w2c = w2cs  # type: List[np.ndarray]

        self.fmap = {}  # type: Dict[int, int]
        for i, fi in enumerate(frame_id):
            self.fmap[fi] = i

        # merge and filter list of points
        num_points = np.cumsum([len(frames[fi].keypoints) for fi in frame_id])
        num_points, points_psa = num_points[-1], num_points-num_points[0]
        self.points_psa = np.concatenate((points_psa, [num_points]))
        points_djs = DisjointSet(num_points)
        points_count = [0]*num_points

        edges = []
        for (i, j), m in matches.items():
            if i not in frame_id or j not in frame_id:
                continue
            i, j = self.fmap[i], self.fmap[j]
            idx0 = points_psa[i] + m.idx0
            idx1 = points_psa[j] + m.idx1
            for i0, i1 in zip(idx0, idx1):
                points_djs.queue_union(i0, i1)
                edges.append((i0, i1))
                points_count[i0] += 1
                points_count[i1] += 1
        if False:
            # points_djs.union_queue()
            points_djs.union_queue_clique3()
        else:
            max_nodes = int(1.5 * BottomUpFrameSequence.n_bottom + 1)
            points_djs = FilteredDisjointSet(num_points, edges, max_nodes).djs
        for i in range(num_points):
            points_djs.find(i)
        self.djs_map = np.array([points_djs.find(i) for i in range(num_points)])

        # initial map
        self.is_remain = np.array([
            int(points_count[i] >= self.min_obs)
            for i in range(num_points)])
        self.create_remain_map_idx()

        # filter points with too few observations
        for idxs in self.remain_idx:
            if len(idxs) < self.min_obs:
                for i in idxs:
                    self.is_remain[i] = False
        self.create_remain_map_idx()

        # create list of observations
        self.observations = []
        for i, fi in enumerate(frame_id):
            for j, kp in enumerate(frames[fi].keypoints):
                pi = self.remain_map[points_psa[i]+j]
                if pi < 0:
                    continue
                self.observations.append((fi, pi, kp[:2]))

        # initialize points
        self.points = -1.0+2.0*np.random.random((self.num_points_remain, 3))
        # TODO: triangulation

        self.sanity_check()

    @property
    def num_points_original(self):
        return self.points_psa[-1]

    @property
    def num_points_remain(self):
        return len(self.remain_idx)

    def create_remain_map_idx(self):
        num_points = self.num_points_original
        remain_idx = []  # type: List[List[int]]
        remain_map = -np.ones(num_points, dtype=np.int32)
        for i in range(num_points):
            ri = self.djs_map[i]
            if ri == i and self.is_remain[ri]:
                remain_map[i] = len(remain_idx)
                remain_idx.append([])
        for i in range(num_points):
            ri = self.djs_map[i]
            if remain_map[ri] != -1:
                remain_map[i] = remain_map[ri]
                remain_idx[remain_map[i]].append(i)
        self.remain_map = remain_map  # map original indices to remaining indices
        self.remain_idx = remain_idx  # list of list, original indices of remaining points

    def update_remain_map_idx(self, pmap):
        # pmap: old indices to new indices
        # update remain map
        pmap_inv = inverse_index_map(pmap, self.num_points_remain)
        for i in range(len(self.remain_map)):
            ri = self.remain_map[i]
            if ri != -1:
                self.remain_map[i] = pmap_inv[ri]
        # update remain idx
        self.remain_idx = [self.remain_idx[i] for i in pmap]

    def sanity_check(self):
        # remain_idx and remain_map
        assert len(self.remain_map) == self.num_points_original
        remain_idx = [[] for _ in self.remain_idx]
        for i in range(self.num_points_original):
            ri = self.remain_map[i]
            if ri != -1:
                remain_idx[ri].append(i)
        for ri0, ri1 in zip(self.remain_idx, remain_idx):
            assert set(ri0) == set(ri1)

        # number of points
        assert len(self.points) == self.num_points_remain
        assert set(self.remain_map).difference([-1]) == set(range(len(self.remain_idx)))
        point_count = np.zeros(self.num_points_remain, dtype=np.int32)
        for pi, i, uv in self.observations:
            point_count[i] += 1
        diff = 0
        for i in range(self.num_points_remain):
            assert point_count[i] >= self.min_obs_ba
            assert point_count[i] <= len(self.remain_idx[i])
            if point_count[i] != len(self.remain_idx[i]) and False:
                if point_count[i] < len(self.remain_idx[i]):
                    print(i, point_count[i], len(self.remain_idx[i]), 'point_count[i] too small')
                if point_count[i] > len(self.remain_idx[i]):
                    print(i, point_count[i], len(self.remain_idx[i]), 'len(remain_idx[i]) too small')
                diff += point_count[i] - len(self.remain_idx[i])
        # print("Total diff:", diff)

    def merge(pg0: "PoseGraph", pg1: "PoseGraph", get_match: Callable, frames: List[Frame]):

        # TODO: add new points

        # get relative transform from closest match pair
        cidx0, cidx1 = [], []
        fidx0, fidx1 = [], []
        fidx0s, fidx1s = set(), set()
        num_new_point = 0
        for stride in range(1, BottomUpFrameSequence.max_sw+1):
            if num_new_point == 0 and stride % BottomUpFrameSequence.n_bottom != 1:
                continue
            num_new_point, num_new_match = 0, 0
            for fi0 in range(len(pg0.frame_id)-stride, len(pg0.frame_id)):
                fi1 = fi0+stride-len(pg0.frame_id)
                if not (fi0 >= 0 and fi1 < len(pg1.frame_id)):
                    continue
                match = get_match(pg0.frame_id[fi0], pg1.frame_id[fi1]) # type: FramePair
                if match is None:
                    continue
                num_new_match += 1
                idx0 = pg0.points_psa[fi0] + match.idx0
                idx1 = pg1.points_psa[fi1] + match.idx1
                ridx0 = pg0.remain_map[idx0]
                ridx1 = pg1.remain_map[idx1]
                rmask = (ridx0 != -1) & (ridx1 != -1)
                for i0, i1 in zip(ridx0[np.where(rmask)], ridx1[np.where(rmask)]):
                    if i0 in fidx0s or i1 in fidx1s:
                        continue
                    fidx0s.add(i0)
                    fidx1s.add(i1)
                    fidx0.append(i0)
                    fidx1.append(i1)
                    cidx0.append(fi0)
                    cidx1.append(fi1)
                    num_new_point += 1
            if num_new_point == 0 and False:
                break
            print(f'stride {stride}: {num_new_match} new matches, {num_new_point} new points')
        fidx0, fidx1 = np.array(fidx0), np.array(fidx1)
        cidx0, cidx1 = np.array(cidx0), np.array(cidx1)
        cpos0 = np.array([np.linalg.inv(cam)[:3, 3] for cam in pg0.w2c])
        cpos1 = np.array([np.linalg.inv(cam)[:3, 3] for cam in pg1.w2c])
        num_points_0 = len(fidx0)

        # TODO: RANSAC
        pnt0, pnt1 = pg0.points[fidx0], pg1.points[fidx1]
        weights = (np.linalg.norm(pnt0-cpos0[cidx0], axis=1) * np.linalg.norm(pnt1-cpos1[cidx1], axis=1))**-0.5
        relmat = scaled_procrustes(pnt1, pnt0, weights)
        error = (pnt1 @ relmat[:3, :3].T + relmat[:3, 3:].T) / relmat[3, 3] - pnt0
        error = np.linalg.norm(error, axis=1) * weights
        mask = np.where(error < 2.5*np.mean(error))
        fidx0, fidx1 = fidx0[mask], fidx1[mask]
        cidx0, cidx1 = cidx0[mask], cidx1[mask]
        pnt0, pnt1 = pg0.points[fidx0], pg1.points[fidx1]
        weights = (np.linalg.norm(pnt0-cpos0[cidx0], axis=1) * np.linalg.norm(pnt1-cpos1[cidx1], axis=1))**-0.5
        print(f'{len(fidx0)}/{num_points_0} points for procrustes')
        relmat = scaled_procrustes(pnt1, pnt0, weights)

        # transform poses
        pg0.frame_id.extend(pg1.frame_id)
        for i, fi in enumerate(pg0.frame_id):
            pg0.fmap[fi] = i
        for mat in pg1.w2c:
            mat = np.linalg.inv(mat)  # c2w
            mat = relmat @ mat
            mat[:, 3] /= mat[3, 3]
            mat = np.linalg.inv(mat)  # w2c
            pg0.w2c.append(mat)

        # plotting
        if len(pg0.frame_id) > 20:
            plt.figure()
            ax = plt.subplot(projection="3d")
            plot_cameras(ax, pg0.w2c)
            set_axes_equal(ax)
            apnt0 = pg0.points
            ax.scatter(apnt0.T[0], apnt0.T[2], apnt0.T[1], c='C0', s=1)
            apnt1 = (pg1.points @ relmat[:3, :3].T + relmat[:3, 3:].T) / relmat[3, 3]
            ax.scatter(apnt1.T[0], apnt1.T[2], apnt1.T[1], c='C1', s=1)
            pnt1v = (pnt1 @ relmat[:3, :3].T + relmat[:3, 3:].T) / relmat[3, 3]
            ax.scatter(pnt0.T[0], pnt0.T[2], pnt0.T[1])
            ax.scatter(pnt1v.T[0], pnt1v.T[2], pnt1v.T[1])
            pnt01 = np.stack((pnt0, pnt1v))
            for i in range(len(pnt0)):
                si = pnt01[:, i].T
                ax.plot(si[0], si[2], si[1], 'k-')
            plt.show()

        # merge points
        num_points_0 = pg0.num_points_remain
        pnt1 = (pg1.points @ relmat[:3, :3].T + relmat[:3, 3:].T) / relmat[3, 3]
        pmap = np.ones(len(pnt1), dtype=np.int32)
        pmap[fidx1] = 0
        pg0.points = np.concatenate((pg0.points, pnt1[np.where(pmap)]))
        pmap = np.cumsum(pmap)-1 + num_points_0
        pmap[fidx1] = fidx0
        # pmap = np.concatenate((np.arange(num_points_0), pmap))

        # merge observations
        for fi, pi, uv in pg1.observations:
            pg0.observations.append((fi, pmap[pi], uv))

        # update remain_map and remain_idx
        num_points_0_orig = pg0.num_points_original
        num_points = np.cumsum([len(frames[fi].keypoints) for fi in pg0.frame_id])
        num_points, points_psa = num_points[-1], num_points-num_points[0]
        pg0.points_psa = np.concatenate((points_psa, [num_points]))
        pg0.djs_map = None
        pg0.is_remain = np.concatenate((pg0.is_remain, pg1.is_remain))
        extra_remain_map = pg1.remain_map.copy()
        for i, ri in enumerate(extra_remain_map):
            if ri != -1:
                extra_remain_map[i] = pmap[ri]
        pg0.remain_map = np.concatenate((pg0.remain_map, extra_remain_map))
        for i, idxs in enumerate(pg1.remain_idx):
            idxs = [ri+num_points_0_orig for ri in idxs]
            if pmap[i] < num_points_0:
                pg0.remain_idx[pmap[i]].extend(idxs)
            else:
                assert len(pg0.remain_idx) == pmap[i]
                pg0.remain_idx.append(idxs)

    def plot(self, frames):
        plt.figure()
        ax = plt.subplot(projection="3d")
        plot_cameras(ax, self.w2c)
        set_axes_equal(ax)
        c, s = self.get_point_appearance(frames)
        ax.scatter(self.points[:,0], self.points[:,2], self.points[:,1],
                alpha=0.5, s=s, c=c, zorder=0)
        plt.show()

    def get_point_appearance(self, frames):
        colors = np.zeros((len(self.points), 3), dtype=np.float32)
        counts = np.zeros(len(self.points))
        for fi, pi, uv in self.observations:
            i, j = np.round(uv).astype(np.int32)
            color = frames[fi].img[j, i]
            colors[pi] += color
            counts[pi] += 1
        assert (counts > 0).all()
        colors = colors / counts[:, None]
        colors = np.flip(colors, -1)
        return colors / 255.0, np.cbrt(counts/3)


class BottomUpFrameSequence:
    n_bottom: int = 5
    """Target frame group size at initialization"""
    image_match_th: float = 0.2
    """Confidence threshold for image matching, 0 to 1"""
    max_sw: int = 10
    """Maximum sliding window size"""

    def __init__(self, frames: FrameSequence):
        self.n = len(frames)
        self.frames = [frame for frame in frames]  # type: List[Frame]
        self.matches = {}  # type: Dict[Tuple[int, int], FramePair]
        # self.w2c = [None for _ in range(self.n)]  # type: List[np.ndarray]
        self.intrins = self._intrins_init()
        self.pose_graphs = []  # type: List[PoseGraph]

        self.initial_match()
        self.run_ba()

        # get_match = lambda i, j: self.match_feature_pair(i, j)
        # self.pose_graphs[0].merge(self.pose_graphs[1], get_match, self.frames)
        # self.pose_graphs[0].sanity_check()

        while len(self.pose_graphs) > 1:
            self.binary_merge()
            self.run_ba()
        self.pose_graphs[0].plot(frames)
        return

    def _intrins_init(self):
        frame = self.frames[0]
        h, w = frame.shape[:2]
        f = 0.4 * np.hypot(h, w)
        return np.array([f, w/2, h/2])

    def match_feature_pair(self, i, j):
        match = FramePair(self.frames[i], self.frames[j], self.intrins)
        # print(i, j,  match.confidence)
        if match.confidence > self.image_match_th or j <= i+2:
            self.matches[(i, j)] = match
            return match
        return None

    def initial_match(self):

        print("==== Feature Matching ====")

        # distribute frames
        counts = np.ones(self.n//self.n_bottom, dtype=np.int32) * self.n_bottom
        for i in range(self.n % self.n_bottom):
            counts[i%len(counts)] += 1
        indices = np.concatenate(([0], np.cumsum(counts)))
        self.indices = []
        for i in range(len(counts)):
            self.indices.append(list(range(indices[i], indices[i+1])))
        self.pose_graphs = []  # type: List[PoseGraph]
        # print(self.indices)

        # match features
        for group in self.indices:

            # pairwise matching
            new_matches = {}
            for i in range(len(group)):
                for j in range(i+1, min(i+self.max_sw+1, len(group))):
                    gi, gj = group[i], group[j]
                    match = FramePair(self.frames[gi], self.frames[gj], self.intrins)
                    # print(i, j,  match.confidence)
                    if match.confidence > self.image_match_th or j <= i+2 or True:
                        self.matches[(gi, gj)] = match
                        new_matches[(gi, gj)] = match
                    else:
                        break

            # initialization
            T = np.eye(4, dtype=np.float64)
            w2cs = [T]
            for i in range(1, len(group)):
                i, j = group[i-1], group[i]
                T = T @ self.matches[(i, j)].T
                w2cs.append(T)
            sc = exp_so3t(-np.mean([log_so3t(T) for T in w2cs], axis=0))
            w2cs = [sc @ T for T in w2cs]

            # pose graph
            pg = PoseGraph(self.frames, new_matches, group, w2cs)
            self.pose_graphs.append(pg)

            print(f"{group} of {len(self.frames)} - {len(new_matches)} matches, {len(pg.points)} points, {len(pg.observations)} observations")
        print()

    def binary_merge(self):
        print("==== Pose Graph Merging ====")
        pose_graphs = []  # type: List[PoseGraph]
        get_match = lambda i, j: self.match_feature_pair(i, j)
        for i in range(0, len(self.pose_graphs), 2):
            if i+1 >= len(self.pose_graphs):
                if len(pose_graphs) > 0:
                    print(f"Merge pose graph {i-2}, {i-1}, and {i}")
                    pg1 = self.pose_graphs[i]
                    pose_graphs[-1].merge(pg1, get_match, self.frames)
            else:
                print(f"Merge pose graph {i} and {i+1}")
                pg0, pg1 = self.pose_graphs[i], self.pose_graphs[i+1]
                pg0.merge(pg1, get_match, self.frames)
                pg0.sanity_check()
                pose_graphs.append(pg0)
            print()
        self.pose_graphs = pose_graphs

    def run_ba(self):

        print("==== Bundle Adjustment ====")

        # merge pose graphs
        poses = []
        points = []
        observations = []
        points_i = []
        observations_i = []
        pi0, obsi0 = 0, 0
        pi_psa, obsi_psa = [pi0], [obsi0]
        for pgi, pg in enumerate(self.pose_graphs):
            for fi, pi, uv in pg.observations:
                observations.append((fi, pi+pi0, uv))
            poses.extend(pg.w2c)
            points.append(pg.points)
            points_i.extend([pgi]*len(pg.points))
            observations_i.extend([pgi]*len(pg.observations))
            pi0 += len(pg.points)
            obsi0 += len(pg.observations)
            pi_psa.append(pi0)
            obsi_psa.append(obsi0)
        points = np.concatenate(points, axis=0)

        # run BA
        self.intrins, poses, points, observations, points_map, obs_map = bundle_adjustment_ceres_3(
            self.intrins, poses, points, observations,
            fixed_intrinsic=True, num_iter=2
        )
        # points_map, obs_map: map new indices to old indices
        if False:  # unused, reserve for later
            points_map_inv = inverse_index_map(points_map, len(points_i), True)
            obs_map_inv = inverse_index_map(obs_map, len(observations_i), True)
        print()

        # update prefix sum arrays
        pi_psa_1 = []
        for i in pi_psa:
            pi_psa_1.append(bisect.bisect_left(points_map, i))

        # update results
        points_i0 = np.array(points_i)
        points_i = points_i0[points_map]
        observations_i = np.array(observations_i)[obs_map]
        pi0 = 0
        for pgi, pg in enumerate(self.pose_graphs):
            assert pi0 == pi_psa_1[pgi]
            np0, no0 = len(pg.points), len(pg.observations)

            pg.observations = [(fi, pi-pi0, uv) for i, (fi, pi, uv) in
                               zip(observations_i, observations) if i == pgi]
            pg.w2c = [poses[i] for i in pg.frame_id]
            pg.points = points[pi_psa_1[pgi]:pi_psa_1[pgi+1]]

            pmap = points_map[pi_psa_1[pgi]:pi_psa_1[pgi+1]] - pi_psa[pgi]
            pg.update_remain_map_idx(pmap)
            pg.sanity_check()
            assert pg.num_points_remain == len(pmap)

            print(f"{pg.frame_id[0]}-{pg.frame_id[-1]} of {len(self.frames)} - "
                  f"{np0}->{len(pg.points)} points, {no0}->{len(pg.observations)} observations")
            pi0 += len(pg.points)

        print()

    def close_loop(self):

        print("==== Loop Closure ====")

        self.poses_R = np.array([R for (R, t) in self.poses])
        self.poses_t = np.array([t for (R, t) in self.poses])

        # compute pairwise distance
        dist_mode = ["rotation", "ray_dir", "ray_dist"][1]
        if dist_mode == "rotation":
            dists = self.poses_R.reshape((-1, 1, 3, 3)) - self.poses_R.reshape((1, -1, 3, 3))
            dists = (dists**2).sum(axis=(-1, -2))
            dists -= 1.0
        elif dist_mode == "ray_dir":
            dir1 = self.poses_R[:, 2].reshape((-1, 1, 3))
            dir2 = dir1.reshape((1, -1, 3))
            dists = ((dir2-dir1)**2).sum(axis=(-1))
            dists -= 1.0/3.0
        elif dist_mode == "ray_dist":
            dir1 = self.poses_R[:, 2].reshape((-1, 1, 3))
            dir2 = dir1.reshape((1, -1, 3))
            n = np.cross(dir1, dir2, axisa=-1, axisb=-1)
            pos1 = -np.einsum('kij,ki->kj', self.poses_R, self.poses_t).reshape((-1, 1, 3))
            pos2 = pos1.reshape((1, -1, 3))
            dp = pos2 - pos1
            t1 = (np.cross(dir2, n, -1, -1) * dp).sum(-1) / (n*n).sum(-1)
            t2 = (np.cross(dir1, n, -1, -1) * dp).sum(-1) / (n*n).sum(-1)
            p1 = pos1 + dir1 * np.fmax(t1, 0.0)[..., None]
            p2 = pos2 + dir2 * np.fmax(t2, 0.0)[..., None]
            dists = np.linalg.norm(p2-p1, axis=-1)
        # dists = gaussian_filter(dists, sigma=10)
        # plt.imshow(dists); plt.show()

        # find potential matches
        lc_list = []
        for i, dist_i in enumerate(dists):
            minima = argrelextrema(dist_i, np.less)[0]
            minima = minima[np.where(dist_i[minima] < 0)]
            # plt.plot(dist_i); plt.show()
            lc_list.extend([(*sorted([i, j]),) for j in minima if i != j])
        lc_list = sorted(set(lc_list))
        lc_list = [ij for ij in lc_list if ij not in self.matches]
        # print(lc_list)
        self.lc_list = lc_list
        # self.plot()

        # match images
        lc_list = []
        for i, j in self.lc_list:
            match = FramePair(self.frames[i], self.frames[j], self.intrins)
            status = ''
            if match.confidence > self.config.image_match_th:
                self.matches[(i, j)] = match
                lc_list.append((i, j))
                status = '- accept'
            print(i, j, match.confidence, status)
        print(f"{len(lc_list)}/{len(self.lc_list)} image matches")
        self.lc_list = lc_list
        # self.plot()

        print()



# Bundle adjustment

BA_OUTLIER_Z = 8

def bundle_adjustment_ceres_3(
        camera_params, poses, points, observations,
        fixed_intrinsic: bool, num_iter: int):
    """points + poses + camera intrinsics (f,cx,cy)"""

    camera_params = np.array(camera_params)
    poses = np.array([log_so3t(pose) for pose in poses])
    points2d = np.array([uv for (pose_i, point_i, uv) in observations])
    poses_i = np.array([o[0] for o in observations])
    points_i = np.array([o[1] for o in observations])
    dist_scales = np.zeros(len(observations), dtype=np.float64)
    points_map = np.arange(len(points))
    obs_map = np.arange(len(observations))

    for iter in range(num_iter):
        print()
        n_pose = len(poses)
        n_point = len(points)
        n_obs = len(points2d)
        print(f"iter {iter+1}/{num_iter}", '-', n_pose, 'poses,', n_point, 'points,', n_obs, 'obs')

        # optimization
        poses = np.asfortranarray(poses.astype(np.float64))
        points = np.array(points, dtype=np.float64, order='F')
        residuals = ba_solver.solve_ba_3(
            camera_params, poses, points, dist_scales,
            poses_i, points_i, points2d,
            fixed_intrinsic or iter < 1, True
        )
        print('intrins:', camera_params)

        # identify outliers
        assert(len(residuals) == n_obs)
        residuals = np.abs(residuals)
        mask = residuals < BA_OUTLIER_Z * np.mean(residuals, axis=0, keepdims=True)
        mask = mask[:,0] & mask[:,1] & mask[:,2]
        inliners = np.array(np.where(mask)).flatten()

        # filter outliers
        points2d = points2d[inliners]
        poses_i = poses_i[inliners]
        points_i = points_i[inliners]
        obs_map = obs_map[inliners]
        dist_scales = dist_scales[inliners]
        # remove points with too few observations
        point_idx = np.zeros(n_point, dtype=np.int32)
        for i in points_i:
            point_idx[i] += 1
        mask_p = point_idx >= PoseGraph.min_obs_ba
        inliners_p = np.array(np.where(mask_p)).flatten()
        points = points[inliners_p]
        points_map = points_map[inliners_p]
        # clean up observations after point removal
        mask_e = mask_p[points_i]
        inliners_e = np.array(np.where(mask_e)).flatten()
        points2d = points2d[inliners_e]
        poses_i = poses_i[inliners_e]
        points_i = np.cumsum(mask_p)[points_i[inliners_e]]-1
        dist_scales = dist_scales[inliners_e]
        obs_map = obs_map[inliners_e]

    poses = [exp_so3t(so3t) for so3t in poses]
    observations = list(zip(poses_i.tolist(), points_i.tolist(), points2d))
    return camera_params, poses, points, observations, points_map, obs_map



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
    ax.plot(vertices[0], vertices[2], vertices[1], '-', zorder=np.inf)

def plot_cameras(ax, transforms, sc=1.0):
    rotations = [t[:3, :3] for t in transforms]
    translations = [t[:3, 3:] for t in transforms]
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

    BottomUpFrameSequence(frames)

