import cv2 as cv
import numpy as np
import scipy.sparse
import scipy.optimize
import scipy.sparse.linalg
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema
import networkx as nx
import concurrent.futures

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

def filter_features(kps, descs=None, keeps=None, r=10, k_max=20):
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
        dist, inds = kd_tree.query(data[i,:], k=k_max, distance_upper_bound=r)
        for j in inds:
            if j>i and (keeps is None or j not in keeps):
                removed.add(j)

    kp_filtered = [kp for i,kp in enumerate(kps) if i not in removed]
    descs_filtered = None
    if descs is not None:
        descs = descs[order]
        descs_filtered = np.array([desc for i,desc in enumerate(descs) if i not in removed])
    # print('filtered', len(kp_filtered), 'of', N, 'features')

    if keeps is not None:
        index_filtered = [order[i] for i in range(N) if i not in removed]
        index_map = -np.ones(N, dtype=np.int32)
        index_map[(index_filtered)] = np.arange(len(index_filtered))
        return kp_filtered, descs_filtered, index_map
    return kp_filtered, descs_filtered

def extract_features(img, num_features=8192, filter=False, detectors=[]):
    if len(detectors) == 0:
        detectors.append(cv.SIFT_create(num_features))
        detectors.append(cv.ORB_create(num_features))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    keypoints, descriptors = detectors[0].detectAndCompute(gray, None)
    if filter:
        r = 0.01 * np.linalg.norm(img.shape[:2])
        keypoints, descriptors = filter_features(keypoints, descriptors, r=r)
    keypoints = [(*p.pt, p.size, p.angle, p.response) for p in keypoints]
    keypoints_orb, descriptors_orb = detectors[1].detectAndCompute(gray, None)
    keypoints_orb = [(*p.pt, p.size, p.angle, p.response) for p in keypoints_orb]
    return (img, np.array(keypoints), descriptors, np.array(keypoints_orb), descriptors_orb)


# Video loading

class Frame:
    def __init__(self, image):
        self.img = image
        _, self.keypoints, self.descriptors, self.keypoints_orb, self.descriptors_orb \
            = extract_features(image, 1536, False)

    @property
    def shape(self):
        return self.img.shape

    def plot(self):
        img = np.empty_like(self.img)
        img[:, :self.img.shape[1], :] = self.img
        keypoints = [cv.KeyPoint(*p) for p in self.keypoints]
        cv.drawKeypoints(img, keypoints, img, (0, 255, 0))
        cv.imshow("Features", img)

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
                # sc = 1.0
                image_shape = np.round([sc*frame.shape[1], sc*frame.shape[0]])
                print("Image shape:", image_shape)
            frame = cv.resize(frame, image_shape.astype(np.int32))

            cv.imshow("Video", frame)
            cv.waitKey(1)
            self.append(frame)

            if len(self) >= max_frames:
                break

        cap.release()
        cv.destroyAllWindows()

        def process_one_frame(args):
            self, i = args
            self[i] = Frame(self[i])
            print(f"{len(self[i].keypoints)} features extracted for frame {i}/{len(self)}")
        if True:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(process_one_frame, (self, i)) for i in range(len(self))]
        else:
            for i in range(len(self)):
                process_one_frame((self, i))
                self[i].plot()
                cv.waitKey(1)
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
    if R.ndim > 2:
        return np.array([pack_pose(R, t) for R, t in zip(R, t)])
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:] = t.reshape((3, 1))
    return T

def unpack_pose(T):
    return T[..., :3, :3], T[..., :3, 3:]

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
    matcher_orb = cv.BFMatcher(cv.NORM_HAMMING)
    matcher_sift = cv.BFMatcher()

    def __init__(self, frame0: Frame, frame1: Frame, intrins: List[float]):
        self.confidence = 0.0
        res_orb = self.match_frames(
            FramePair.matcher_orb,
            frame0.keypoints_orb, frame0.descriptors_orb,
            frame1.keypoints_orb, frame1.descriptors_orb,
            frame0.shape, intrins
        )
        if res_orb is None:
            return
        res = self.match_frames(
            FramePair.matcher_sift,
            frame0.keypoints, frame0.descriptors,
            frame1.keypoints, frame1.descriptors,
            frame0.shape, intrins
        )
        if res is None:
            return
        T0, T1 = res[-1], res_orb[-1]
        if res[0] > 0.2 and res_orb[0] > 0.2 and False:
            s0, s1 = log_so3t(T0), log_so3t(T1)
            s = s0 * s1
            if s.sum() < 0.0:
                print("Ignore: opposite direction")
                return
            if s[:3].sum() < 0.0 and s[3:].sum() < 0.0:
                print("Ignore: opposite direction")
                return
        self.confidence, self.idx0s, self.idx1s, self.T = res

    @staticmethod
    def match_frames(
            matcher,
            keypoints0, descriptors0,
            keypoints1, descriptors1,
            frame_shape: List[int],
            intrins: List[float]
        ):

        # feature matching
        matches = matcher.match(descriptors0, descriptors1)

        # filter good matches
        if True:
            dist_th = np.hypot(*frame_shape[:2]) / 20
            good_matches = []
            min_dist = min([match.distance for match in matches])
            for match in matches:
                if match.distance <= max(2 * min_dist, dist_th):
                    good_matches.append(match)
            matches = good_matches

        # plt.figure()
        # plt.hist([match.distance for match in matches])
        # plt.show()

        # get matched points
        points0 = keypoints0[:, :2]
        points1 = keypoints1[:, :2]
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
        if inliner_mask_f is None:
            mat_f = np.zeros((3, 3))
            inliner_mask_f = np.zeros((len(mpoints0),))
        # print(inliner_mask_f.sum(), '/', len(inliner_mask_f))
        elif False:  # TODO
            inliner_mask_f |= True
        inliners = np.where(inliner_mask_f.flatten())
        fpoints0, fpoints1 = mpoints0[inliners], mpoints1[inliners]
        idx0 = np.array(idx0)[inliners]
        idx1 = np.array(idx1)[inliners]
        if len(fpoints0) < 5:
            return None

        # essential matrix
        f, cx, cy = intrins[:4]
        K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
        mat_e, inliner_mask_e = cv.findEssentialMat(fpoints0, fpoints1, cameraMatrix=K, method=0)
        if inliner_mask_e is None or mat_e is None or mat_e.shape != (3, 3):
            return None
        # print(inliner_mask_e.sum(), '/', len(inliner_mask_e))
        n, R, t, _ = cv.recoverPose(mat_e, fpoints0, fpoints1, cameraMatrix=K)
        # print(R); print(t)
        T = pack_pose(R, t)

        # update
        confidence = inliner_mask_e.sum() / len(matches)
        # print(inliner_mask_e.sum(), '/', len(inliner_mask_e))
        inliners = np.where(inliner_mask_e.flatten())
        idx0s = idx0[inliners]
        idx1s = idx1[inliners]

        return confidence, idx0s, idx1s, T



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


class MatchedFrameSequence:
    min_obs: int = 3
    match_cache_path = "cache/matches.npy"
    ba_initial_cache_path = "cache/ba_initial.npy"

    def __init__(self, frames: FrameSequence, max_sw=10):
        self.frames = frames
        self.matches = {}  # type: Dict[Tuple[int, int], FramePair]
        self.intrins = self._intrins_init()

        self.match_features(max_sw)

        self.run_initial_ba()
        self.plot()

        self.close_loop()
        self.plot()

        self.run_final_ba()
        self.plot()

    def plot(self):
        ax = plt.subplot(projection="3d")
        if hasattr(self, 'lc_list'):
            R, t = unpack_pose(self.poses)
            for i, j in self.lc_list:
                ti = -R[i].T @ t[i]
                tj = -R[j].T @ t[j]
                ti, tj = ti[[0, 2, 1]], tj[[0, 2, 1]]
                plt.plot(*zip(ti, tj), 'k-', zorder=5)
        plot_cameras(ax, self.poses)
        set_axes_equal(ax)
        c, s = self.get_point_appearance()
        ax.scatter(self.points[:,0], self.points[:,2], self.points[:,1],
                alpha=0.5, s=s, c=c, zorder=0)
        plt.show()

    def get_point_appearance(self):
        colors = np.zeros((len(self.points), 3), dtype=np.float32)
        counts = np.zeros(len(self.points))
        for fi, pi, uv in self.observations:
            i, j = np.round(uv).astype(np.int32)
            color = self.frames[fi].img[j, i]
            colors[pi] += color
            counts[pi] += 1
        assert (counts > 0).all()
        colors = colors / counts[:, None]
        colors = np.flip(colors, -1)
        return colors / 255.0, np.cbrt(counts/3)

    def _intrins_init(self):
        frame = self.frames[0]
        h, w = frame.shape[:2]
        # f = 0.4 * np.hypot(h, w)
        f = 0.6085 * np.hypot(h, w)
        return np.array([f, w/2, h/2])

    def match_features(self, max_sw: int):
        try:
            self.matches = np.load(self.match_cache_path, allow_pickle=True).tolist()
            print(len(self.matches), 'matches loaded', end='\n\n')
        except FileNotFoundError:
            pass
        if len(self.matches.keys()) > 0:
            return

        print("==== Feature Matching ====")
        def process_one(args):
            self, i = args
            num_matches = 0
            for j in range(i+1, min(i+max_sw+1, len(frames))):
                match = FramePair(self.frames[i], self.frames[j], self.intrins)
                # print(i, j,  match.confidence)
                if (match.confidence > 0.2 or j <= i+2) and match.confidence > 0.0:
                    self.matches[(i, j)] = match
                    num_matches += 1
                # else:
                #     break
            print(f"{i+1}/{len(frames)} - {num_matches} image matches")
        if True:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_one, (self, i)) for i in range(len(self.frames))]
                results = concurrent.futures.as_completed(futures)
        else:
            results = [process_one((self, i)) for i in range(len(self.frames))]
        np.save(self.match_cache_path, self.matches)

        print()

    def rotation_averaging(self):
        print("==== Rotation Averaging ====")

        # Initialization
        rotations = [np.eye(3)*1.0]
        translations = [np.zeros((3,1))*1.0]
        for j in range(1, len(frames)):
            R1s, t1s = [], []
            for i in range(max(j-10, 0), j, 1):
                if (i, j) not in self.matches:
                    continue
                R0, t0 = rotations[i], translations[i]
                match = self.matches[(i, j)]
                R, t = unpack_pose(match.T)
                R1, t1 = R0@R, R0@t+t0
                R1s.append(R1)
                t1s.append(t1)
            if len(R1s) > 0:
                R1 = np.mean(R1s, axis=0)
                U, S, Vh = np.linalg.svd(R1)
                S = np.diag([1, 1, np.linalg.det(U@Vh)])
                R1 = U @ S @ Vh
                t1 = np.mean(t1s, axis=0)
                rotations.append(R1)
                translations.append(t1)
            else:
                rotations.append(rotations[-1])
                translations.append(translations[-1])
            # matches = [cv.DMatch(p[0], p[1], 1.0) for p in zip(match.idx0s, match.idx1s)]
            # draw_matches_flow(self.frames[i], self.frames[j], matches)

        # Optimize poses to refine initial estimate
        terms = [(*key, match.T[:3, :3], match.confidence) for (key, match) in self.matches.items()]

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

        # Do this for translations btw
        translations = np.array(translations)
        translations -= translations.mean(0)
        translations /= 2.0*np.std(translations)

        self.poses = pack_pose(rotations, translations)

        if True:
            ax = plt.subplot(projection="3d")
            plot_cameras(ax, self.poses)
            set_axes_equal(ax)
            plt.show()

    def translation_averaging(self):
        print("==== Translation Averaging ====")

        # Initialization
        rotations = unpack_pose(self.poses)[0]
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
        self.poses = pack_pose(rotations, translations)

        if True:
            ax = plt.subplot(projection="3d")
            plot_cameras(ax, self.poses)
            set_axes_equal(ax)
            plt.show()

    def setup_initial_ba(self):
        """Create list of points and observations"""
        if not hasattr(self, 'poses'):
            # All rotations and translations are world to camera
            self.rotation_averaging()
            # self.translation_averaging()
            print()

        print("==== BA Setup ====")

        self.points_psa = np.concatenate(([0], np.cumsum([len(frame.keypoints) for frame in self.frames])))
        num_points = self.points_psa[-1]
        points_djs = DisjointSet(num_points)
        points_count = [0]*num_points

        edges = []
        for (i, j), m in self.matches.items():
            idx0 = self.points_psa[i] + m.idx0s
            idx1 = self.points_psa[j] + m.idx1s
            for i0, i1 in zip(idx0, idx1):
                points_djs.queue_union(i0, i1)
                edges.append((i0, i1))
                points_count[i0] += 1
                points_count[i1] += 1
        if False:
            # points_djs.union_queue()
            points_djs.union_queue_clique3()
        else:
            max_nodes = min(len(self.frames), 25)
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
        for i, frame in enumerate(self.frames):
            for j, kp in enumerate(frame.keypoints):
                pi = self.remain_map[self.points_psa[i]+j]
                if pi < 0:
                    continue
                self.observations.append((i, pi, kp[:2]))

        self.points = -1.0+2.0*np.random.random((self.num_points_remain, 3))
        self.poses[: :3, 3] = -1.0+2.0*np.random.random(self.poses[: :3, 3].shape)

        print(self.num_points_remain, 'points')
        print(len(self.observations), 'observations')
        print()

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

    def run_initial_ba(self):
        try:
            self.intrins, self.poses, self.points, self.observations \
                  = np.load(self.ba_initial_cache_path, allow_pickle=True).tolist()
            print("BA results loaded -", len(self.poses), 'poses,', len(self.points), 'points,', len(self.observations), 'observations')
            print("Camera intrinsics:", self.intrins.tolist())
            print()
        except FileNotFoundError:
            pass
        else:
            return

        self.setup_initial_ba()

        print("==== Bundle Adjustment ====")

        time0 = perf_counter()

        intrins, poses, points, observations = bundle_adjustment_ceres_3(
        # intrins, poses, points, observations = bundle_adjustment_ceres_3_hierarchy(
            self.intrins, self.poses, self.points, self.observations,
            fixed_intrinsic=True, num_iter=2)

        time1 = perf_counter()
        print()
        print(f"BA solved in {time1-time0:.2f} s")

        self.intrins, self.poses, self.points, self.observations = \
              intrins, poses, points, observations
        np.save(self.ba_initial_cache_path, np.array(
            (intrins, poses, points, observations), dtype=object))

        print()

    def close_loop(self):

        print("==== Loop Closure ====")

        poses_R, poses_t = unpack_pose(self.poses)

        # compute pairwise distance
        dist_mode = ["rotation", "ray_dir", "ray_dist"][1]
        if dist_mode == "rotation":
            dists = poses_R.reshape((-1, 1, 3, 3)) - poses_R.reshape((1, -1, 3, 3))
            dists = (dists**2).sum(axis=(-1, -2))
            dists -= 1.0
        elif dist_mode == "ray_dir":
            dir1 = poses_R[:, :, 2].reshape((-1, 1, 3))
            dir2 = dir1.reshape((1, -1, 3))
            dists = ((dir2-dir1)**2).sum(axis=(-1))
            dists -= 1.0/3.0
        elif dist_mode == "ray_dist":
            dir1 = poses_R[:, 2].reshape((-1, 1, 3))
            dir2 = dir1.reshape((1, -1, 3))
            n = np.cross(dir1, dir2, axisa=-1, axisb=-1)
            pos1 = -np.einsum('kij,ki->kj', poses_R, poses_t).reshape((-1, 1, 3))
            pos2 = pos1.reshape((1, -1, 3))
            dp = pos2 - pos1
            t1 = (np.cross(dir2, n, -1, -1) * dp).sum(-1) / (n*n).sum(-1)
            t2 = (np.cross(dir1, n, -1, -1) * dp).sum(-1) / (n*n).sum(-1)
            p1 = pos1 + dir1 * np.fmax(t1, 0.0)[..., None]
            p2 = pos2 + dir2 * np.fmax(t2, 0.0)[..., None]
            dists = np.linalg.norm(p2-p1, axis=-1)
        # dists = gaussian_filter(dists, sigma=10)
        dists -= min(np.sort(dists.flatten())[len(dists.flatten())//8], 0)
        # plt.imshow(dists); plt.show()

        # find potential matches
        lc_list = []
        for i, dist_i in enumerate(dists):
            minima = argrelextrema(dist_i, np.less)[0]
            minima = minima[np.where(dist_i[minima] <= 0)]
            # plt.plot(dist_i); plt.show()
            lc_list.extend([(*sorted([i, j]),) for j in minima if i != j])
        lc_list = sorted(set(lc_list))
        lc_list = [ij for ij in lc_list if ij not in self.matches]
        # print(lc_list)
        self.lc_list = lc_list
        # self.plot()

        # match images
        lc_list = []
        # multithreading
        def process_pair(args):
            self, i, j = args
            match = FramePair(self.frames[i], self.frames[j], self.intrins)
            if match.confidence > 0.2 and len(match.idx0s) > 20:
                self.matches[(i, j)] = match
                print(i, j, match.confidence, len(match.idx0s), '- accept')
                if False:
                    matches = [cv.DMatch(p[0], p[1], 1.0) for p in zip(match.idx0s, match.idx1s)]
                    draw_matches(self.frames[i], self.frames[j], matches)
                return (i, j)
            return None
        if True:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_pair, (self, i, j)) for i, j in self.lc_list]
                lc_list = [future.result() for future in concurrent.futures.as_completed(futures)
                            if future.result() is not None]
        else:
            lc_list = [process_pair((self, i, j)) for i, j in self.lc_list]
            lc_list = [_ for _ in lc_list if _ is not None]
        print(f"{len(lc_list)}/{len(self.lc_list)} image matches")
        self.lc_list = lc_list
        # self.plot()

        print()

    def run_final_ba(self):

        self.setup_initial_ba()

        print("==== Bundle Adjustment ====")

        time0 = perf_counter()

        intrins, poses, points, observations = bundle_adjustment_ceres_3(
            self.intrins, self.poses, self.points, self.observations,
            fixed_intrinsic=True, num_iter=5)

        time1 = perf_counter()
        print()
        print(f"BA solved in {time1-time0:.2f} s")

        self.intrins, self.poses, self.points, self.observations = \
              intrins, poses, points, observations

        print()


# Bundle adjustment

BA_OUTLIER_Z = 6

def bundle_adjustment_3(camera_params, poses, points, observations):
    """points + poses + camera intrinsics (f,cx,cy)"""

    # params
    n_poses = len(poses)
    n_points = len(points)
    n_obs = len(observations)
    poses = np.array([log_so3t(pose) for pose in poses])
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
        fix_rotation = iter < 1 and num_iter >= 3
        residuals = ba_solver.solve_ba_3(
            camera_params, poses, points, dist_scales,
            poses_i, points_i, points2d,
            fixed_intrinsic or iter < 1, fix_rotation, True
        )
        print('intrins:', camera_params)

        # identify outliers
        residuals = np.abs(residuals)
        mask = residuals < BA_OUTLIER_Z * np.mean(residuals, axis=0, keepdims=True)
        mask |= fix_rotation
        mask = mask[:,0] & mask[:,1] & mask[:,2]
        inliners = np.array(np.where(mask)).flatten()

        # filter outliers
        points2d = points2d[inliners]
        poses_i = poses_i[inliners]
        points_i = points_i[inliners]
        obs_map = obs_map[inliners]
        dist_scales = dist_scales[inliners]
        # remove points with fewer than 3 observations
        point_idx = np.zeros(len(points))
        for i in points_i:
            point_idx[i] += 1
        mask_p = point_idx >= 3
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

    poses = np.array([exp_so3t(so3t) for so3t in poses])
    observations = list(zip(poses_i.tolist(), points_i.tolist(), points2d))
    return camera_params, poses, points, observations

def bundle_adjustment_ceres_3_hierarchy(
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

    tree_b = 3
    poses_indices = [[i] for i in range(len(poses))]
    num_tree, tree_stride = len(poses), tree_b
    while poses_indices[0][-1] != poses_indices[-1][-1]:
        for i in range(0, len(poses)):
            if i % tree_stride == 0:
                num_tree += 1
            poses_indices[i].append(num_tree-1)
        tree_stride *= tree_b
    poses = np.concatenate((poses, np.zeros((num_tree-len(poses), 6), dtype=poses.dtype)))
    poses_indices_flattened = np.array(sum(poses_indices, []))
    poses_indices_psa = np.concatenate([[0], np.cumsum([len(x) for x in poses_indices])])

    pose_reg_indices = [[] for _  in range(num_tree)]
    for i, pi in enumerate(poses_indices):
        for j in pi:
            pose_reg_indices[j].append(i)
    pose_reg_indices_psa = np.concatenate([[0], np.cumsum([len(x) for x in pose_reg_indices])])
    pose_reg_indices = np.array(sum(pose_reg_indices, []))

    for iter in range(num_iter):
        print()
        n_pose = len(poses)
        n_point = len(points)
        n_obs = len(points2d)
        print(f"iter {iter+1}/{num_iter}", '-', n_pose, 'poses,', n_point, 'points,', n_obs, 'obs')

        # optimization
        poses = np.asfortranarray(poses.astype(np.float64))
        points = np.array(points, dtype=np.float64, order='F')
        residuals = ba_solver.solve_ba_3_hierarchy(
            camera_params, poses, points, dist_scales,
            poses_indices_psa.astype(np.int32), poses_indices_flattened.astype(np.int32),
            poses_i, points_i, points2d,
            # pose_reg_indices_psa.astype(np.int32), pose_reg_indices.astype(np.int32),
            [], [],
            fixed_intrinsic or iter < 1, True
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
        obs_map = obs_map[inliners]
        dist_scales = dist_scales[inliners]
        # remove points with fewer than 3 observations
        point_idx = np.zeros(len(points))
        for i in points_i:
            point_idx[i] += 1
        mask_p = point_idx >= 3
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
    for i, indices in enumerate(poses_indices):
        assert indices[0] == i
        T = poses[i]
        for j in indices[1:]:
            T = poses[j] @ T
        poses[i] = T
    poses = np.array(poses[:len(poses_indices)])

    observations = list(zip(poses_i.tolist(), points_i.tolist(), points2d))
    return camera_params, poses, points, observations



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

def plot_cameras(ax, poses, sc=1.0):
    rotations, translations = unpack_pose(poses)
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
    # video_filename = videos.videos[4]
    video_filename = "/media/harry7557558/New Volume/a2rl_gate/r4w1d1_2.mp4"
    frames = FrameSequence(video_filename, max_frames=100, skip=10)

    MatchedFrameSequence(frames)

