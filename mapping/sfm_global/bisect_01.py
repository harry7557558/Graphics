import cv2 as cv
import numpy as np
import scipy.sparse
import scipy.optimize
import scipy.sparse.linalg
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from typing import Union, Optional, Dict, Tuple, List

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


class PoseGraph:
    min_obs: int = 3
    """Minimum number of observations per point"""

    def __init__(self, frames, matches, frame_id, w2cs):
        assert len(frame_id) == len(w2cs)
        self.frame_id = frame_id
        self.w2c = w2cs

        self.fmap = {}
        for i, fi in enumerate(frame_id):
            self.fmap[fi] = i

        # merge and filter list of points
        num_points = np.cumsum([len(frames[fi].keypoints) for fi in frame_id])
        num_points, points_psa = num_points[-1], num_points-num_points[0]
        points_djs = DisjointSet(num_points)
        points_count = [0]*num_points
        self.points_psa = np.concatenate((points_psa, [num_points]))

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
        # points_djs.union_queue()
        points_djs.union_queue_clique3()
        for i in range(num_points):
            points_djs.find(i)

        is_remain = np.array([int(
            points_djs.find(i) == i and points_count[i] >= self.min_obs)
            for i in range(num_points)])
        remain_idx, remain_map = self.update_remain_maps(is_remain)

        # create list of observations
        def get_observations():
            observations = []
            for i, fi in enumerate(frame_id):
                for j, kp in enumerate(frames[fi].keypoints):
                    pi = remain_map[points_djs.find(points_psa[i]+j)]
                    if pi < 0:
                        continue
                    assert remain_idx[pi] == points_djs.find(points_psa[i]+j)
                    assert points_djs.find(remain_idx[pi]) == remain_idx[pi]
                    observations.append((fi, pi, kp[:2]))
            return observations
        observations = get_observations()

        # filter observations
        points_count = [0]*len(remain_idx)
        for pi, i, uv in observations:
            points_count[i] += 1
        is_remain[remain_idx] &= (np.array(points_count) >= self.min_obs)
        remain_idx, remain_map = self.update_remain_maps(is_remain)
        observations = get_observations()

        # sanity check
        point_count = np.zeros(len(remain_idx), dtype=np.int32)
        for pi, i, uv in observations:
            point_count[i] += 1
        assert (point_count >= self.min_obs).all()

        # save
        self.is_remain = is_remain
        self.remain_idx = remain_idx  # original indices of remaining points
        self.remain_map = remain_map  # map original indices to remaining indices
        self.observations = observations

        # initialize points
        self.points = -1.0+2.0*np.random.random((len(remain_idx), 3))
        # TODO: triangulation

    def update_remain_maps(self, is_remain):
        num_points = self.points_psa[-1]
        remain_idx = np.array(np.where(is_remain)).flatten()
        remain_map = -np.ones(num_points, dtype=np.int32)
        remain_map[remain_idx] = np.cumsum(is_remain)[remain_idx] - 1
        return remain_idx, remain_map


class BottomUpFrameSequence:
    n_bottom: int = 4
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

        self.initial_match()
        self.run_ba()
        self.run_ba()
        return

    def _intrins_init(self):
        frame = self.frames[0]
        h, w = frame.shape[:2]
        f = 0.4 * np.hypot(h, w)
        return np.array([f, w/2, h/2])

    def match_feature_pair(self, i, j):
        match = FramePair(self.frames[i], self.frames[j], self.intrins)
        # print(i, j,  match.confidence)
        if match.confidence > self.config.image_match_th or j <= i+2:
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
        print()

        # update results
        points_i = np.array(points_i)[points_map]
        observations_i = np.array(observations_i)[obs_map]
        pi0 = 0
        for pgi, pg in enumerate(self.pose_graphs):
            np0, no0 = len(pg.points), len(pg.observations)

            pg.observations = [(fi, pi-pi0, uv) for i, (fi, pi, uv) in
                               zip(observations_i, observations) if i == pgi]
            pg.w2c = [poses[i] for i in pg.frame_id]
            pg.points = points[np.where(points_i==pgi)]
            pi0 += len(pg.points)

            # update indices
            pmap = points_map[np.where(points_i==pgi)]
            pmap = pmap - np.amin(pmap)
            outliers = sorted(set(range(len(pg.remain_idx))).difference(pmap))
            pg.is_remain[pg.remain_idx[outliers]] = False
            pg.remain_idx, pg.remain_map = pg.update_remain_maps(pg.is_remain)
            assert len(pg.remain_idx) == len(pmap)

            print(f"{pg.frame_id[0]}-{pg.frame_id[-1]} of {len(self.frames)} - "
                  f"{np0}->{len(pg.points)} points, {no0}->{len(pg.observations)} observations")

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

    def plot(self):
        ax = plt.subplot(projection="3d")
        if hasattr(self, 'lc_list'):
            for i, j in self.lc_list:
                ti = -self.poses[i][0].T @ self.poses[i][1]
                tj = -self.poses[j][0].T @ self.poses[j][1]
                ti, tj = ti[[0, 2, 1]], tj[[0, 2, 1]]
                plt.plot(*zip(ti, tj), 'k-', zorder=5)
        plot_cameras(ax, [_[0] for _ in self.poses], [_[1] for _ in self.poses])
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



# Bundle adjustment

BA_OUTLIER_Z = 3

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
        mask_p = point_idx >= PoseGraph.min_obs
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
    frames = FrameSequence(video_filename, max_frames=18, skip=5)

    BottomUpFrameSequence(frames)

