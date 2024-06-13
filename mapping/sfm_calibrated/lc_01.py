import cv2 as cv
import numpy as np
import scipy.sparse
import scipy.optimize
from scipy.spatial import KDTree

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from typing import Union, Optional

from ba_solver import ba_solver
from lc_solver import lc_solver



IMG_SHAPE = np.array([960.0, 540.0])
# IMG_SHAPE = np.array([2000.0, 1500.0])

# initial guess
F = 1.0 * np.prod(IMG_SHAPE)**0.5
K = np.array([[F, 0., IMG_SHAPE[0]/2],
              [0., F, IMG_SHAPE[1]/2],
              [0., 0., 1.]])

# global map
camera_params = None  # depends on BA function
all_frames = []  # (keypoints, descriptor)
all_poses = []  # (R, t) | None
all_keypoints = []  # [f][i]: index of 3d point
all_points_frames = []  # list of points [(f,i)]
all_points = []  # 3d points
all_observations = {}  # (fi, pi): uv
vocabulary = lc_solver.Vocabulary()
all_bows = []



# feature detection and matching

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

def extract_features(img, filter=False):
    detector = cv.ORB_create(6000)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return (img, keypoints, descriptors)

def match_feature_pair(frame1, frame2, div_factor=20):

    img_1, keypoints_1, descriptors_1 = frame1
    img_2, keypoints_2, descriptors_2 = frame2

    if div_factor == -1:
        matcher = cv.BFMatcher(cv.NORM_HAMMING)
        matches = matcher.match(descriptors_1, descriptors_2)
        # plt.figure()
        # plt.hist([match.distance for match in matches])
        # plt.show()
        good_matches = []
        min_dist = min([match.distance for match in matches])
        for match in matches:
            if match.distance <= max(2 * min_dist, np.prod(IMG_SHAPE)**0.5 / 20):
            # if True:
                good_matches.append(match)
        matches = good_matches

    else:
        kp1_flat = np.array([kp.pt for kp in keypoints_1], dtype=np.float32)
        if len(kp1_flat) < 2:
            return []
        gray_1 = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)
        gray_2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)
        maxlevel = max(3, int(np.log2(np.prod(IMG_SHAPE)**0.5 / 8)))
        p2, st, err = cv.calcOpticalFlowPyrLK(gray_1, gray_2, kp1_flat, None, maxLevel=maxlevel)

        kp2_flat = np.array([kp.pt for kp in keypoints_2], dtype=np.float32)
        if len(kp2_flat) < 2:
            return []
        tree = KDTree(kp2_flat)
        matches = []
        for i, (p, st) in enumerate(zip(p2, st)):
            dists, js = tree.query(p, k=2)
            for dist, j in zip(dists, js):
                xor_result = np.bitwise_xor(descriptors_1[i], descriptors_2[j])
                dist = np.unpackbits(xor_result).sum()
                matches.append((i, j, dist))
        # plt.figure()
        # plt.hist([match[2] for match in matches])
        # plt.show()

        good_matches = []
        dists_sorted = sorted([match[2] for match in matches])
        dist_th = dists_sorted[max(len(matches)//div_factor, min(len(matches)-1, 10))]
        for (i, j, dist) in matches:
            if dist <= max(dist_th, 2):
                m = cv.DMatch(i, j, np.linalg.norm(kp1_flat[i]-kp2_flat[j]))
                good_matches.append(m)
        matches = good_matches

    # draw_matches(frame1, frame2, matches)
    
    return matches

def draw_matches(frame1, frame2, matches, inliner_mask=None):
    img_1, keypoints_1, descriptors_1 = frame1
    img_2, keypoints_2, descriptors_2 = frame2
    if inliner_mask is None:
        inliner_mask = [True] * len(matches)
    img_matches = np.empty((img_1.shape[0], img_1.shape[1]+img_2.shape[1], 3), dtype=np.uint8)
    img_matches[:, :img_1.shape[1], :] = img_1
    img_matches[:, img_1.shape[1]:, :] = img_2
    for match, inliner in zip(matches, inliner_mask):
        if not inliner:
            continue
        kp1 = keypoints_1[match.queryIdx].pt
        kp2 = keypoints_2[match.trainIdx].pt
        pt1 = (int(kp1[0]), int(kp1[1]))
        pt2 = (int(kp2[0]) + img_1.shape[1], int(kp2[1]))
        cv.line(img_matches, pt1, pt2, (0, 255, 0), 1)
    
    # Display the matches
    cv.imshow("Matches", img_matches)
    cv.waitKey(0)
    cv.destroyAllWindows()



# epipolar geometry and PnP

def pixel2cam(p):
    return np.array([(p[0]-K[0,2])/K[0,0], (p[1]-K[1,2])/K[1,1]])

def triangulation(points1, points2, R1, t1, R2, t2):
    points1 = np.array([pixel2cam(p) for p in points1])
    points2 = np.array([pixel2cam(p) for p in points2])
    T1 = np.hstack((R1, t1))
    T2 = np.hstack((R2, t2))
    pts_4d_homogeneous = cv.triangulatePoints(T1, T2, points1.T, points2.T)
    pts_4d = pts_4d_homogeneous / np.tile(pts_4d_homogeneous[-1, :], (4, 1))
    points_3d = pts_4d[:3, :].T
    return points_3d

def pose_estimation_2d2d(points1, points2, mask_only=False):
    points1 = np.array(points1)
    points2 = np.array(points2)
    if camera_params is not None and len(camera_params) > 4:
        dist_coeffs = camera_params[-4:]
        points1 = cv.undistortImagePoints(points1, K, dist_coeffs)[:,0,:]
        points2 = cv.undistortImagePoints(points2, K, dist_coeffs)[:,0,:]

    f = np.sqrt(K[0,0]*K[1,1])
    c = K[0:2,2]

    # from essential matrix
    essential_matrix, inliner_mask = cv.findEssentialMat(points1, points2, f, c)
    if mask_only:
        return inliner_mask
    n, R, t, _ = cv.recoverPose(essential_matrix, points1, points2, cameraMatrix=K)

    # triangulation
    points = triangulation(points1, points2, np.eye(3), np.zeros((3,1)), R, t)
    inliner_mask &= (points[:,2:] > 0)
    points_r = points @ R.T + t.T
    inliner_mask &= (points_r[:,2:] > 0)

    return (R, t), points, inliner_mask.flatten().astype(np.bool_)

def pose_estimation_3d2d_ransac(points3d, points2d, R0, t0):
    points3d = np.array(points3d)
    points2d = np.array(points2d)
    if camera_params is not None and len(camera_params) > 4:
        dist_coeffs = camera_params[-4:]
        points2d = cv.undistortImagePoints(points2d, K, dist_coeffs)[:,0,:]

    _, r, t, inliners = cv.solvePnPRansac(
        points3d, points2d, K, None,
        useExtrinsicGuess=True, rvec=cv.Rodrigues(R0)[0], tvec=1.0*t0,
        reprojectionError=0.015*np.prod(IMG_SHAPE)**0.5, confidence=0.99,
        flags=cv.SOLVEPNP_ITERATIVE)
    R, _ = cv.Rodrigues(r)

    inliner_mask = np.zeros(len(points3d), dtype=np.bool_)
    inliner_mask[inliners] = True

    return (R, t), inliner_mask

def pose_estimation_3d2d(points3d, points2d, R0, t0):
    points3d = np.array(points3d)
    points2d = np.array(points2d)
    if camera_params is not None and len(camera_params) > 4:
        dist_coeffs = camera_params[-4:]
        points2d = cv.undistortImagePoints(points2d, K, dist_coeffs)[:,0,:]

    _, r, t = cv.solvePnP(
        points3d, points2d, K, None,
        useExtrinsicGuess=True, rvec=cv.Rodrigues(R0)[0], tvec=1.0*t0,
        flags=cv.SOLVEPNP_ITERATIVE)
    R, _ = cv.Rodrigues(r)

    points_r = points3d @ R.T + t.T
    points_proj = points_r[:,:2] @ K[:2,:2] / points_r[:,2:] + K[:2,2:].T
    residuals = points_proj-points2d
    cutoff = 3 * 1.414 * np.std(residuals)
    inliner_mask = (points_r[:,2] > 0) & (np.linalg.norm(residuals, axis=1) < cutoff)

    if np.sum(inliner_mask.astype(np.int32)) >= 4:
        inliners = np.where(inliner_mask)
        _, r, t = cv.solvePnP(
            points3d[inliners], points2d[inliners], K, None,
            useExtrinsicGuess=True, rvec=1.0*r, tvec=1.0*t,
            flags=cv.SOLVEPNP_ITERATIVE)
        R, _ = cv.Rodrigues(r)

    return (R, t), inliner_mask



# bundle adjustment

def exp_so3t(T):
    phi = T[0:3]
    t = T[3:6]
    R, _ = cv.Rodrigues(phi)
    return R, t

def log_so3t(R, t):
    assert np.linalg.norm(R@R.T-np.eye(3)) < 1e-6
    phi, _ = cv.Rodrigues(R)
    return np.concatenate((phi.flatten(), t.flatten()))

BA_OUTLIER_Z = 8
BA_TH_RMSE = np.prod(IMG_SHAPE)**0.5 / 1000
BA_SW = 30
LC_MUTE = 5

def bundle_adjustment_ceres_3(camera_params, poses, points, observations, force=True):
    """points + poses + camera intrinsics (fx,fy,cx,cy,*dist_coeffs)"""

    # params
    if camera_params is None:
        f0 = np.sqrt(K[0,0]*K[1,1])
        cx0, cy0 = K[:2,2]
        camera_params = np.array([f0, cx0, cy0])
    n_pose = len(poses)
    n_point = len(points)
    n_obs = len(observations)
    poses = np.array([log_so3t(*pose) for pose in poses])
    points_2d = np.array([uv for (pose_i, point_i, uv) in observations])
    params_init = np.concatenate((camera_params, poses.flatten(), points.flatten()))
    poses_i = np.array([o[0] for o in observations])
    points_i = np.array([o[1] for o in observations])

    # function
    def fun(params):
        global K
        f, cx, cy = params[:3]
        K = np.array([[f, 0., cx], [0., f, cy], [0., 0., 1.]])
        so3ts = params[3:6*n_pose+3].reshape((n_pose, 6))
        poses = [exp_so3t(so3t) for so3t in so3ts]
        R = np.array([p[0] for p in poses])
        t = np.array([p[1] for p in poses])
        points_3d = params[6*n_pose+3:].reshape((n_point, 3))
        points_r = np.einsum('kij,kj->ki', R[poses_i], points_3d[points_i]) + t[poses_i]
        points_c = points_r @ K.T
        points_proj = points_c[:,0:2] / points_c[:,2:3]
        residual = (points_proj-points_2d).flatten()
        # return residual
        # return np.arcsinh(residual)
        delta = 2
        qr_residual = np.sign(residual) * np.sqrt(delta*np.fmax(2.0*np.abs(residual)-delta, 0.0))
        return residual + (qr_residual-residual) * (np.abs(residual) > delta)

    residuals = fun(params_init)
    rmse = np.mean(residuals**2)**0.5
    print('rmse before:', rmse)
    if rmse < BA_TH_RMSE and not force:
        return None

    # optimization
    poses = np.asfortranarray(poses.astype(np.float64))
    points = np.array(points, dtype=np.float64, order='F')
    old_poses = np.array(poses)
    old_points = np.array(points)
    residuals = ba_solver.solve_ba_3(
        camera_params, poses, points,
        poses_i, points_i, points_2d,
        force
    )
    # print(np.median(np.abs(poses-old_poses)), np.median(np.abs(points-old_points)))

    residuals = np.abs(residuals)
    mask = residuals > BA_OUTLIER_Z*np.mean(residuals)
    outliers = np.where(mask[:,0] | mask[:,1])

    params_init = np.concatenate((camera_params, poses.flatten(), points.flatten()))
    residuals = fun(params_init)
    rmse = np.mean(residuals**2)**0.5
    print('rmse after:', rmse)

    poses = [exp_so3t(so3t) for so3t in poses]
    return camera_params, poses, points, outliers

def bundle_adjustment_ceres_8(camera_params, poses, points, observations, force=True):
    """points + poses + camera intrinsics (fx,fy,cx,cy,*dist_coeffs)"""

    # params
    n_int = 8  # number of camera intrinsics, 3+len(dist_coeffs)
    if camera_params is None or len(camera_params) < n_int:
        camera_params = np.array([K[0,0], K[1,1], K[0,2], K[1,2]])
        camera_params = np.concatenate((camera_params[:4], [0]*(n_int-4)))
    n_pose = len(poses)
    n_point = len(points)
    n_obs = len(observations)
    poses = np.array([log_so3t(*pose) for pose in poses])
    points_2d = np.array([uv for (pose_i, point_i, uv) in observations])
    params_init = np.concatenate((camera_params, poses.flatten(), points.flatten()))
    poses_i = np.array([o[0] for o in observations])
    points_i = np.array([o[1] for o in observations])

    # cost function for early termination
    def fun(params):
        global K
        fx, fy, cx, cy, k1, k2, p1, p2 = params[:n_int]
        K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
        so3ts = params[n_int:n_int+6*n_pose].reshape((n_pose, 6))
        poses = [exp_so3t(so3t) for so3t in so3ts]
        R = np.array([p[0] for p in poses])
        t = np.array([p[1] for p in poses])
        points_3d = params[n_int+6*n_pose:].reshape((n_point, 3))
        points_r = np.einsum('kij,kj->ki', R[poses_i], points_3d[points_i]) + t[poses_i]
        points_proj = points_r[:,0:2] / points_r[:,2:3]
        x, y = points_proj[:,0], points_proj[:,1]
        r2 = x**2 + y**2
        distortion = 1.0 + r2 * (k1 + k2 * r2)
        points_proj[:,0] = x * distortion + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        points_proj[:,1] = y * distortion + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        points_proj = points_proj @ K[:2,:2] + K[0:2,2]
        residual = (points_proj-points_2d).flatten()
        # return residual
        # return np.arcsinh(residual)
        delta = 2
        qr_residual = np.sign(residual) * np.sqrt(delta*np.fmax(2.0*np.abs(residual)-delta, 0.0))
        return residual + (qr_residual-residual) * (np.abs(residual) > delta)

    residuals = fun(params_init)
    rmse = np.mean(residuals**2)**0.5
    print('rmse before:', rmse)
    if rmse < BA_TH_RMSE and not force:
        return None

    # optimization
    poses = np.asfortranarray(poses.astype(np.float64))
    points = np.array(points, dtype=np.float64, order='F')
    old_poses = np.array(poses)
    old_points = np.array(points)
    residuals = ba_solver.solve_ba_8(
        camera_params, poses, points,
        poses_i, points_i, points_2d,
        force
    )
    # print(np.median(np.abs(poses-old_poses)), np.median(np.abs(points-old_points)))

    residuals = np.abs(residuals)
    mask = residuals > BA_OUTLIER_Z*np.mean(residuals)
    outliers = np.where(mask[:,0] | mask[:,1])

    params_init = np.concatenate((camera_params, poses.flatten(), points.flatten()))
    residuals = fun(params_init)
    rmse = np.mean(residuals**2)**0.5
    print('rmse after:', rmse)

    poses = [exp_so3t(so3t) for so3t in poses]
    return camera_params, poses, points, outliers

def bundle_adjustment_update(frames=None, fix_start=True):
    global camera_params
    global all_observations

    if frames is None:
        frames = range(len(all_poses))
    frames = [i for i in frames if all_poses[i] is not None]

    # poses
    poses = []
    poses_map = []
    poses_invmap = {}
    for i, fi in enumerate(frames):
        poses.append(all_poses[fi])
        poses_map.append(fi)
        poses_invmap[fi] = i

    # observations and points
    observations = []
    observations_map = []

    # points
    points = []
    points_map = []
    points_invmap = {}
    for i, ((fi, pi), uv) in enumerate(all_observations.items()):
        if fi not in poses_invmap:
            continue
        if pi == -1 or all_points[pi] is None:
            continue
        if pi not in points_invmap:
            points_invmap[pi] = len(points)
            points.append(all_points[pi])
            points_map.append(pi)
        observations.append((poses_invmap[fi], points_invmap[pi], uv))
        observations_map.append(i)

    # run bundle adjustment
    print(len(frames), 'frames for ba')
    ba_result = bundle_adjustment(
        camera_params, poses, np.array(points), observations,
        force=not fix_start)
    if ba_result is None:
        print("skip BA")
        return
    camera_params, poses_updated, points_updated, outliers = ba_result
    if len(outliers[0]) > 0:
        print(f"{len(outliers[0])}/{len(observations)} outliers")

    # align to first pose
    if fix_start:
        def camera_to_mat4(R, t):
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.flatten()
            return np.linalg.inv(T)
        T0 = camera_to_mat4(*poses[0])
        T1 = camera_to_mat4(*poses_updated[0])
        Tt = T0 @ np.linalg.inv(T1)
        for i in range(len(poses_updated)):
            T = camera_to_mat4(*poses_updated[i])
            T = np.linalg.inv(Tt @ T)
            poses_updated[i] = (T[:3,:3], T[:3,3])
        points_updated = points_updated @ Tt[:3,:3].T + Tt[:3,3:].T

    # put back
    for i, (R, t) in zip(poses_map, poses_updated):
        all_poses[i] = (R, t.reshape((3,1)))
    for i, point in zip(points_map, points_updated):
        all_points[i] = point
    outliers = set(*outliers)
    all_observations_new = {}
    for i, ((fi, pi), uv) in enumerate(all_observations.items()):
        if i not in outliers:
            all_observations_new[(fi, pi)] = uv
    all_observations = all_observations_new



# loop closure

def find_similar_frames(bow, max_checks=50):
    scores = {}
    # for i, bow1 in enumerate(all_bows):
    skip = max(len(all_bows)//max_checks, 1)
    start = np.random.randint(skip)
    for i in range(start, len(all_bows), skip):
        bow1 = all_bows[i]
        if i >= len(all_poses) or all_poses[i] is None:
            continue
        score = bow.score(bow, bow1)
        if score > 0.0:
            scores[i] = score
    return sorted(scores.items(), key=lambda _: -_[1])



# incremental reconstruction

def find_frame_matches(frame1_i: int, frame2_i: int):
    pmap = {}  # point index: (kp index 1, kp index 2)
    for kpi, pi in enumerate(all_keypoints[frame1_i]):
        if pi != -1:
            pmap[pi] = kpi
    matches = []  # (point index, kp index 1, kp index 2)
    for kpi, pi in enumerate(all_keypoints[frame2_i]):
        if pi in pmap:
            matches.append((pi, pmap[pi], kpi))
    return matches

def add_frame_init(frame):
    global camera_params
    global all_points

    N = len(all_poses)
    img, keypoints, descriptors = frame

    # try matching until success
    success_pair = None
    for fi, frame0 in enumerate(all_frames):
        if fi >= N:
            break
        img0, keypoints0, descriptors0 = frame0

        matches = find_frame_matches(fi, N)
        if len(matches) < 9:
            continue

        points0 = [keypoints0[ki0].pt for (pi,ki0,ki) in matches]
        points = [keypoints[ki].pt for (pi,ki0,ki) in matches]
        pose, points3d, inliner_mask = pose_estimation_2d2d(points0, points)
        if pose is None:
            continue
        R, t = pose

        if (1-inliner_mask.astype(np.int32)).sum() <= len(inliner_mask)//10+1:
            success_pair = (fi, N)
            break

    # matching fail
    if success_pair is None:
        all_poses.append(None)
        return False

    # outlier rejection
    points_proj = points3d[:,:2] @ K[:2,:2] / points3d[:,2:] + K[:2,2:].T
    residuals = points_proj-points0
    cutoff = 3 * 1.414 * np.std(residuals)
    outlier_mask = np.linalg.norm(residuals, axis=1) > cutoff
    inliner_mask &= ~outlier_mask
    
    # add data to global map
    all_poses[fi] = (np.eye(3), np.zeros((3,1)))
    all_poses.append((R, t))
    for j, ((pi,ki0,ki), p0, p, p3d) in enumerate(zip(matches, points0, points, points3d)):
        if not inliner_mask[j]:
            continue
        all_points[pi] = p3d
        all_observations[(fi, pi)] = p0
        all_observations[(N, pi)] = p
        assert all_keypoints[fi][ki0] == pi
        assert all_keypoints[N][ki] == pi
    print(len(all_points), 'initial points')

    return True

def add_frame_incremental(frame: Union[int, tuple]):

    if isinstance(frame, int):
        img, keypoints, descriptors = all_frames[frame]
        N = frame
    else:
        img, keypoints, descriptors = frame
        N = len(all_poses)

    SW = 8  # sliding window
    min_i = max(N-SW, 0)
    count = 0
    for i in range(N-1, -1, -1):
        if all_poses[i] is not None:
            min_i = min(min_i, i)
            count += 1
            if count >= SW:
                break
    max_i = min(N+SW, len(all_poses)) if isinstance(frame, int) else N

    # find new pose using PnP
    success_pair = None
    pts_3d, pts_2d = [], []
    for i in range(max_i-1, min_i-1, -1):
        if all_poses[i] is None:
            continue
        img_0, keypoints_0, descriptors_0 = all_frames[i]
        matches = find_frame_matches(i, N)

        # PnP points
        pnp_matches = []
        for mi, (pi, ki0, ki) in enumerate(matches):
            assert pi == all_keypoints[i][ki0]
            if all_points[pi] is None:
                continue
            all_keypoints[N][ki] = pi
            pts_3d.append(all_points[pi])
            pts_2d.append(keypoints[ki].pt)
            pnp_matches.append(mi)
        # if len(pts_3d) < 4:
        if len(pts_3d) < 10:
            print(len(pts_3d), 'point pairs for PnP - too few')
            continue
        print(len(pts_3d), 'point pairs for PnP')

        # solve PnP
        for di in range(1,len(all_poses)):
            if i-di >= 0 and all_poses[i-di] is not None:
                R0, t0 = all_poses[i-di]
                break
            elif i+di < N and all_poses[i+di] is not None:
                R0, t0 = all_poses[i+di]
                break
        # (R, t), inliner_mask_pnp = pose_estimation_3d2d_ransac(pts_3d, pts_2d, R0, t0)
        (R, t), inliner_mask_pnp = pose_estimation_3d2d(pts_3d, pts_2d, R0, t0)
        points_0 = [keypoints_0[ki0].pt for (pi, ki0, ki) in matches]
        points = [keypoints[ki].pt for (pi, ki0, ki) in matches]
        try:
            points_3d = triangulation(points_0, points, *all_poses[i], R, t)
        except cv.error as e:
            print(e.err)
            continue
        num_invalid, num_point = np.sum(1-inliner_mask_pnp), len(pts_3d)
        if num_invalid < num_point//10+1:
            success_pair = (i, N)
            break
        else:
            print(f"PnP failed ({num_invalid}/{num_point} invalid)")
            pts_2d = [p for (p, valid) in zip(pts_2d, inliner_mask_pnp) if valid]
            pts_3d = [p for (p, valid) in zip(pts_3d, inliner_mask_pnp) if valid]

    if success_pair is None:
        if not isinstance(frame, int):
            all_poses.append(None)
        return False

    # outlier rejection
    inliner_mask = np.ones(len(matches), dtype=np.bool_)
    for mi, valid in zip(pnp_matches, inliner_mask_pnp):
        if not valid:
            inliner_mask[mi] = False
    
    # add data to global map
    if isinstance(frame, int):
        all_poses[frame] = (R, t)
    else:
        all_poses.append((R, t))
    num_new_points = 0
    for j, (p0, p, p3d, (pi, ki0, ki), valid) in enumerate(zip(
            points_0, points, points_3d, matches, inliner_mask)):
        if not valid:
            continue
        assert pi == all_keypoints[i][ki0]
        assert pi == all_keypoints[N][ki]
        # add keypoint
        if all_points[pi] is None:
            all_points[pi] = p3d
            num_new_points += 1
        all_observations[(i, pi)] = p0
        all_observations[(N, pi)] = p
    print(num_new_points, 'new points added')

    return True

def add_frame(img, previous_ba=[-1], previous_lc=[-1]):
    frame = extract_features(img)  # (img, keypoints, descriptors)
    N = len(all_poses)
    if N == 0:
        frame = (frame[0], *filter_features(frame[1], frame[2]))

    # add matched features
    success = False
    for fi in range(N-1, -1, -1):
        previous_frame = all_frames[fi]
        div_factor = 20 if N == 1 else np.clip(len(frame[1])//len(previous_frame[1]), 2, 50)
        print('div factor:', div_factor)
        matches = match_feature_pair(previous_frame, frame, div_factor)
        print(len(matches), "matches")
        if len(matches) < 10:
            continue

        # skip frames that are too close
        is_init = all_poses.count(None) == len(all_poses)
        if fi == (0 if is_init else N-1):
            dists = [m.distance for m in matches]
            sc = 0.05 if fi == 0 else 0.02
            if np.median(dists) < sc * np.prod(IMG_SHAPE)**0.5:
                print("skip frame - too close")
                return

        # clean features
        keeps = [m.trainIdx for m in matches]
        keypoints, descriptors, index_map = filter_features(frame[1], frame[2], keeps)
        frame = (frame[0], keypoints, descriptors)
        for m in matches:
            m.trainIdx = index_map[m.trainIdx]
            assert m.trainIdx != -1
        all_frames.append(frame)
        all_keypoints.append([-1]*len(frame[1]))

        # add matches
        for m in matches:
            j = all_keypoints[fi][m.queryIdx]
            if j == -1:
                all_keypoints[fi][m.queryIdx] = len(all_points)
                all_keypoints[N][m.trainIdx] = len(all_points)
                all_points_frames.append([(fi,m.queryIdx), (N,m.trainIdx)])
                all_points.append(None)
            else:
                all_keypoints[N][m.trainIdx] = j
                all_points_frames[j].append((N,m.trainIdx))
        
        success = True
        break

    # add vocabulary
    bow = vocabulary.transform(frame[2])
    all_bows.append(bow)

    # usually first frame
    if not success:
        all_frames.append(frame)
        all_keypoints.append([-1]*len(frame[1]))
        all_poses.append(None)
        return

    # add the current frame
    if all_poses.count(None) == len(all_poses):
        status = add_frame_init(frame)
        print("initialization success:", status)
        return
    status = add_frame_incremental(frame)
    print("add frame success:", status)
    if status:
        # fix_start = status
        fix_start = True
        f_start = max(min(previous_ba[0]-BA_SW//2, N-BA_SW+1), 0)
        bundle_adjustment_update(frames=range(f_start, N+1), fix_start=fix_start)
        previous_ba[0] = N
    # if not status:
    #     status = add_frame_incremental(N)
    #     print("add frame success:", status)

    # add back missing frames
    if status:
        for fi in range(N-1, max(N-10, 0), -1):
            if all_poses[fi] is not None:
                continue
            status = add_frame_incremental(fi)
            print("add missing frame success:", status)
            if status:
                f_start = max(min(min(previous_ba[0],fi)-BA_SW//2, N-BA_SW+1), 0)
                bundle_adjustment_update(frames=range(f_start, N+1), fix_start=True)
                previous_ba[0] = N
    else:
        return

    # loop closure
    if previous_lc[0] > N-LC_MUTE:
        return
    if success:
        similar_frames = find_similar_frames(bow)
    if len(similar_frames) == 0:
        return
    current_keypoints = set(all_keypoints[N])
    current_keypoints.discard(-1)
    found = False
    attempt_count = 0
    for fi, score in similar_frames:
        if abs(fi-N) < 20:
            continue
        # keypoints = set(all_keypoints[fi])
        # if len(keypoints.intersection(current_keypoints)) > 0:
        #     continue

        # print(N, fi, score)
        img_i, keypoints_i, descriptors_i = all_frames[fi]
        img_N, keypoints_N, descriptors_N = all_frames[N]

        matches = match_feature_pair(all_frames[fi], all_frames[N], -1)
        if len(matches) <= 30:
            continue
        points_i = np.array([keypoints_i[m.queryIdx].pt for m in matches])
        points_N = np.array([keypoints_N[m.trainIdx].pt for m in matches])
        inliner_mask = pose_estimation_2d2d(points_i, points_N, True)
        inliner_ratio = np.mean(inliner_mask.astype(np.float32))
        attempt_count += 1

        if inliner_ratio > 0.8:
            found = True
            print("Close loop:", "frames", (N, fi),
                  "score", score, "matches", len(matches), "inliners", inliner_ratio)
            inliners = np.where(inliner_mask.flatten())
            points_i, points_N = points_i[inliners], points_N[inliners]
            matches = [m for m, valid in zip(matches, inliner_mask) if valid]
            draw_matches(all_frames[fi], all_frames[N], matches)
            points_3d = triangulation(points_i, points_N, *all_poses[fi], *all_poses[N])
            break
        if attempt_count >= 1:
            break
    if not found:
        return
    for j, (p0, p, p3d, m) in enumerate(zip(
            points_i, points_N, points_3d, matches)):
        idx_i = all_keypoints[fi][m.queryIdx]
        idx_N = all_keypoints[N][m.trainIdx]
        if idx_i != -1 and idx_N != -1:
            # merge keypoints to idx_N
            for f, i in all_points_frames[idx_i]:
                all_keypoints[f][i] = idx_N
                if (f, idx_i) in all_observations:
                    all_observations[(f, idx_N)] = all_observations[(f, idx_i)]
                    del all_observations[(f, idx_i)]
            all_points_frames[idx_i] = []
            continue
        if idx_i == -1 and idx_N == -1:
            # create new point
            idx_i = idx_N = len(all_points)
            all_keypoints[fi][m.queryIdx] = idx_i
            all_keypoints[N][m.trainIdx] = idx_N
            all_points.append(p3d)
            all_points_frames.append([(fi, m.queryIdx), (N, m.trainIdx)])
            all_observations[(fi, idx_i)] = keypoints_i[m.queryIdx].pt
            all_observations[(N, idx_N)] = keypoints_N[m.trainIdx].pt
            continue
        if idx_i == -1:
            idx_i = idx_N
            all_keypoints[fi][m.queryIdx] = idx_i
            all_points_frames[idx_i].append((fi, m.queryIdx))
            all_observations[(fi, idx_i)] = keypoints_i[m.queryIdx].pt
            continue
        if idx_N == -1:
            idx_N = idx_i
            all_keypoints[N][m.trainIdx] = idx_N
            all_points_frames[idx_N].append((N, m.trainIdx))
            all_observations[(N, idx_N)] = keypoints_N[m.trainIdx].pt
            continue
    previous_lc[0] = N
    bundle_adjustment_update(fix_start=False)



# plotting and exporting

def cull_points(points, colors, stdev):
    points, colors = np.array(points), np.array(colors)
    mean = np.mean(points, axis=0)
    cov_matrix = np.cov(points, rowvar=False)
    mahalanobis_dist = np.sqrt(np.sum(np.dot((points-mean), np.linalg.inv(cov_matrix)) * (points-mean), axis=1))
    mask = mahalanobis_dist < stdev
    return points[mask], colors[mask]

def plot_points(ax, points, colors):
    if colors is None:
        ax.scatter(points[:,0], points[:,2], points[:,1], zorder=4)
        return
    ax.scatter(points[:,0], points[:,2], points[:,1], c=colors, marker='.', zorder=4)

def plot_camera(ax, R, t, sc=1.0):
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


def export_nerfstudio(path):
    import os
    os.makedirs(path, exist_ok=True)

    # images
    img_filenames = ["frame_{:05d}.jpg".format(i) for i in range(len(all_frames))]
    for sc in [1, 2, 4, 8]:
        imgdir = os.path.join(path, f"images_{sc}".rstrip("_1"))
        os.makedirs(imgdir, exist_ok=True)
        for filename, (img, keypoints, descriptors) in zip(img_filenames, all_frames):
            img = cv.resize(img, [img.shape[1]//sc, img.shape[0]//sc])
            filename = os.path.join(imgdir, filename)
            cv.imwrite(filename, img, [int(cv.IMWRITE_JPEG_QUALITY), 95])

    # point cloud
    pcl_path = os.path.join(path, "sparse_pc.ply")
    if False:
        from plyfile import PlyData, PlyElement
        vertex = np.array([
            (x, y, z, r, g, b) for (x, y, z), (r, g, b) in zip(points, colors*255.0)],
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('red', 'uint8'), ('green', 'uint8'), ('blue', 'uint8')])
        vertex_element = PlyElement.describe(vertex, 'vertex')
        PlyData([vertex_element], text=False).write(pcl_path)
    else:
        fp = open(pcl_path, 'w')
        fp.write("ply\n")
        fp.write("format ascii 1.0\n")
        fp.write(f"element vertex {len(points)}\n", )
        for a in 'xyz':
            fp.write(f"property float {a}\n")
        for c in ['red', 'green', 'blue']:
            fp.write(f"property uint8 {c}\n")
        fp.write("end_header\n")
        for i, (p, c) in enumerate(zip(points, colors)):
            c = np.clip(255.0*c, 0, 255).astype(np.uint8)
            fp.write("{:.6f} {:.6f} {:.6f} {} {} {}\n".format(*p, *c))
        fp.close()

    # transforms
    dist_params = np.zeros(4, dtype=np.float32)
    if len(camera_params) > 4:
        dist_params = camera_params[-4:]
    dist_params = dist_params.tolist()
    transforms = {
        'w': int(IMG_SHAPE[0]),
        'h': int(IMG_SHAPE[1]),
        'fl_x': K[0,0],
        'fl_y': K[1,1],
        'cx': K[0,2],
        'cy': K[1,2],
        'k1': dist_params[0],
        'k2': dist_params[1],
        'p1': dist_params[2],
        'p2': dist_params[3],
        'camera_model': 'OPENCV',
        'frames': [],
        'applied_transform': [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
        ],
        'ply_file_path': 'sparse_pc.ply'
    }
    for filename, pose in zip(img_filenames, all_poses):
        if pose is None:
            continue
        R, t = pose
        mat = np.eye(4)
        mat[:3,:3] = R
        mat[:3,3:] = t
        mat = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]) @ mat
        mat = np.linalg.inv(mat)
        transforms['frames'].append({
            'file_path': os.path.join('images', filename),
            'transform_matrix': mat.tolist()
        })
    import json
    with open(os.path.join(path, 'transforms.json'), 'w') as fp:
        json.dump(transforms, fp, indent=4)


if __name__ == "__main__":
    from time import perf_counter

    np.random.seed(42)
    cv.setRNGSeed(42)

    # read images + extract features
    imgs = [
        # cv.imread(f"img/pit_{i}.jpg", cv.IMREAD_COLOR)
        # for i in range(0, 11, 1)
        # cv.imread(f"img/arena_{i}.jpg", cv.IMREAD_COLOR)
        # for i in range(0, 20, 1)
        # for i in range(0, 40, 1)
        # cv.imread(f"img/float_{i}.jpg", cv.IMREAD_COLOR)
        # for i in range(0, 20, 1)
        # for i in range(0, 30, 1)
        cv.imread(f"img/temp_{i}.jpg", cv.IMREAD_COLOR)
        for i in range(0, 200, 1)
    ]
    IMG_SHAPE = np.array([imgs[0].shape[1], imgs[0].shape[0]])
    imgs = [cv.resize(img, IMG_SHAPE.astype(np.int32)) for img in imgs]

    vocab_path = "/home/harry7557558/GitHub/external/fbow/vocabularies/orb_mur.fbow"
    vocabulary.readFromFile(vocab_path)

    time_start = perf_counter()

    # reconstruction
    bundle_adjustment = bundle_adjustment_ceres_3
    # bundle_adjustment = bundle_adjustment_ceres
    for i, img in enumerate(imgs):
        print(f"adding new frame ({i}/{len(imgs)})")
        add_frame(img)
        print()
        if i >= max(4, BA_SW//2):
            bundle_adjustment = bundle_adjustment_ceres_3
    # bundle_adjustment = bundle_adjustment_ceres_8
    bundle_adjustment_update(fix_start=False)
    print()

    print(len(all_frames), "result frames")
    if camera_params is not None:
        print("camera params:", camera_params[:4], camera_params[4:])

    # rgb colors for points
    colors = np.zeros((len(all_points), 3))
    counts = np.zeros((len(colors), 1), dtype=np.int32)
    for (pose_i, point_i), uv in all_observations.items():
        x, y = map(int, uv)
        colors[point_i] += imgs[pose_i][y, x]
        counts[point_i] += 1
    nonzero_i = set(np.where(counts.flatten() > 1)[0])
    for i, p in enumerate(all_points):
        if p is None and i in nonzero_i:
            nonzero_i.remove(i)
    # print(counts.flatten().tolist())
    print("ba density:", np.mean(counts))
    nonzero_i = sorted(nonzero_i)
    colors = colors[nonzero_i] / (255.0*counts[nonzero_i])
    points = np.array([all_points[i] for i in nonzero_i])
    assert len(points) > 2 and "reconstruction failed"
    points, colors = cull_points(points, colors, 3.5)
    points, colors = cull_points(points, colors, 3.5)
    points, colors = cull_points(points, colors, 3.0)
    colors = np.flip(colors, axis=1)
    print(len(points), "final points")

    time_end = perf_counter()
    print("time elapsed: {:.1f} s".format(time_end-time_start))

    # export
    try:
        export_nerfstudio("/home/harry7557558/data/temp")
    except BaseException as e:
        print("Failed to export:", e)

    # plot
    plt.figure(figsize=(8, 8))
    ax = plt.axes(projection='3d')
    ax.computed_zorder = False
    sc = np.linalg.det(np.cov(points.T))**(1/6)
    print('sc:', sc)
    sc *= 0.2 * np.sqrt(K[0,0]*K[1,1]) / np.sqrt(np.prod(IMG_SHAPE)) * (10/len(all_poses))**0.5
    for pose in all_poses:
        if pose is not None:
            plot_camera(ax, *pose, sc)
        else:
            ax.scatter(np.nan, np.nan)
    # plot_points(ax, points, colors)
    plot_points(ax, points, colors)
    set_axes_equal(ax)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.invert_zaxis()
    plt.show()
