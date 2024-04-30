import cv2 as cv
import numpy as np
import scipy.sparse
import scipy.optimize

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# https://github.com/gaoxiang12/slambook2


IMG_SHAPE = np.array([960.0, 540.0])

# initial guess
F = 800.
K = np.array([[F, 0., IMG_SHAPE[0]/2],
              [0., F, IMG_SHAPE[1]/2],
              [0., 0., 1.]])



def filter_features(kps, descs=None, r=15, k_max=20):
    """ https://stackoverflow.com/a/57390077
    Use kd-tree to perform local non-maximum suppression of key-points
    kps - key points obtained by one of openCVs 2d features detectors (SIFT, SURF, AKAZE etc..)
    r - the radius of points to query for removal
    k_max - maximum points retreived in single query
    """
    from scipy.spatial import KDTree

    # sort by score to keep highest score features in each locality
    neg_responses = [-kp.response for kp in kps]
    order = np.argsort(neg_responses)
    kps = np.array(kps)[order].tolist()

    # create kd-tree for quick NN queries
    data = np.array([list(kp.pt) for kp in kps])
    kd_tree = KDTree(data)

    # perform NMS using kd-tree, by querying points by score order, 
    # and removing neighbors from future queries
    N = len(kps)
    removed = set()
    for i in range(N):
        if i in removed:
            continue
        dist, inds = kd_tree.query(data[i,:],k=k_max,distance_upper_bound=r)
        for j in inds:
            if j>i:
                removed.add(j)

    kp_filtered = [kp for i,kp in enumerate(kps) if i not in removed]
    descs_filtered = None
    if descs is not None:
        descs = descs[order]
        descs_filtered = np.array([desc for i,desc in enumerate(descs) if i not in removed])
    print('filtered', len(kp_filtered), 'of', N, 'features')
    return kp_filtered, descs_filtered

def extract_features(imgs):
    detector = cv.ORB_create(6000)
    # detector = cv.SIFT_create(10000)

    features = []
    for img in imgs:
        feature = detector.detectAndCompute(img, None)
        feature = filter_features(feature[0], feature[1])
        features.append(feature)

    return features

def match_feature_pair(feature1, feature2):
    matcher = cv.BFMatcher(cv.NORM_HAMMING)
    # matcher = cv.BFMatcher(cv.NORM_L2)

    keypoints_1, descriptors_1 = feature1
    keypoints_2, descriptors_2 = feature2

    matches = matcher.match(descriptors_1, descriptors_2)

    good_matches = []
    min_dist = min([match.distance for match in matches])
    for match in matches:
        if match.distance <= max(2 * min_dist, 30):
            good_matches.append(match)
    
    return good_matches

def draw_matches(img_1, keypoints_1, img_2, keypoints_2, matches):
    img_matches = np.empty((img_1.shape[0], img_1.shape[1], 3), dtype=np.uint8)
    cv.drawKeypoints(img_1, keypoints_1, img_matches)
    for match in matches:
        kp1 = keypoints_1[match.queryIdx].pt
        kp2 = keypoints_2[match.trainIdx].pt
        pt1 = (int(kp1[0]), int(kp1[1]))
        pt2 = (int(kp2[0]), int(kp2[1]))
        cv.line(img_matches, pt1, pt2, (0, 255, 0), 1)
    cv.imshow("Matches", img_matches)
    cv.waitKey(0)
    cv.destroyAllWindows()



def pixel2cam(p):
    return np.array([(p[0]-K[0,2])/K[0,0], (p[1]-K[1,2])/K[1,1]])

def pose_estimation(points1, points2):
    points1 = np.array(points1)
    points2 = np.array(points2)
    f = np.sqrt(K[0,0]*K[1,1])
    c = (K[0,2], K[1,2])
    essential_matrix, _ = cv.findEssentialMat(points1, points2, f, c)
    n, R, t, _ = cv.recoverPose(essential_matrix, points1, points2, cameraMatrix=K)
    return R, t

def triangulation(points1, points2, R1, t1, R2, t2):
    points1 = np.array([pixel2cam(p) for p in points1])
    points2 = np.array([pixel2cam(p) for p in points2])
    T1 = np.hstack((R1, t1))
    T2 = np.hstack((R2, t2))
    pts_4d_homogeneous = cv.triangulatePoints(T1, T2, points1.T, points2.T)
    pts_4d = pts_4d_homogeneous / np.tile(pts_4d_homogeneous[-1, :], (4, 1))
    points_3d = pts_4d[:3, :].T
    return points_3d



def exp_so3t(T):
    phi = T[0:3]
    t = T[3:6]
    R, _ = cv.Rodrigues(phi)
    return R, t

def log_so3t(R, t):
    assert np.linalg.norm(R@R.T-np.eye(3)) < 1e-6
    phi, _ = cv.Rodrigues(R)
    return np.concatenate((phi.flatten(), t.flatten()))

OUTLIER_Z = 8

def bundle_adjustment_01(camera_params, poses, points, observations):
    """points and poses only"""

    # params
    n_pose = len(poses)
    n_point = len(points)
    n_obs = len(observations)
    poses = np.array([log_so3t(*pose) for pose in poses])
    points_2d = np.array([uv for (pose_i, point_i, uv) in observations])
    params_init = np.concatenate((poses.flatten(), points.flatten()))
    poses_i = np.array([o[0] for o in observations])
    points_i = np.array([o[1] for o in observations])

    # function
    def fun(params):
        so3ts = params[:6*n_pose].reshape((n_pose, 6))
        poses = [exp_so3t(so3t) for so3t in so3ts]
        R = np.array([p[0] for p in poses])
        t = np.array([p[1] for p in poses])
        points_3d = params[6*n_pose:].reshape((n_point, 3))
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
    print('rmse before:', np.mean(residuals**2)**0.5)
    # plt.plot(abs(fun(params_init)), '.')
    # plt.yscale('log')
    # plt.show()
    # __import__('sys').exit(0)

    # jacobian sparsity
    sp = scipy.sparse.lil_matrix(
        (2*n_obs, 6*n_pose+3*n_point), dtype=int)
    for i, (pose_i, point_i, uv) in enumerate(observations):
        p0 = 6*pose_i
        sp[2*i:2*i+2, p0:p0+6] = 1  # to pose
        p0 = 6*n_pose+3*point_i
        sp[2*i:2*i+2, p0:p0+3] = 1  # to point

    # optimization
    res = scipy.optimize.least_squares(
        fun, params_init, jac_sparsity=sp,
        verbose=0, x_scale='jac', ftol=1e-4, method='trf')
    print('(nfev, njev):', res.nfev, res.njev)
    print('rmse after:', np.mean(fun(res.x)**2)**0.5)
    # plt.plot(abs(fun(res.x)), '.')
    # plt.yscale('log')
    # plt.show()
    # __import__('sys').exit(0)

    residual = np.abs(res.fun.reshape((-1, 2)))
    mask = residual > OUTLIER_Z*np.mean(residual)
    outliers = np.where(mask[:,0] | mask[:,1])

    params = res.x
    so3ts = params[:6*n_pose].reshape((n_pose, 6))
    poses = [exp_so3t(so3t) for so3t in so3ts]
    points_3d = params[6*n_pose:].reshape((n_point, 3))
    return camera_params, poses, points_3d, outliers

def bundle_adjustment_02(camera_params, poses, points, observations):
    """points + poses + camera intrinsics (f,cx,cy)"""

    # params
    if camera_params is None:
        f0 = np.sqrt(K[0,0]*K[1,1])
        cx0, cy0 = K[:2,2]
        camera_params = [f0, cx0, cy0]
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
    print('rmse before:', np.mean(residuals**2)**0.5)
    # plt.plot(abs(fun(params_init)), '.')
    # plt.yscale('log')
    # plt.show()
    # __import__('sys').exit(0)

    # jacobian sparsity
    sp = scipy.sparse.lil_matrix(
        (2*n_obs, 3+6*n_pose+3*n_point), dtype=int)
    for i, (pose_i, point_i, uv) in enumerate(observations):
        p0 = 3+6*pose_i
        sp[2*i:2*i+2, p0:p0+6] = 1  # to pose
        p0 = 3+6*n_pose+3*point_i
        sp[2*i:2*i+2, p0:p0+3] = 1  # to point
    sp[:, :3] = 1

    # optimization
    res = scipy.optimize.least_squares(
        fun, params_init, jac_sparsity=sp,
        verbose=0, x_scale='jac', ftol=1e-4, method='trf')
    print('(nfev, njev):', res.nfev, res.njev)
    print('rmse after:', np.mean(fun(res.x)**2)**0.5)
    # plt.plot(abs(res.fun), '.')
    # plt.hist(abs(res.fun), bins=20)
    # plt.yscale('log')
    # plt.show()
    # __import__('sys').exit(0)

    # residual = np.linalg.norm(res.fun.reshape((-1,2)), axis=1)
    # outliers = np.where(residual > OUTLIER_Z*np.mean(residual))
    residual = np.abs(res.fun.reshape((-1, 2)))
    mask = residual > OUTLIER_Z*np.mean(residual)
    outliers = np.where(mask[:,0] | mask[:,1])

    params = res.x
    camera_params = params[:3]
    so3ts = params[3:3+6*n_pose].reshape((n_pose, 6))
    poses = [exp_so3t(so3t) for so3t in so3ts]
    points_3d = params[3+6*n_pose:].reshape((n_point, 3))
    return camera_params, poses, points_3d, outliers

def bundle_adjustment_03(camera_params, poses, points, observations):
    """points + poses + camera intrinsics (f,cx,cy,*dist_coeffs)"""

    # params
    n_int = 7  # number of camera intrinsics, 3+len(dist_coeffs)
    if camera_params is None:
        f0 = np.sqrt(K[0,0]*K[1,1])
        cx0, cy0 = K[:2,2]
        camera_params = [f0, cx0, cy0] + [0]*(n_int-3)
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
        dist_coeffs = params[3:n_int]
        K = np.array([[f, 0., cx], [0., f, cy], [0., 0., 1.]])
        so3ts = params[n_int:n_int+6*n_pose].reshape((n_pose, 6))
        poses = [exp_so3t(so3t) for so3t in so3ts]
        R = np.array([p[0] for p in poses])
        t = np.array([p[1] for p in poses])
        points_3d = params[n_int+6*n_pose:].reshape((n_point, 3))
        points_r = np.einsum('kij,kj->ki', R[poses_i], points_3d[points_i]) + t[poses_i]
        points_c = points_r @ K.T
        points_proj = points_c[:,0:2] / points_c[:,2:3]
        points_proj = (points_proj-K[0:2,2]) @ np.linalg.inv(K[:2,:2])
        points_proj = cv.undistortPoints(points_proj, np.eye(3), dist_coeffs)[:,0,:]
        points_proj = points_proj @ K[:2,:2] + K[0:2,2]
        residual = (points_proj-points_2d).flatten()
        # return residual
        # return np.arcsinh(residual)
        delta = 2
        qr_residual = np.sign(residual) * np.sqrt(delta*np.fmax(2.0*np.abs(residual)-delta, 0.0))
        return residual + (qr_residual-residual) * (np.abs(residual) > delta)

    residuals = fun(params_init)
    print('rmse before:', np.mean(residuals**2)**0.5)
    # plt.plot(abs(fun(params_init)), '.')
    # plt.yscale('log')
    # plt.show()
    # __import__('sys').exit(0)

    # jacobian sparsity
    sp = scipy.sparse.lil_matrix(
        (2*n_obs, n_int+6*n_pose+3*n_point), dtype=int)
    for i, (pose_i, point_i, uv) in enumerate(observations):
        p0 = n_int+6*pose_i
        sp[2*i:2*i+2, p0:p0+6] = 1  # to pose
        p0 = n_int+6*n_pose+3*point_i
        sp[2*i:2*i+2, p0:p0+3] = 1  # to point
    sp[:, :n_int] = 1

    # optimization
    res = scipy.optimize.least_squares(
        fun, params_init, jac_sparsity=sp,
        verbose=0, x_scale='jac', ftol=1e-4, method='trf')
    print('(nfev, njev):', res.nfev, res.njev)
    print('rmse after:', np.mean(fun(res.x)**2)**0.5)
    # plt.plot(abs(fun(res.x)), '.')
    # plt.yscale('log')
    # plt.show()
    # __import__('sys').exit(0)

    residual = np.abs(res.fun.reshape((-1, 2)))
    mask = residual > OUTLIER_Z*np.mean(residual)
    outliers = np.where(mask[:,0] | mask[:,1])

    params = res.x
    camera_params = params[:n_int]
    so3ts = params[n_int:n_int+6*n_pose].reshape((n_pose, 6))
    poses = [exp_so3t(so3t) for so3t in so3ts]
    points_3d = params[n_int+6*n_pose:].reshape((n_point, 3))
    return camera_params, poses, points_3d, outliers



# global map
camera_params = None  # depends on BA function
all_frames = []  # (keypoints, descriptor)
all_poses = []  # (R, t) | None
all_matches = {}  # (f1, f2): matches
all_keypoints = {}  # (f, i) : i_3d
all_points = []  # 3d points
all_observations = []  # (fi, pi, uv)



def bundle_adjustment_update(frames=None):
    global camera_params
    global all_observations

    if frames is None:
        frames = [i for i in range(len(all_poses)) if all_poses[i] is not None]

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
    for i, (fi, pi, uv) in enumerate(all_observations):
        if fi not in poses_invmap:
            continue
        if pi not in points_invmap:
            points_invmap[pi] = len(points)
            points.append(all_points[pi])
            points_map.append(pi)
        observations.append((poses_invmap[fi], points_invmap[pi], uv))
        observations_map.append(i)

    # run bundle adjustment
    camera_params, poses_updated, points_updated, outliers = bundle_adjustment(
        camera_params, poses, np.array(points), observations)
    if len(outliers[0]) > 0:
        print(f"{len(outliers[0])}/{len(observations)} outliers")

    # to-do: relative pose matching using SVD

    # put back
    for i, (R, t) in zip(poses_map, poses_updated):
        all_poses[i] = (R, t.reshape((3,1)))
    for i, point in zip(points_map, points_updated):
        all_points[i] = point
    outliers = set(*outliers)
    all_observations = [all_observations[i] for i in range(len(all_observations)) if i not in outliers]


def add_frame_init(features):
    global camera_params
    global all_points

    N = len(all_poses)
    keypoints, descriptors = features

    # try matching until success
    success_pair = None
    for i, features_0 in enumerate(all_frames):
        if i >= N:
            break
        keypoints_0, descriptors_0 = features_0
        matches = match_feature_pair(features_0, features)
        all_matches[(i, N)] = matches
        if len(matches) < 9:
            continue

        points_0 = [keypoints_0[m.queryIdx].pt for m in matches]
        points = [keypoints[m.trainIdx].pt for m in matches]
        R, t = pose_estimation(points_0, points)
        points_3d = triangulation(points_0, points, np.eye(3), np.zeros((3,1)), R, t)
        if np.sum(points_3d[:,2] <= 0) < len(points_3d) / 50:
            success_pair = (i, N)
            break
    if success_pair is None:
        all_poses.append(None)
        return False

    # add data to global map
    all_poses[i] = (np.eye(3), np.zeros((3,1)))
    all_poses.append((R, t))
    for j, (p0, p, p3d) in enumerate(zip(points_0, points, points_3d)):
        if not p3d[2] > 0:
            continue
        pi = len(all_points)
        all_points.append(p3d)
        all_observations.append((i, pi, p0))
        all_observations.append((N, pi, p))
        all_keypoints[(i, matches[j].queryIdx)] = pi
        all_keypoints[(N, matches[j].trainIdx)] = pi

    return True

def add_frame_incremental(features):

    N = len(all_poses)
    keypoints, descriptors = features

    success_pair = None
    pts_3d, pts_2d = [], []
    for i in range(len(all_poses)-1, -1, -1):
        if all_poses[i] is None:
            continue
        features_0 = all_frames[i]
        keypoints_0 = features_0[0]
        matches = match_feature_pair(features_0, features)
        all_matches[(i, N)] = matches

        # PnP points
        for m in matches:
            key0 = (i, m.queryIdx)
            if key0 not in all_keypoints:
                continue
            pi = all_keypoints[key0]
            all_keypoints[(N,m.trainIdx)] = pi
            pts_3d.append(all_points[pi])
            pts_2d.append(keypoints[m.trainIdx].pt)
        if len(pts_3d) < 4:
            continue
        print(len(pts_3d), 'points for PnP')

        # solve PnP
        for di in range(1,N):
            if i-di >= 0 and all_poses[i-di] is not None:
                R0, t0 = all_poses[i-di]
                break
            elif i+di < N and all_poses[i+di] is not None:
                R0, t0 = all_poses[i+di]
                break
        _, r, t = cv.solvePnP(np.array(pts_3d), np.array(pts_2d), K, None,
                            useExtrinsicGuess=True, rvec=cv.Rodrigues(R0)[0], tvec=1.0*t0,
                            flags=cv.SOLVEPNP_ITERATIVE)
        R, _ = cv.Rodrigues(r)
        points_0 = [keypoints_0[m.queryIdx].pt for m in matches]
        points = [keypoints[m.trainIdx].pt for m in matches]
        points_3d = triangulation(points_0, points, *all_poses[i], R, t)
        points_3d_r = points_3d @ R.T + t.T
        num_invalid, num_point = np.sum(points_3d_r[:,2] <= 0), len(points_3d_r)
        if num_invalid < num_point / 50:
            success_pair = (i, N)
            break
        else:
            print(f"PnP failed ({num_invalid}/{num_point} invalid)")

    if success_pair is None:
        all_poses.append(None)
        return False

    # add data to global map
    all_poses.append((cv.Rodrigues(r)[0], t))
    for j, (p0, p, p3d) in enumerate(zip(points_0, points, points_3d)):
        if not p3d[2] > 0:
            continue
        key0 = (i, matches[j].queryIdx)
        if key0 in all_keypoints:
            pi = all_keypoints[key0]
        else:
            pi = len(all_points)
            all_points.append(p3d)
            all_keypoints[key0] = pi
            all_observations.append((i, pi, p0))
        all_keypoints[(N, matches[j].trainIdx)] = pi
        all_observations.append((N, pi, p))
    # to-do: add missed points from previously-matched frames to shared points
    #  - create an indice map between all_keypoints and all_points?

    return True

def add_frame(features):
    all_frames.append(features)
    if len(all_frames) < 2:
        all_poses.append(None)
        return
    
    if all_poses.count(None) == len(all_poses):
        status = add_frame_init(features)
        print("initialization success:", status)
        return
    
    status = add_frame_incremental(features)
    print("add frame success:", status)
    if status:
        print("running ba")
        bundle_adjustment_update()



def plot_points(ax, points_3d, colors, cull=None):
    if cull is not None:
        points_array = np.array(points_3d)
        mean = np.mean(points_array, axis=0)
        cov_matrix = np.cov(points_array, rowvar=False)
        mahalanobis_dist = np.sqrt(np.sum(np.dot((points_array-mean), np.linalg.inv(cov_matrix)) * (points_array-mean), axis=1))
        mask = mahalanobis_dist < cull
        points = points_array[mask]
        colors = colors[mask]
    else:
        points = np.array(points_3d)
        colors = np.array(colors)
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



if __name__ == "__main__":
    from time import perf_counter

    # read images + extract features
    imgs = [
        # cv.imread(f"img/pit_{i}.jpg", cv.IMREAD_COLOR)
        # for i in range(0, 11, 1)
        cv.imread(f"img/arena_{i}.jpg", cv.IMREAD_COLOR)
        for i in range(0, 14, 1)
        # for i in range(0, 20, 1)
        # for i in range(30, 40, 1)
        # cv.imread(f"img/float_{i}.jpg", cv.IMREAD_COLOR)
        # for i in range(0, 12, 1)
        # for i in range(10, 20, 1)
    ]
    time0 = perf_counter()
    features = extract_features(imgs)
    time1 = perf_counter()
    print("features extracted in {:.1f} ms".format(1000*(time1-time0)))
    print()

    # reconstruction
    bundle_adjustment = bundle_adjustment_02
    for i, feature in enumerate(features):
        print(f"adding new frame ({i}/{len(features)})")
        add_frame(feature)
        print()
        if i >= 10:
            bundle_adjustment = bundle_adjustment_02

    print("camera params:", camera_params)

    points = np.array(all_points)

    # rgb colors for points
    colors = np.zeros_like(points)
    counts = np.zeros((len(colors), 1))
    for pose_i, point_i, uv in all_observations:
        x, y = map(int, uv)
        colors[point_i] += imgs[pose_i][y, x]
        counts[point_i] += 1
    colors = colors / (255.0*counts)

    # plot
    plt.figure(figsize=(8, 8))
    ax = plt.axes(projection='3d')
    ax.computed_zorder = False
    sc = np.linalg.det(np.cov(points.T))**(1/6)
    print('sc:', sc)
    sc *= 0.2 * np.sqrt(K[0,0]*K[1,1]) / np.sqrt(np.prod(IMG_SHAPE))
    for pose in all_poses:
        if pose is not None:
            plot_camera(ax, *pose, sc)
        else:
            ax.scatter(np.nan, np.nan)
    # plot_points(ax, points, colors)
    plot_points(ax, points, colors, 2.5)
    set_axes_equal(ax)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.invert_zaxis()
    plt.show()
