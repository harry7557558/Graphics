import cv2 as cv
import numpy as np
import scipy.sparse
import scipy.optimize

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# https://github.com/gaoxiang12/slambook2


IMG_SHAPE = np.array([960.0, 540.0])
K = np.array([[751.3, 0., 477.8],
              [0., 754.9, 286.8],
              [0., 0., 1.]])


def pixel2cam(p):
    return np.array([(p[0]-K[0,2])/K[0,0], (p[1]-K[1,2])/K[1,1]])


def KDT_NMS(kps, descs=None, r=15, k_max=20):
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
    print('Filtered',len(kp_filtered),'of',N)
    return kp_filtered, descs_filtered


def extract_features(imgs):
    detector = cv.ORB_create(10000)
    # detector = cv.SIFT_create(1000)

    features = []
    for img in imgs:
        feature = detector.detectAndCompute(img, None)
        feature = KDT_NMS(feature[0], feature[1])
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


def pose_estimation(keypoints_1, keypoints_2, matches):

    points1 = np.array([keypoints_1[match.queryIdx].pt for match in matches])
    points2 = np.array([keypoints_2[match.trainIdx].pt for match in matches])

    f = np.sqrt(K[0,0]*K[1,1])
    c = (K[0,2], K[1,2])
    essential_matrix, _ = cv.findEssentialMat(points1, points2, f, c)

    n, R, t, _ = cv.recoverPose(essential_matrix, points1, points2, cameraMatrix=K)

    return R, t

def triangulation(keypoints_1, keypoints_2, matches, R1, t1, R2, t2):

    pts_1 = np.array([pixel2cam(keypoints_1[match.queryIdx].pt) for match in matches])
    pts_2 = np.array([pixel2cam(keypoints_2[match.trainIdx].pt) for match in matches])

    T1 = np.hstack((R1, t1))
    T2 = np.hstack((R2, t2))

    pts_4d_homogeneous = cv.triangulatePoints(T1, T2, pts_1.T, pts_2.T)
    pts_4d = pts_4d_homogeneous / np.tile(pts_4d_homogeneous[-1, :], (4, 1))
    points_3d = pts_4d[:3, :].T

    return points_3d


def estimate_poses_points(features):

    # relative pose estimation between 0 and 1
    keypoints_0, descriptors_0 = features[0]
    keypoints_1, descriptors_1 = features[1]
    keypoints_2, descriptors_2 = features[2]
    matches_01 = match_feature_pair(features[0], features[1])
    matches_02 = match_feature_pair(features[0], features[2])
    matches_12 = match_feature_pair(features[1], features[2])
    R01, t01 = pose_estimation(keypoints_0, keypoints_1, matches_01)
    points_3d = triangulation(keypoints_0, keypoints_1, matches_01,
                              np.eye(3), np.zeros((3,1)), R01, t01)
    # draw_matches(imgs[0], keypoints_0, imgs[1], keypoints_1, matches_01)

    # find pose for 2
    points_map = {}
    for m, p in zip(matches_01, points_3d):
        points_map[(0, m.queryIdx)] = p
        points_map[(1, m.trainIdx)] = p
    pts_2d, pts_3d = [], []
    for m in matches_02:
        key = (0, m.queryIdx)
        if key in points_map:
            pts_2d.append(keypoints_2[m.trainIdx].pt)
            pts_3d.append(points_map[key])
    for m in matches_12:
        key = (1, m.queryIdx)
        if key in points_map:
            pts_2d.append(keypoints_2[m.trainIdx].pt)
            pts_3d.append(points_map[key])
    print(len(pts_2d), 'points for PnP')
    _, r, t = cv.solvePnP(np.array(pts_3d), np.array(pts_2d), K, None,
                          useExtrinsicGuess=True, rvec=cv.Rodrigues(R01)[0], tvec=1*t01,
                          flags=cv.SOLVEPNP_ITERATIVE)
    R, _ = cv.Rodrigues(r)
    poses = [
        (np.eye(3), np.zeros((3,1))),
        (R01, t01),
        (R, t)
    ]

    keypoints_3d = []  # list of list of 3d keypoints
    keypoints_map = {}  # (fi, pi): keypoints_3d index
    observations = []  # (pose index, point index, uv)
    for fi, (f1, f2) in enumerate([(0, 1), (0, 2), (1, 2)]):
        # if fi == 1 or fi == 2:
        #     continue

        # estimate 3d point locations
        keypoints_1, descriptors_1 = features[f1]
        keypoints_2, descriptors_2 = features[f2]
        matches = [matches_01, matches_02, matches_12][fi]
        R1, t1 = poses[f1]
        R2, t2 = poses[f2]
        points_3d = triangulation(keypoints_1, keypoints_2, matches, R1, t1, R2, t2)

        # add keypoints
        num_new_points = 0
        for match, point in zip(matches, points_3d):
            i1, i2 = match.queryIdx, match.trainIdx
            key1, key2 = (f1, i1), (f2, i2)
            if key1 in keypoints_map and key2 in keypoints_map:
                j = keypoints_map[key1]
                keypoints_3d[j].append(point)
            elif key1 in keypoints_map and key2 not in keypoints_map:
                j = keypoints_map[key1]
                keypoints_3d[j].append(point)
                keypoints_map[key2] = j
                observations.append((f2, j, keypoints_2[i2].pt))
            elif key1 not in keypoints_map and key2 in keypoints_map:
                j = keypoints_map[key2]
                keypoints_3d[j].append(point)
                keypoints_map[key1] = j
                observations.append((f1, j, keypoints_1[i1].pt))
            else:
                assert(key1 not in keypoints_map and key2 not in keypoints_map)
                j = len(keypoints_3d)
                keypoints_3d.append([point])
                num_new_points += 1
                keypoints_map[key1] = keypoints_map[key2] = j
                observations.append((f1, j, keypoints_1[i1].pt))
                observations.append((f2, j, keypoints_2[i2].pt))
        print(num_new_points, 'new points')

    points = [np.mean(p, axis=0) for p in keypoints_3d]
    # points = [p[0] for p in keypoints_3d]
    return poses, np.array(points), observations


def exp_so3t(T):
    phi = T[0:3]
    t = T[3:6]
    theta = np.linalg.norm(phi)
    R = np.eye(3)
    if theta != 0.0:
        n = phi / theta
        nnT = np.outer(n, n)
        n_star = np.array([[0.0, -n[2], n[1]], [n[2], 0.0, -n[0]], [-n[1], n[0], 0.0]])
        R = np.cos(theta) * R + \
            (1.0-np.cos(theta)) * nnT + \
            np.sin(theta) * n_star
    assert np.linalg.norm(R@R.T-np.eye(3)) < 1e-6
    return R, t

def log_so3t(R, t):
    assert np.linalg.norm(R@R.T-np.eye(3)) < 1e-6
    theta = np.arccos((np.trace(R)-1)/2)
    if theta != 0.0:
        eigvals, eigvecs = np.linalg.eig(R)
        n = eigvecs[:, np.argmin(np.abs(eigvals-1))].real
        nnT = np.outer(n, n)
        n_star = -(np.cos(theta) * np.eye(3) + \
            (1.0-np.cos(theta)) * nnT - \
            R) / np.sin(theta)
        n = np.array([n_star[2][1], n_star[0][2], n_star[1][0]])
    else:
        n = np.array([0.0, 0.0, 1.0])
    phi = theta * n
    return np.concatenate((phi, t.flatten()))

def bundle_adjustment_01(poses, points, observations):
    """https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html"""

    # params
    n_pose = len(poses)
    n_point = len(points)
    n_obs = len(observations)
    poses = np.array([log_so3t(*pose) for pose in poses])
    points_2d = np.array([uv for (pose_i, point_i, uv) in observations])
    # points_2d = (points_2d-K[0:2,2]) @ np.linalg.inv(K[:2,:2])
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
        # points_proj = (points_proj-K[0:2,2]) @ np.linalg.inv(K[:2,:2])
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

    # outlier rejection
    # residuals = np.linalg.norm(residuals.reshape((2, -1)), axis=0)
    # non_outliers = np.where(residuals < 10)

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
    
    params = res.x
    so3ts = params[:6*n_pose].reshape((n_pose, 6))
    poses = [exp_so3t(so3t) for so3t in so3ts]
    points_3d = params[6*n_pose:].reshape((n_point, 3))
    return poses, points_3d


def plot_points(ax, points_3d, max_norm, colors):
    points_3d_norms = np.linalg.norm(points_3d, axis=1)
    i = np.where(points_3d_norms<max_norm)
    filtered = points_3d[i]
    if colors is None:
        ax.scatter(filtered[:,0], filtered[:,2], filtered[:,1])
        return
    colors = colors[i]
    ax.scatter(filtered[:,0], filtered[:,2], filtered[:,1], c=colors)

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

    # read images + extract features
    imgs = [
        cv.imread(f"img/{i}.jpg", cv.IMREAD_COLOR)
        for i in [0, 1, 2]
        # for i in [0, 2, 4]
        # for i in [1, 2, 3]
        # for i in [0, 2, 5]
        # for i in [0, 3, 6]
        # for i in [7, 8, 9]
        # for i in [6, 8, 10]
    ]
    features = extract_features(imgs)

    # pose estimation
    poses, points, observations = estimate_poses_points(features)
    poses, points = bundle_adjustment_01(poses, points, observations)

    # rgb colors for points
    colors = np.zeros_like(points)
    counts = np.zeros((len(colors), 1))
    for pose_i, point_i, uv in observations:
        x, y = map(int, uv)
        colors[point_i] += imgs[pose_i][y, x]
        counts[point_i] += 1
    colors = colors / (255.0*counts)

    # plot
    plt.figure(figsize=(8, 8))
    ax = plt.axes(projection='3d')
    sc = np.linalg.det(np.cov(points.T))**(1/6)
    print('sc:', sc)
    for R, t in poses:
        plot_camera(ax, R, t, 0.5*sc)
    plot_points(ax, points, 1000.0, colors)
    set_axes_equal(ax)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.invert_zaxis()
    plt.show()
