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


def extract_features(imgs):
    detector = cv.ORB_create(1000)

    features = []
    for img in imgs:
        feature = detector.detectAndCompute(img, None)
        features.append(feature)

    return features

def match_feature_pair(feature1, feature2):
    matcher = cv.BFMatcher(cv.NORM_HAMMING)

    keypoints_1, descriptors_1 = feature1
    keypoints_2, descriptors_2 = feature2

    matches = matcher.match(descriptors_1, descriptors_2)

    good_matches = []
    min_dist = min([match.distance for match in matches])
    for match in matches:
        if match.distance <= max(2 * min_dist, 30):
            good_matches.append(match)
    
    return good_matches


def pose_estimation(keypoints_1, keypoints_2, matches):

    points1 = np.array([keypoints_1[match.queryIdx].pt for match in matches])
    points2 = np.array([keypoints_2[match.trainIdx].pt for match in matches])

    f = np.sqrt(K[0,0]*K[1,1])
    c = (K[0,2], K[1,2])
    essential_matrix, _ = cv.findEssentialMat(points1, points2, f, c)

    n, R, t, _ = cv.recoverPose(essential_matrix, points1, points2, cameraMatrix=K)

    return R, t

def triangulation(keypoints_1, keypoints_2, matches, R, t):

    pts_1 = np.array([pixel2cam(keypoints_1[match.queryIdx].pt) for match in matches])
    pts_2 = np.array([pixel2cam(keypoints_2[match.trainIdx].pt) for match in matches])

    T1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]], dtype=np.float32)
    T2 = np.hstack((R, t))

    pts_4d_homogeneous = cv.triangulatePoints(T1, T2, pts_1.T, pts_2.T)
    pts_4d = pts_4d_homogeneous / np.tile(pts_4d_homogeneous[-1, :], (4, 1))
    points_3d = pts_4d[:3, :].T

    return points_3d


def estimate_poses_points(features):
    poses = [
        (np.eye(3), np.zeros((3, 1)))
    ]
    res_points = []
    keypoints_map = {}  # index of current frame: keypoints index
    observations = []  # (pose index, point index, uv)
    for i in range(1, len(features)):
        # feature matching
        keypoints_1, descriptors_1 = features[i-1]
        keypoints_2, descriptors_2 = features[i]
        matches = match_feature_pair(features[i-1], features[i])
        
        # estimate pose
        R, t = pose_estimation(keypoints_1, keypoints_2, matches)
        R0, t0 = poses[-1]
        R1, t1 = R @ R0, R @ t0 + t
        poses.append((R1, t1))

        # estimate location
        points_3d = triangulation(keypoints_1, keypoints_2, matches, R, t)
        # points_3d = points_3d @ R0.T + t0.T
        points_3d = (points_3d - t0.T) @ np.linalg.inv(R0).T
        new_keypoints_map = {}
        num_new_points = 0
        for match, point in zip(matches, points_3d):
            i1, i2 = match.queryIdx, match.trainIdx
            if i1 in keypoints_map:
                j = keypoints_map[i1]
                res_points[j].append(point)
            else:
                j = len(res_points)
                res_points.append([point])
                num_new_points += 1
            new_keypoints_map[i2] = j
            # add observations
            if i == 1:
                observations.append((i-1, j, keypoints_1[i1].pt))
            observations.append((i, j, keypoints_2[i2].pt))
        keypoints_map = new_keypoints_map
        print(num_new_points, 'new points')

    points = [np.mean(p, axis=0) for p in res_points]
    # points = [p[0] for p in res_points]

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
        return (points_proj-points_2d).flatten()

    residuals = fun(params_init)
    print('rmse before:', np.mean(residuals**2)**0.5)
    # plt.plot(fun(params_init), '.')
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
    # plt.plot(fun(res.x), '.')
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
    points_3d = R @ np.linalg.inv(K) @ points + t.reshape((3, 1))
    idx = [0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1]
    vertices = points_3d[:,idx]
    ax.plot(vertices[0], vertices[2], vertices[1], 'k-')

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
    imgs = [
        cv.imread(f"img/{i}.jpg", cv.IMREAD_COLOR)
        # for i in range(0, 4+1, 4)
        for i in [0, 4, 6]
    ]
    features = extract_features(imgs)

    poses, points, observations = estimate_poses_points(features)
    poses, points = bundle_adjustment_01(poses, points, observations)

    plt.figure(figsize=(8, 8))
    ax = plt.axes(projection='3d')
    sc = np.linalg.det(np.cov(points.T))**(1/6)
    print('sc:', sc)
    sc = 1.0
    for R, t in poses:
        plot_camera(ax, R, t, sc)
    plot_points(ax, points, 1000.0, None)
    set_axes_equal(ax)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.invert_zaxis()
    plt.show()
