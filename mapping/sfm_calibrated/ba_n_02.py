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
    detector = cv.ORB_create(10000)
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



def match_features_pairwise(features):

    # match features
    n = len(features)
    matches_pairwise = {}
    for j in range(n):
        for i in range(j):
            matches = match_feature_pair(features[i], features[j])
            matches_pairwise[(i, j)] = matches

    # create observations
    num_points = 0
    keypoints_map = {}  # (fi, pi): keypoints_3d index
    observations = []  # (pose index, point index, uv)
    for f2 in range(n):
        for f1 in range(f2):
            keypoints_1, descriptors_1 = features[f1]
            keypoints_2, descriptors_2 = features[f2]
            matches = matches_pairwise[(f1, f2)]
            num_new_points = 0
            for match in matches:
                i1, i2 = match.queryIdx, match.trainIdx
                key1, key2 = (f1, i1), (f2, i2)
                if key1 not in keypoints_map and key2 not in keypoints_map:
                    j = num_points
                    num_points += 1
                    num_new_points += 1
                    keypoints_map[key1] = keypoints_map[key2] = j
                    observations.append((f1, j, keypoints_1[i1].pt))
                    observations.append((f2, j, keypoints_2[i2].pt))
                elif key1 in keypoints_map and key2 not in keypoints_map:
                    j = keypoints_map[key1]
                    keypoints_map[key2] = j
                    observations.append((f2, j, keypoints_2[i2].pt))
                elif key1 not in keypoints_map and key2 in keypoints_map:
                    j = keypoints_map[key2]
                    keypoints_map[key1] = j
                    observations.append((f1, j, keypoints_1[i1].pt))
            print(f"frame {f1}-{f2}:", num_new_points, 'new points')

    return observations


def estimate_poses_points(observations):

    n_pose = max([fi for (fi, pi, uv) in observations]) + 1
    n_point = max([pi for (fi, pi, uv) in observations]) + 1
    points_3d = [(np.zeros(3), np.inf) for _ in range(n_point)]  # (p, cov)
    poses = [None for _ in range(n_pose)]

    # restore matches
    points_3d_map = [[] for _ in range(n_point)]
    for fi, pi, uv in observations:
        points_3d_map[pi].append((fi, uv))
    matches_pairwise = {}  # (f1,f2): list[(pi,uv1,uv2)]
    for j in range(n_pose):
        for i in range(j):
            matches_pairwise[(i, j)] = []
    for pi, obs in enumerate(points_3d_map):
        obs.sort(key=lambda _: _[0])
        for i2 in range(len(obs)):
            for i1 in range(i2):
                f1, uv1 = obs[i1]
                f2, uv2 = obs[i2]
                if f1 == f2:
                    continue
                matches_pairwise[(f1, f2)].append((pi, uv1, uv2))

    # find two good (consecutive) frames to start reconstruction
    order_0 = list(range(len(features)))
    best = {
        'idx': (-1, -1),
        'cost': np.inf,
        'pose': (np.eye(3), np.zeros((3,1))),
        'points': None
    }
    pairwise_check = []
    for s in range(1, min(3,n_pose)):
        for i0, i1 in zip(order_0[:-s], order_0[s:]):
            pairwise_check.append((i0, i1))
    for i0, i1 in pairwise_check:
        matches = matches_pairwise[(i0, i1)]
        points_0 = [uv1 for (pi, uv1, uv2) in matches]
        points_1 = [uv2 for (pi, uv1, uv2) in matches]
        R1, t1 = pose_estimation(points_0, points_1)
        triangulated_points = triangulation(points_0, points_1,
                                            np.eye(3), np.zeros((3,1)), R1, t1)
        cost = np.sum(triangulated_points[:,2] <= 0)
        eigvals = np.linalg.eigvalsh(np.cov(triangulated_points.T))
        cond = (max(eigvals)/min(eigvals))**(1/2)
        cost += 1.0/(1.0+10.0/np.log(cond))
        if cost < best['cost']:
            best = {
                'idx': (i0, i1),
                'cost': cost,
                'pose': (R1, t1),
                'points': triangulated_points
            }
    i0, i1 = best['idx']
    matches = matches_pairwise[(i0, i1)]
    R1, t1 = best['pose']
    poses[i0] = (np.eye(3), np.zeros((3,1)))
    poses[i1] = (R1, t1)
    triangulated_points = best['points']
    # print(np.linalg.eigvalsh(np.cov(triangulated_points.T))**0.5)
    cov = np.linalg.det(np.cov(triangulated_points.T))**(1/6)
    for p3, (pi, uv1, uv2) in zip(triangulated_points, matches):
        points_3d[pi] = (p3, cov)
    print(len(triangulated_points), 'points in initial map')
    # recreate order, BFS
    order = []
    edge1 = []
    def add_i(i):
        nonlocal edge1
        if i >= 0 and i < n_pose and i not in order:
            order.append(i)
            edge1.append(i)
    add_i(i0)
    add_i(i1)
    edge = edge1
    while len(edge) > 0:
        edge1 = []
        for e in edge:
            add_i(e-1)
            add_i(e+1)
        edge = edge1
    print('incremental order:', ' '.join(map(str, order)))

    # PnP for the rest
    for fi in order[2:]:
        # generate PnP points
        pts_3d, pts_2d = [], []
        for i in order[:order.index(fi)]:
            matches = matches_pairwise[(min(i,fi), max(i,fi))]
            for pi, uv1, uv2 in matches:
                if points_3d[pi][1] == np.inf:
                    continue
                pts_3d.append(points_3d[pi][0])
                pts_2d.append(uv2 if i<fi else uv1)
        print(len(pts_2d), 'points for PnP')
        # solve PnP
        for i in range(1,n_pose):
            if fi-i >= 0 and poses[fi-i] is not None:
                R0, t0 = poses[fi-i]
                break
            elif fi+i < n_pose and poses[fi+i] is not None:
                R0, t0 = poses[fi+i]
                break
        _, r, t = cv.solvePnP(np.array(pts_3d), np.array(pts_2d), K, None,
                            useExtrinsicGuess=True, rvec=cv.Rodrigues(R0)[0], tvec=1.0*t0,
                            flags=cv.SOLVEPNP_ITERATIVE)
        poses[fi] = (cv.Rodrigues(r)[0], t)
        # add points
        for i in order[:order.index(fi)]:
            matches = matches_pairwise[(min(i,fi), max(i,fi))]
            points_0 = [uv1 for (pi, uv1, uv2) in matches]
            points_1 = [uv2 for (pi, uv1, uv2) in matches]
            if i > fi:
                points_0, points_1 = points_1, points_0
            triangulated_points = triangulation(points_0, points_1, *poses[i], *poses[fi])
            # print(np.linalg.eigvalsh(np.cov(triangulated_points.T))**0.5)
            cov = np.linalg.det(np.cov(triangulated_points.T))**(1/6)
            for p3, (pi, uv1, uv2) in zip(triangulated_points, matches):
                if cov < points_3d[pi][1]:
                    points_3d[pi] = (p3, cov)
                    # points_3d[pi] = (np.zeros(3), cov)

    # print(''.join(['#' if 0<=p[1]<np.inf else '.' for p in points_3d]))
    points = []
    for p, cov in points_3d:
        assert 0 <= cov < np.inf
        points.append(p)
    return poses, np.array(points)



def exp_so3t(T):
    phi = T[0:3]
    t = T[3:6]
    R, _ = cv.Rodrigues(phi)
    return R, t

def log_so3t(R, t):
    assert np.linalg.norm(R@R.T-np.eye(3)) < 1e-6
    phi, _ = cv.Rodrigues(R)
    return np.concatenate((phi.flatten(), t.flatten()))

def bundle_adjustment_01(poses, points, observations):
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

    params = res.x
    so3ts = params[:6*n_pose].reshape((n_pose, 6))
    poses = [exp_so3t(so3t) for so3t in so3ts]
    points_3d = params[6*n_pose:].reshape((n_point, 3))
    return poses, points_3d

def bundle_adjustment_02(poses, points, observations):
    """points + poses + camera intrinsics (f,cx,cy)"""

    # params
    n_pose = len(poses)
    n_point = len(points)
    n_obs = len(observations)
    poses = np.array([log_so3t(*pose) for pose in poses])
    points_2d = np.array([uv for (pose_i, point_i, uv) in observations])
    f0 = np.sqrt(K[0,0]*K[1,1])
    cx0, cy0 = K[:2,2]
    params_init = np.concatenate(([f0,cx0,cy0], poses.flatten(), points.flatten()))
    poses_i = np.array([o[0] for o in observations])
    points_i = np.array([o[1] for o in observations])

    # function
    def fun(params):
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
    # plt.plot(abs(fun(res.x)), '.')
    # plt.yscale('log')
    # plt.show()
    # __import__('sys').exit(0)

    params = res.x
    print("camera intrinsics:", params[:3])
    so3ts = params[3:3+6*n_pose].reshape((n_pose, 6))
    poses = [exp_so3t(so3t) for so3t in so3ts]
    points_3d = params[3+6*n_pose:].reshape((n_point, 3))
    return poses, points_3d

def bundle_adjustment_03(poses, points, observations):
    """points + poses + camera intrinsics (f,cx,cy,*dist_coeffs)"""

    # params
    n_int = 7  # number of camera intrinsics, 3+len(dist_coeffs)
    n_pose = len(poses)
    n_point = len(points)
    n_obs = len(observations)
    poses = np.array([log_so3t(*pose) for pose in poses])
    points_2d = np.array([uv for (pose_i, point_i, uv) in observations])
    f0 = np.sqrt(K[0,0]*K[1,1])
    cx0, cy0 = K[:2,2]
    params_init = np.concatenate(([f0,cx0,cy0]+[0]*(n_int-3), poses.flatten(), points.flatten()))
    poses_i = np.array([o[0] for o in observations])
    points_i = np.array([o[1] for o in observations])

    # function
    def fun(params):
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

    params = res.x
    print("camera intrinsics:", params[:3], params[3:n_int])
    so3ts = params[n_int:n_int+6*n_pose].reshape((n_pose, 6))
    poses = [exp_so3t(so3t) for so3t in so3ts]
    points_3d = params[n_int+6*n_pose:].reshape((n_point, 3))
    return poses, points_3d



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
    ax.scatter(points[:,0], points[:,2], points[:,1], c=colors, zorder=4)

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
        cv.imread(f"img/{i}.jpg", cv.IMREAD_COLOR)
        # for i in [0, 1]
        # for i in [0, 1, 2]
        # for i in [1, 2, 3]
        # for i in [1, 2, 3, 4]
        # for i in [1, 2, 3, 5]
        # for i in [1, 2, 3, 4, 5]
        # for i in [0, 2, 4, 5]
        # for i in [0, 3, 6, 7]
        # for i in [0, 2, 4, 6]  # fail
        # for i in [7, 8, 9]
        # for i in [7, 8, 10]  # almost fail
        # for i in [6, 8, 10]  # fail
        # for i in [0, 1, 2, 3, 4, 5]
        for i in [0, 1, 2, 3, 4, 5, 6]
        # for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # fail
        # for i in [0, 2, 4, 6, 8, 10]  # fail
    ]
    time0 = perf_counter()
    features = extract_features(imgs)
    time1 = perf_counter()
    print("features extracted in {:.1f} ms".format(1000*(time1-time0)))

    # feature matching
    time0 = perf_counter()
    observations = match_features_pairwise(features)
    time1 = perf_counter()
    print("features matched in {:.1f} ms".format(1000*(time1-time0)))

    # pose estimation
    time0 = perf_counter()
    poses, points = estimate_poses_points(observations)
    time1 = perf_counter()
    print("poses estimated in {:.1f} ms".format(1000*(time1-time0)))

    # bundle adjustment
    time0 = perf_counter()
    poses, points = bundle_adjustment_03(poses, points, observations)
    time1 = perf_counter()
    print("BA solved in {:.1f} ms".format(1000*(time1-time0)))

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
    ax.computed_zorder = False
    sc = np.linalg.det(np.cov(points.T))**(1/6)
    print('sc:', sc)
    sc = 1.0
    for R, t in poses:
        plot_camera(ax, R, t, sc)
    # plot_points(ax, points, colors)
    plot_points(ax, points, colors, 2.5)
    set_axes_equal(ax)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.invert_zaxis()
    plt.show()
