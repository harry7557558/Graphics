import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# https://github.com/gaoxiang12/slambook2


IMG_SHAPE = np.array([960.0, 540.0])
K = np.array([[751.3, 0., 477.8],
              [0., 754.9, 286.8],
              [0., 0., 1.]])


def pixel2cam(p):
    return np.array([(p[0]-K[0,2])/K[0,0], (p[1]-K[1,2])/K[1,1]])


def find_feature_matches(img_1, img_2):
    detector = cv.ORB_create(1000)
    matcher = cv.BFMatcher(cv.NORM_HAMMING)
    # detector = cv.SIFT_create(1000)
    # matcher = cv.BFMatcher(cv.NORM_L2)
    
    keypoints_1, descriptors_1 = detector.detectAndCompute(img_1, None)
    keypoints_2, descriptors_2 = detector.detectAndCompute(img_2, None)
    
    matches = matcher.match(descriptors_1, descriptors_2)

    # plt.figure()
    # plt.boxplot([match.distance for match in matches])
    # plt.show()

    good_matches = []
    min_dist = min([match.distance for match in matches])
    for match in matches:
        if match.distance <= max(2 * min_dist, 30):
            good_matches.append(match)
    
    return keypoints_1, keypoints_2, good_matches

def draw_matches(img_1, keypoints_1, img_2, keypoints_2, matches):
    # # Create a new image showing the matches
    # img_matches = np.empty((max(img_1.shape[0], img_2.shape[0]), img_1.shape[1] + img_2.shape[1], 3), dtype=np.uint8)
    # cv.drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches)
    img_matches = np.empty((img_1.shape[0], img_1.shape[1], 3), dtype=np.uint8)
    cv.drawKeypoints(img_1, keypoints_1, img_matches)
    
    # Draw lines connecting the matches
    for match in matches:
        # Get the matching keypoints
        kp1 = keypoints_1[match.queryIdx].pt
        kp2 = keypoints_2[match.trainIdx].pt
        # Convert keypoints to integer
        pt1 = (int(kp1[0]), int(kp1[1]))
        # pt2 = (int(kp2[0]) + img_1.shape[1], int(kp2[1]))
        pt2 = (int(kp2[0]), int(kp2[1]))
        # Draw a line between keypoints
        cv.line(img_matches, pt1, pt2, (0, 255, 0), 1)
    
    # Display the matches
    cv.imshow("Matches", img_matches)
    cv.waitKey(0)
    cv.destroyAllWindows()


def pose_estimation(keypoints_1, keypoints_2, matches):

    points1 = np.array([keypoints_1[match.queryIdx].pt for match in matches])
    points2 = np.array([keypoints_2[match.trainIdx].pt for match in matches])

    fundamental_matrix, _ = cv.findFundamentalMat(points1, points2, cv.FM_8POINT)
    print("fundamental matrix:")
    print(fundamental_matrix)

    f = np.sqrt(K[0,0]*K[1,1])
    c = (K[0,2], K[1,2])
    essential_matrix, _ = cv.findEssentialMat(points1, points2, f, c)
    print("essential matrix:")
    print(essential_matrix)

    homography_matrix, _ = cv.findHomography(points1, points2, cv.RANSAC, 3)
    print("homography matrix:")
    print(homography_matrix)
    
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



def plot_points(ax, points_3d, max_norm, colors):
    points_3d_norms = np.linalg.norm(points_3d, axis=1)
    i = np.where(points_3d_norms<max_norm)
    filtered = points_3d[i]
    colors = colors[i]
    ax.scatter(filtered[:,0], filtered[:,2], filtered[:,1], c=colors)

def plot_camera(ax, R, t):
    points = np.array([
        (0, 0, 0),
        (0, 0, 1),
        (IMG_SHAPE[0], 0, 1),
        (IMG_SHAPE[0], IMG_SHAPE[1], 1),
        (0, IMG_SHAPE[1], 1)
    ]).T
    points_3d = R @ np.linalg.inv(K) @ points + t
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
    img_1 = cv.imread("img/0.jpg", cv.IMREAD_COLOR)
    img_2 = cv.imread("img/2.jpg", cv.IMREAD_COLOR)

    keypoints_1, keypoints_2, matches = find_feature_matches(img_1, img_2)
    print("Total {} matching points found".format(len(matches)))
    # draw_matches(img_1, keypoints_1, img_2, keypoints_2, matches)

    R, t = pose_estimation(keypoints_1, keypoints_2, matches)
    print("R:\n", R)
    print("t:\n", t)

    points_3d = triangulation(keypoints_1, keypoints_2, matches, R, t)

    points_rgb = np.zeros((len(matches), 3), dtype=np.float32)
    for i, m in enumerate(matches):
        x, y = map(int, keypoints_1[m.queryIdx].pt)
        color1 = img_1[y, x] / 255.0
        x, y = map(int, keypoints_2[m.trainIdx].pt)
        color2 = img_2[y, x] / 255.0
        points_rgb[i] = 0.5*(color1+color2)

    plt.figure(figsize=(8, 8))
    ax = plt.axes(projection='3d')
    plot_points(ax, points_3d, 1000.0, points_rgb)
    plot_camera(ax, np.eye(3), 0.0)
    plot_camera(ax, R, t)
    set_axes_equal(ax)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.invert_zaxis()
    plt.show()
