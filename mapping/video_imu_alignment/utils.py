import torch
import numpy as np
import matplotlib.pyplot as plt

import os
from recording_pb2 import VideoCaptureData


device = "cuda" if torch.cuda.is_available() else "cpu"



def rotmat_to_quat(rot: np.ndarray):
    rot = np.array(rot)
    assert rot.shape[:] == (3, 3), "Input must be a single 3x3 matrix"
    identity = np.eye(3)
    orthogonal_check = np.allclose(rot.T @ rot, identity, atol=1e-6)
    assert orthogonal_check, "Rotation matrix must be orthogonal"
    det_check = np.allclose(np.linalg.det(rot), 1.0, atol=1e-6)
    assert det_check, "Rotation matrix must have a determinant of 1"

    m00, m01, m02 = rot[..., 0, 0], rot[..., 0, 1], rot[..., 0, 2]
    m10, m11, m12 = rot[..., 1, 0], rot[..., 1, 1], rot[..., 1, 2]
    m20, m21, m22 = rot[..., 2, 0], rot[..., 2, 1], rot[..., 2, 2]

    trace = m00 + m11 + m22
    q = np.zeros(rot.shape[:-2] + (4,))

    def safe_sqrt(x):
        return np.sqrt(np.clip(x, a_min=0, a_max=None))

    if np.all(trace > 0):
        s = safe_sqrt(trace + 1.0) * 2
        q[..., 0] = 0.25 * s
        q[..., 1] = (m21 - m12) / s
        q[..., 2] = (m02 - m20) / s
        q[..., 3] = (m10 - m01) / s
    elif np.all((m00 > m11) & (m00 > m22)):
        s = safe_sqrt(1.0 + m00 - m11 - m22) * 2
        q[..., 0] = (m21 - m12) / s
        q[..., 1] = 0.25 * s
        q[..., 2] = (m01 + m10) / s
        q[..., 3] = (m02 + m20) / s
    elif np.all(m11 > m22):
        s = safe_sqrt(1.0 + m11 - m00 - m22) * 2
        q[..., 0] = (m02 - m20) / s
        q[..., 1] = (m01 + m10) / s
        q[..., 2] = 0.25 * s
        q[..., 3] = (m12 + m21) / s
    else:
        s = safe_sqrt(1.0 + m22 - m00 - m11) * 2
        q[..., 0] = (m10 - m01) / s
        q[..., 1] = (m02 + m20) / s
        q[..., 2] = (m12 + m21) / s
        q[..., 3] = 0.25 * s

    # return q
    return q * np.sign(q[..., :1])


def quat_to_rotmat(q):
    if isinstance(q, torch.Tensor):
        w, x, y, z = q.T
        return torch.stack([
            torch.stack([1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w]),
            torch.stack([2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w]),
            torch.stack([2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2])
        ]).permute(2, 0, 1)
    elif isinstance(q, (np.ndarray, np.generic)):
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        return np.stack([
            1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w,
            2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w,
            2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2
        ], axis=-1).reshape(*q.shape[:-1], 3, 3)
    else:
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        return np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2]
        ])


def quat_mul(q: torch.Tensor, r: torch.Tensor):
    t0 = q[:, 0] * r[:, 0] - torch.sum(q[:, 1:] * r[:, 1:], dim=1)
    t1 = q[:, 0].unsqueeze(1) * r[:, 1:] + r[:, 0].unsqueeze(1) * q[:, 1:] + torch.cross(q[:, 1:], r[:, 1:], dim=1)
    return torch.cat((t0.unsqueeze(1), t1), dim=1)


def rotmat_to_quat_torch_(R: torch.Tensor):
    assert R.shape == (3, 3)
    q_w = torch.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.0
    q_x = (R[2, 1] - R[1, 2]) / (4.0 * q_w)
    q_y = (R[0, 2] - R[2, 0]) / (4.0 * q_w)
    q_z = (R[1, 0] - R[0, 1]) / (4.0 * q_w)
    q = torch.stack([q_w, q_x, q_y, q_z])
    return q / torch.norm(q)

def rotmat_to_quat_torch(rot: torch.Tensor):
    m00, m01, m02 = rot[..., 0, 0], rot[..., 0, 1], rot[..., 0, 2]
    m10, m11, m12 = rot[..., 1, 0], rot[..., 1, 1], rot[..., 1, 2]
    m20, m21, m22 = rot[..., 2, 0], rot[..., 2, 1], rot[..., 2, 2]

    trace = m00 + m11 + m22
    q = torch.zeros(rot.shape[:-2] + (4,)).to(rot)

    def safe_sqrt(x):
        return torch.sqrt(torch.relu(x))

    if torch.all(trace > 0):
        s = safe_sqrt(trace + 1.0) * 2
        q[..., 0] = 0.25 * s
        q[..., 1] = (m21 - m12) / s
        q[..., 2] = (m02 - m20) / s
        q[..., 3] = (m10 - m01) / s
    elif torch.all((m00 > m11) & (m00 > m22)):
        s = safe_sqrt(1.0 + m00 - m11 - m22) * 2
        q[..., 0] = (m21 - m12) / s
        q[..., 1] = 0.25 * s
        q[..., 2] = (m01 + m10) / s
        q[..., 3] = (m02 + m20) / s
    elif torch.all(m11 > m22):
        s = safe_sqrt(1.0 + m11 - m00 - m22) * 2
        q[..., 0] = (m02 - m20) / s
        q[..., 1] = (m01 + m10) / s
        q[..., 2] = 0.25 * s
        q[..., 3] = (m12 + m21) / s
    else:
        s = safe_sqrt(1.0 + m22 - m00 - m11) * 2
        q[..., 0] = (m10 - m01) / s
        q[..., 1] = (m02 + m20) / s
        q[..., 2] = (m12 + m21) / s
        q[..., 3] = 0.25 * s

    return q / torch.norm(q, keepdim=True) * torch.sign(q[..., :1])


def rotmat_quat_mul(R: torch.Tensor, q: torch.Tensor):
    qR = rotmat_to_quat_torch(R)
    return quat_mul(qR.unsqueeze(0).repeat(len(q), 1), q)


def exp_so3(phi: torch.Tensor):
    theta = torch.norm(phi, dim=-1, keepdim=True)
    theta = torch.maximum(theta, 1e-12 * torch.ones_like(theta))
    n = phi / theta
    nnT = torch.einsum('...i,...j->...ij', n, n)
    n_star = torch.zeros_like(nnT)
    n_star[...,0,1] = -n[...,2]
    n_star[...,0,2] = n[...,1]
    n_star[...,1,0] = n[...,2]
    n_star[...,1,2] = -n[...,0]
    n_star[...,2,0] = -n[...,1]
    n_star[...,2,1] = n[...,0]
    I = torch.eye(3).reshape((*([1]*(len(phi.shape)-1)),3,3)).repeat(*phi.shape[:-1],1,1)
    I = I.to(phi.device)
    theta = theta.reshape((*phi.shape[:-1],1,1))
    R = torch.cos(theta) * I + \
        (1.0 - torch.cos(theta)) * nnT + \
        torch.sin(theta) * n_star
    if False:
        residual = torch.einsum('...ij,...kj->...ik', R, R) - I
        residual = torch.norm(residual, dim=(-2,-1))
        assert (residual < 1e-6).all()
    return R.contiguous()


def so3_to_quat(phi: torch.Tensor):
    """so3 to quaternion, with careful consideration for small angles"""

    theta = torch.norm(phi, dim=-1, keepdim=True)
    eps = 1e-6
    mask = theta < eps
    theta = torch.where(mask, torch.ones_like(theta), theta)

    xyz = phi / theta
    w = torch.cos(theta / 2)
    xyz = torch.sin(theta / 2) * xyz

    w = torch.where(mask, torch.ones_like(w), w)
    xyz = torch.where(mask, phi / 2, xyz)
    
    quats = torch.cat([w, xyz], dim=-1)
    return quats / torch.norm(quats, dim=-1, keepdim=True)


def log_so3_(R):
    assert np.linalg.norm(R@R.T-np.eye(3)) < 1e-6
    tr = np.trace(R)
    cos = 0.5 * (tr - 1)
    sin = 0.5 * np.sqrt(max(0, (3 - tr) * (1 + tr)))
    theta = np.arctan2(sin, cos)
    w_ = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    if abs(sin) < 1e-6:
        c = 0.5 * (1 + 1/6 * theta**2 + 7/360 * theta**4)
        return c * w_
    return theta / (2 * sin) * w_

def log_so3(R):
    """https://github.com/nurlanov-zh/so3_log_map"""
    trR = R[0, 0] + R[1, 1] + R[2, 2]
    cos_theta = max(min(0.5 * (trR - 1), 1), -1)
    sin_theta = 0.5 * np.sqrt(max(0, (3 - trR) * (1 + trR)))
    theta = np.arctan2(sin_theta, cos_theta)
    R_minus_R_T_vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    
    if abs(3 - trR) < 1e-8:
        # return log map at Theta = 0
        c = 0.5 * (1 + theta*theta / 6 + 7 / 360 * (theta**4))
        return c * R_minus_R_T_vee

    S = R + R.transpose() + (1 - trR) * np.eye(3)
    rest_tr = 3 - trR
    n = np.ones(3)
    # Fix modules of n_i
    for i in range(3):
        n[i] = np.sqrt(max(0, S[i, i] / rest_tr))
    max_i = np.argmax(n)
    
    # Fix signs according to the sign of max element
    for i in range(3):
        if i != max_i:
            n[i] *= np.sign(S[max_i, i])

    # Fix an overall sign
    if any(np.sign(n) * np.sign(R_minus_R_T_vee) < 0):
        n = -n
    return theta * n


def get_frame_data(work_dir):
    import os
    import json

    time_file = os.path.join(work_dir, "timestamps.json")
    with open(time_file) as fp:
        time_file = json.load(fp)
    time_map = {}
    for obj in time_file:
        time_map[obj['file_path']] = obj['time']

    transform_file = os.path.join(work_dir, "transforms.json")
    with open(transform_file) as fp:
        transform_file = json.load(fp)
    frames = transform_file['frames']

    times, pos, quat = [], [], []
    for frame in frames:
        time = time_map[frame['file_path']]
        c2w = np.array(frame['transform_matrix'])
        c2w = c2w @ np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        p = c2w[:3, 3]
        q = rotmat_to_quat(c2w[:3, :3])
        times.append(time)
        pos.append(p)
        quat.append(q)

    order = np.argsort(times)
    return (
        torch.tensor(np.array(times)[order], dtype=torch.float, device=device),
        torch.tensor(np.array(pos)[order], dtype=torch.float, device=device),
        torch.tensor(np.array(quat)[order], dtype=torch.float, device=device),
    )


def get_imu_data(work_dir):
    proto_file = os.path.join(work_dir, "video_meta.pb3")
    with open(proto_file,'rb') as f:
        proto = VideoCaptureData.FromString(f.read())
    # print(proto.camera_meta)
    # print(proto.imu_meta)

    times, accel, gyro = [], [], []

    for i, data in enumerate(proto.imu):
        times.append(data.time_ns)
        accel.append(getattr(data, "accel"))
        gyro.append(getattr(data, "gyro"))
    times = 1e-9 * (np.array(times)-times[0])
    accel = np.array(accel)   # (n, 3)
    gyro = np.array(gyro)  # (n, 3)

    fig, ax = plt.subplots(2, 1, sharex='all', figsize=(12.0, 8.0))

    params = { 'linewidth': 1, 'markersize': 0.5 }
    for i in range(3):
        ax[0].plot(times, accel[:, i], "rgb"[i]+".", **params)
        ax[1].plot(times, gyro[:, i], "rgb"[i]+".", **params)
    ax[0].plot(times, np.linalg.norm(accel, axis=1), "k.", **params)
    ax[1].plot(times, np.linalg.norm(gyro, axis=1), "k.", **params)
    for a in ax:
        a.grid()
        a.set_xlim([times[0], times[-1]])
    ax[0].set_ylabel("accel")
    ax[1].set_ylabel("gyro")
    ax[-1].set_xlabel('Time [s]')
    fig.tight_layout()
    plt.savefig(os.path.join(work_dir, 'imu.pdf'))
    plt.close(fig)

    return (
        torch.tensor(np.array(times), dtype=torch.float, device=device),
        torch.tensor(np.array(accel), dtype=torch.float, device=device),
        torch.tensor(np.array(gyro), dtype=torch.float, device=device),
    )


def get_point_cloud(work_dir):
    from plyfile import PlyData
    ply_file = os.path.join(work_dir, "sparse_pc.ply")
    ply_data = PlyData.read(ply_file)
    
    vertex_data = ply_data['vertex']
    xyz = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
    rgb = np.vstack((vertex_data['red'], vertex_data['green'], vertex_data['blue'])).T

    mask = np.ones(len(xyz), dtype=np.bool_)
    xyz_masked = xyz[np.where(mask)]
    mean = np.mean(xyz_masked, axis=0)
    cov_matrix = np.cov(xyz_masked, rowvar=False)
    mahalanobis_dist = np.sqrt(np.sum(np.dot((xyz-mean), np.linalg.inv(cov_matrix)) * (xyz-mean), axis=1))
    mask &= (mahalanobis_dist < 3.0)

    return xyz[mask], rgb[mask]


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
