from recording_pb2 import VideoCaptureData
import os
import numpy as np
import torch
import json
import yaml
import matplotlib.pyplot as plt

from traj import Trajectory
from traj_siren import SirenTrajectory


device = "cuda" if torch.cuda.is_available() else "cpu"


def rotmat_to_quat(rot):
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

    return q

def quat_to_rotmat(q):
    if isinstance(q, torch.Tensor):
        w, x, y, z = q.T
        return torch.stack([
            torch.stack([1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w]),
            torch.stack([2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w]),
            torch.stack([2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2])
        ]).permute(2, 0, 1)
    else:
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        return np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2]
        ])

def quat_mul(q, r):
    t0 = q[:, 0] * r[:, 0] - torch.sum(q[:, 1:] * r[:, 1:], dim=1)
    t1 = q[:, 0].unsqueeze(1) * r[:, 1:] + r[:, 0].unsqueeze(1) * q[:, 1:] + torch.cross(q[:, 1:], r[:, 1:], dim=1)
    return torch.cat((t0.unsqueeze(1), t1), dim=1)


def get_frame_data(work_dir):

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

    return (
        torch.tensor(np.array(times), dtype=torch.float, device=device),
        torch.tensor(np.array(pos), dtype=torch.float, device=device),
        torch.tensor(np.array(quat), dtype=torch.float, device=device),
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

    # original: x left, y down, z back
    accel = (np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]) @ accel.T).T
    # original: x up, y front, z right
    gyro = (np.array([
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0]
    ]) @ gyro.T).T
    # normalize to: x right, y down, z back

    fig, ax = plt.subplots(2, 1, sharex='all', figsize=(12.0, 8.0))

    params = { 'linewidth': 1, 'markersize': 1 }
    for i in range(3):
        ax[0].plot(times, accel[:, i], "rgb"[i]+".-", **params)
        ax[1].plot(times, gyro[:, i], "rgb"[i]+".-", **params)
    ax[0].plot(times, np.linalg.norm(accel, axis=1), "k.-", **params)
    ax[1].plot(times, np.linalg.norm(gyro, axis=1), "k.-", **params)
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



def test_traj_grad(traj):
    def fun(t):
        # return traj(t)
        # return traj(t**2)
        return traj(torch.sin(2*t))
    def fun_grad(t):
        # return traj.forward_grad(t, torch.ones_like(t))
        # return traj.forward_grad(t**2, 2.0*t)
        return traj.forward_grad(torch.sin(2*t), 2*torch.cos(2*t))
    def fun_grad2(t):
        # return traj.forward_grad2(t, torch.ones_like(t), 0.0*t)
        # return traj.forward_grad2(t**2, 2.0*t, 2.0*torch.ones_like(t))
        return traj.forward_grad2(torch.sin(2*t), 2*torch.cos(2*t), -4*torch.sin(2*t))
    t = torch.randn((100, 1))
    h = 0.01
    x0 = fun(t)
    xt0 = (fun(t+h)-fun(t-h)) / (2.0*h)
    xtt0 = (fun(t+h)+fun(t-h)-2.0*x0) / (h*h)
    x1, xt1 = fun_grad(t)
    xtt1 = (fun_grad(t+h)[1]-fun_grad(t-h)[1]) / (2.0*h)
    x2, xt2, xtt2 = fun_grad2(t)

    def check_close(msg, a, b):
        err = abs(b-a).mean().item()
        mag = abs(a).mean().item()
        print(msg, "{:.4f}".format(err/(mag+1e-12)), '\t', err, mag)
    check_close("x0|x1:", x0, x1)
    check_close("x1|x2:", x1, x2)
    check_close("xt0|xt1:", xt0, xt1)
    check_close("xt1|xt2:", xt1, xt2)
    check_close("xtt0|xtt1:", xtt0, xtt1)
    check_close("xtt0|xtt2:", xtt0, xtt2)
    check_close("xtt1|xtt2:", xtt1, xtt2)
    __import__('sys').exit(0)


def get_imu_measurements(a, q, q_dot):
    with torch.no_grad():
        q_norm = torch.norm(q, dim=1, keepdim=True)
        assert (abs(q_norm-1) < 1e-5).all()

    w = 2 * (
        q[:, 0].unsqueeze(1) * q_dot[:, 1:] -
        q_dot[:, 0].unsqueeze(1) * q[:, 1:] -
        torch.cross(q_dot[:, 1:], q[:, 1:], dim=1)
    )

    g = torch.tensor([0.0, 0.0, -9.80665], device=a.device).unsqueeze(0).expand(len(a), -1)
    a = (quat_to_rotmat(q).transpose(-2,-1) @ (a+g).unsqueeze(-1)).squeeze(-1)

    return a, w


def procrustes_transform(points1, points2):

    weight = torch.ones((len(points1), 1)).to(points1) / len(points1)

    centroid1 = (weight * points1).sum(dim=0)
    centroid2 = (weight * points2).sum(dim=0)
    centered1 = points1 - centroid1
    centered2 = points2 - centroid2
    cov = (centered2 * weight).transpose(-1, -2) @ centered1
    U, S, Vh = torch.linalg.svd(cov)
    s = torch.sum(S) / torch.sum(torch.square(centered2))
    S = torch.eye(3, device=device).float()
    S[2, 2] = (U.det() * Vh.det()).sign()
    R = U @ S @ Vh
    t = centroid2 - s * R @ centroid1
    return (s, R, t)


def get_loss(traj: Trajectory, frame_data, imu_data, imu_batch_size=-1):

    # loss for frames, L1
    timef, pos, quat = frame_data
    p, v, a, q, q_dot = traj.forward_grad(timef)
    s, R, t = procrustes_transform(pos, p)
    p_residual = (s * pos @ R.T + t) - p
    R_residual = R.unsqueeze(0) @ quat_to_rotmat(quat) - quat_to_rotmat(q)
    assert (p_residual**2).sum() <= ((pos-p)**2).sum()
    assert (p_residual**2).sum() <= ((s*pos+t-p)**2).sum()
    lossf = torch.norm(p_residual, dim=-1).mean() + \
        0.4 * torch.norm(R_residual, dim=(-2, -1)).mean()

    # loss for imu, L2
    timei, accel, gyro = imu_data
    if imu_batch_size > 0:
        batch = torch.randint(len(timei), (imu_batch_size,)).to(device)
        timei, accel, gyro = timei[batch], accel[batch], gyro[batch]
    p, v, a0, q, q_dot = traj.forward_grad(timei)
    a, w = get_imu_measurements(a0, q, q_dot)
    lossi = torch.sum((a-accel)**2, dim=-1).mean() + \
        10.0 * torch.sum((w-gyro)**2, dim=-1).mean()

    # loss for gravity, L2
    a1 = (quat_to_rotmat(q) @ accel.unsqueeze(-1)).squeeze()
    g = torch.tensor([0.0, 0.0, -9.80665], device=a.device).unsqueeze(0).expand(len(a1), -1)
    lossg = ((a1-g)**2).mean()**0.5

    return 10.0 * lossf, 0.5 * lossi, 1.0 * lossg


def get_rigid_transform(traj: Trajectory, frame_data):
    timef, pos, quat = frame_data
    p, v, a, q, q_dot = traj.forward_grad(timef)
    return procrustes_transform(pos, p)


def train_trajectory_adam(traj, frame_data, imu_data, imu_batch_size=-1, num_epochs=200):

    optimizer = torch.optim.Adam(traj.parameters(), lr=5e-3)

    if imu_batch_size > 0:
        num_batches = int(len(imu_data[0])/imu_batch_size+1)
    else:
        num_batches = 1

    for epoch in range(num_epochs):
        traj.train()
        total_loss = np.array([0.0, 0.0, 0.0])
        for i in range(num_batches):
            optimizer.zero_grad()
            loss = get_loss(traj, frame_data, imu_data, imu_batch_size)
            sum(loss).backward()
            optimizer.step()
            total_loss += [l.item() for l in loss]
        total_loss /= num_batches

        if epoch == 0 or (epoch+1) % 10 == 0:
            print(f'{epoch+1}/{num_epochs}',
                  f'lossf {total_loss[0]:.6f}',
                  f'lossi {total_loss[1]:.6f}',
                  f'lossg {total_loss[2]:.6f}',
                  sep='  ')
    
    return traj


def train_trajectory_lbfgs(traj, frame_data, imu_data, num_epochs=200):

    optimizer = torch.optim.LBFGS(
        traj.parameters(), max_iter=num_epochs,
        line_search_fn="strong_wolfe",
        history_size=40
    )

    loss = 0.0
    nfev = 0
    def closure():
        nonlocal nfev, loss
        optimizer.zero_grad()
        loss = get_loss(traj, frame_data, imu_data)
        sum(loss).backward()
        nfev += 1

        if (nfev+1) % 10 == 0:
            print(f'{nfev+1}/{num_epochs}',
                  f'lossf {loss[0].item():.6f}',
                  f'lossi {loss[1].item():.6f}',
                  f'lossg {loss[2].item():.6f}',
                  sep='  ')

        return sum(loss)

    for epoch in range(1):
        optimizer.step(closure)

    return traj


def train_trajectory(traj, frame_data, imu_data):
    # return train_trajectory_lbfgs(traj, frame_data, imu_data, 200)
    # traj = train_trajectory_adam(traj, frame_data, imu_data, num_epochs=200)
    traj = train_trajectory_adam(traj, frame_data, imu_data, num_epochs=500)
    traj = train_trajectory_lbfgs(traj, frame_data, imu_data, 500)
    return traj



@torch.no_grad()
def plot_trajectory(traj, frame_data, imu_data, point_cloud=None):

    fig, axs = plt.subplots(2, 1, sharex='all', figsize=(12.0, 8.0))
    times, accel, gyro = [x.cpu().numpy() for x in imu_data]

    p, v, a, q, q_dot = traj.forward_grad(imu_data[0])
    a, w = [x.cpu().numpy() for x in get_imu_measurements(a, q, q_dot)]

    params = { 'linewidth': 1, 'markersize': 1 }
    for i in range(3):
        axs[0].plot(times, accel[:, i], "rgb"[i]+".-", **params)
        axs[1].plot(times, gyro[:, i], "rgb"[i]+".-", **params)
        axs[0].plot(times, a[:, i], "rgb"[i]+"--", **params)
        axs[1].plot(times, w[:, i], "rgb"[i]+"--", **params)
    axs[0].plot(times, np.linalg.norm(accel, axis=1), "k.-", **params)
    axs[1].plot(times, np.linalg.norm(gyro, axis=1), "k.-", **params)
    axs[0].plot(times, np.linalg.norm(a, axis=1), "k--", **params)
    axs[1].plot(times, np.linalg.norm(w, axis=1), "k--", **params)
    for ax in axs:
        ax.grid()
        ax.set_xlim([times[0], times[-1]])
    axs[0].set_ylabel("accel")
    axs[1].set_ylabel("gyro")
    axs[-1].set_xlabel('Time [s]')
    fig.tight_layout()
    plt.savefig(os.path.join(work_dir, 'imu.pdf'))
    # plt.show()
    plt.close(fig)

    ts, tR, tt = get_rigid_transform(traj, frame_data)
    ts = ts.cpu().numpy()
    tR = tR.cpu().numpy()
    tt = tt.cpu().numpy()
    print('s:', ts, sep='\n')
    print('R:', tR, sep='\n')
    print('t:', tt, sep='\n')


    axs = plt.axes(projection='3d')
    axs.computed_zorder = False

    if point_cloud is not None:
        xyz, rgb = point_cloud
        idx = np.arange(len(xyz))
        np.random.shuffle(idx)
        idx = idx[:2000]
        xyz, rgb = xyz[idx], rgb[idx]
        xyz = ts * xyz @ tR.T + tt
        axs.scatter(xyz.T[0], xyz.T[1], xyz.T[2], c=rgb/255.0, s=1)

    sc = (torch.linalg.det(torch.cov(frame_data[1].T))**(1/6)).item() / len(frame_data[0])**(1/3)

    _, ps, qs = [x.cpu().numpy() for x in frame_data]
    p1s, q1s = [x.cpu().numpy() for x in traj.forward(frame_data[0])]
    for p, q, p1, q1 in zip(ps, qs, p1s, q1s):
        p = ts * tR @ p + tt
        R = ts * tR @ quat_to_rotmat(q)
        R1 = ts * quat_to_rotmat(q1)
        axs.plot([p[0], p1[0]], [p[1], p1[1]], [p[2], p1[2]], 'm')
        for i in range(3):
            a = p1 + sc * R1.T[i] #* (-1 if i > 0 else 1)
            axs.plot([p1[0], a[0]], [p1[1], a[1]], [p1[2], a[2]], 'rgb'[i])
        for i in range(3):
            a = p + sc * R.T[i] #* (-1 if i > 0 else 1)
            axs.plot([p[0], a[0]], [p[1], a[1]], [p[2], a[2]], 'rgb'[i])
        axs.plot(p[0], p[1], p[2], 'k.')

    p, q = [x.cpu().numpy() for x in traj.forward(imu_data[0])]
    axs.plot(p.T[0], p.T[1], p.T[2], 'k-', linewidth=1)
    # R = np.array([quat_to_rotmat(qi).T for qi in q])
    # p1 = p + 1.0*sc * np.einsum('nij,nj->ni',R,accel) / 9.8
    # p1 = p + 1.0*sc * accel / 9.8
    # axs.plot(p1.T[0], p1.T[1], p1.T[2], 'k-', linewidth=1)

    set_axes_equal(axs)
    axs.set_xlabel('x')
    axs.set_ylabel('y')
    axs.set_zlabel('z')
    plt.show()


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
    torch.manual_seed(42)

    config = yaml.safe_load(open("config.yaml"))
    work_dir = config['work_dir']

    frame_data = get_frame_data(work_dir)
    imu_data = get_imu_data(work_dir)
    point_cloud = get_point_cloud(work_dir)

    omega0 = imu_data[0][-1]/30.0
    if True:
        nh = 15 * imu_data[0][-1].item()**0.5
        nh = int(nh//8+1)*8
        traj = SirenTrajectory(n_hidden=nh, depth=2, first_omega_0=omega0).to(device)
    else:
        nh = 8 * imu_data[0][-1].item()
        nh = int(nh//64+1)*64
        traj = SirenTrajectory(n_hidden=nh, depth=1, first_omega_0=omega0).to(device)
    # test_traj_grad(traj)
    traj = train_trajectory(traj, frame_data, imu_data)
    plot_trajectory(traj, frame_data, imu_data, point_cloud)

