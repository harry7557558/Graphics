from recording_pb2 import VideoCaptureData
import os
import numpy as np
import torch
import json
import yaml
import matplotlib.pyplot as plt

from traj import Trajectory
from traj_hermite import HermiteTrajectorySO3t
from utils import *



def get_imu_measurements(a, q, q_dot):
    with torch.no_grad():
        q_norm = torch.norm(q, dim=1, keepdim=True)
        assert (abs(q_norm-1) < 1e-5).all()

    q_inv = q * torch.tensor([[1]+[-1]*3]).to(q)
    # print(quat_mul(q_inv, q_dot)[...,0].abs().mean())
    w = 2 * quat_mul(q_inv, q_dot)[..., 1:]

    g = torch.tensor([0.0, 0.0, -9.80665], device=a.device).unsqueeze(0)
    a = (quat_to_rotmat(q).transpose(-2,-1) @ (a-g).unsqueeze(-1)).squeeze(-1)

    return a, w


def get_rigid_transform(traj: Trajectory):
    return traj.s, traj.R, traj.t

def get_loss(traj: Trajectory, frame_data, imu_data, imu_batch_size=-1):

    # loss for frames, L1
    timef, pos, quat = frame_data
    p, v, a, q, q_dot = traj.forward_grad(timef)
    s, R, t = get_rigid_transform(traj)
    p_residual = (s * pos @ R.T + t) - p
    R_residual = R.unsqueeze(0) @ quat_to_rotmat(quat) - quat_to_rotmat(q)
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
        50.0 * torch.sum((w-gyro)**2, dim=-1).mean()

    return 10.0 * lossf, 1.0 * lossi


def train_trajectory_adam(traj, frame_data, imu_data, num_epochs=200):

    optimizer = torch.optim.Adam(traj.parameters(), lr=1e-2)

    for epoch in range(num_epochs):
        traj.train()
        optimizer.zero_grad()
        loss = get_loss(traj, frame_data, imu_data)
        sum(loss[1:]).backward()
        optimizer.step()

        if epoch == 0 or (epoch+1) % 10 == 0:
            print(f'{epoch+1}/{num_epochs}',
                #   f'lossf {loss[0].item():.6f}',
                  f'lossi {loss[1].item():.4f}',
                  f's {traj.s.item():.4g}',
                  sep='  ')
    
    return traj


def train_trajectory_lbfgs(traj, frame_data, imu_data, num_epochs=200):

    optimizer = torch.optim.LBFGS(
        traj.parameters(), max_iter=num_epochs,
        line_search_fn="strong_wolfe",
        history_size=40,
        tolerance_grad=1e-10, tolerance_change=1e-12
    )

    loss = 0.0
    nfev = 0
    def closure():
        nonlocal nfev, loss
        optimizer.zero_grad()
        loss = get_loss(traj, frame_data, imu_data)
        sum(loss[1:]).backward()
        nfev += 1

        if (nfev+1) % 10 == 0:
            print(f'{nfev+1}/{num_epochs}',
                #   f'lossf {loss[0].item():.6f}',
                  f'lossi {loss[1].item():.4f}',
                  f's {traj.s.item():.4g}',
                  sep='  ')

        return sum(loss)

    for epoch in range(1):
        optimizer.step(closure)

    return traj


def train_trajectory(traj, frame_data, imu_data):
    torch.autograd.set_detect_anomaly(True)
    # return train_trajectory_lbfgs(traj, frame_data, imu_data, 200)
    # traj = train_trajectory_adam(traj, frame_data, imu_data, num_epochs=50)
    # traj = train_trajectory_adam(traj, frame_data, imu_data, num_epochs=500)
    traj = train_trajectory_lbfgs(traj, frame_data, imu_data, 300)
    # traj = train_trajectory_adam(traj, frame_data, imu_data, num_epochs=50)
    return traj



@torch.no_grad()
def plot_trajectory(traj, frame_data, imu_data, point_cloud=None):

    fig, axs = plt.subplots(2, 1, sharex='all', figsize=(12.0, 8.0))
    times, accel, gyro = [x.cpu().numpy() for x in imu_data]

    p, v, a, q, q_dot = traj.forward_grad(imu_data[0])
    a, w = [x.cpu().numpy() for x in get_imu_measurements(a, q, q_dot)]

    params = { 'linewidth': 1, 'markersize': 0.5 }
    for i in range(3):
        axs[0].plot(times, accel[:, i], "rgb"[i]+".", **params)
        axs[1].plot(times, gyro[:, i], "rgb"[i]+".", **params)
        axs[0].plot(times, a[:, i], "rgb"[i]+"-", **params)
        axs[1].plot(times, w[:, i], "rgb"[i]+"-", **params)
    axs[0].plot(times, np.linalg.norm(accel, axis=1), "k.", **params)
    axs[1].plot(times, np.linalg.norm(gyro, axis=1), "k.", **params)
    axs[0].plot(times, np.linalg.norm(a, axis=1), "k-", **params)
    axs[1].plot(times, np.linalg.norm(w, axis=1), "k-", **params)
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

    ts, tR, tt = get_rigid_transform(traj)
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
    a1 = np.einsum('nij,nj->ni', quat_to_rotmat(q), accel)
    p = p + 0.05 * a1
    # axs.plot(p.T[0], p.T[1], p.T[2], 'm-', linewidth=1)

    set_axes_equal(axs)
    axs.set_xlabel('x')
    axs.set_ylabel('y')
    axs.set_zlabel('z')
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(42)

    config = yaml.safe_load(open("config.yaml"))
    work_dir = config['work_dir']

    frame_data = get_frame_data(work_dir)
    imu_data = get_imu_data(work_dir)
    point_cloud = get_point_cloud(work_dir)

    mask = torch.where((imu_data[0] > frame_data[0][0]) & (imu_data[0] < frame_data[0][-1]))
    imu_data = [i[mask] for i in imu_data]

    traj = HermiteTrajectorySO3t(
        *frame_data,
        degree=7, continuity_order=2, mie_init_order=3
    ).to(device)
    traj = train_trajectory(traj, frame_data, imu_data)
    # print(traj.model.additional_coeffs)
    plot_trajectory(traj, frame_data, imu_data, point_cloud)

