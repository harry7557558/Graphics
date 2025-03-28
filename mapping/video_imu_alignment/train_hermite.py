from recording_pb2 import VideoCaptureData
import os
import numpy as np
import torch
import json
import yaml
import matplotlib.pyplot as plt

from traj import Trajectory
from traj_hermite import HermiteTrajectory
from utils import *



def get_imu_measurements(a, q, q_dot):
    with torch.no_grad():
        q_norm = torch.norm(q, dim=1, keepdim=True)
        assert (abs(q_norm-1) < 1e-5).all()

    q_inv = q * torch.tensor([[1]+[-1]*3]).to(q)
    w = 2 * quat_mul(q_inv, q_dot)[..., 1:]

    g = torch.tensor([0.0, 0.0, -9.80665], device=a.device).unsqueeze(0)
    a = (quat_to_rotmat(q).transpose(-2,-1) @ (a-g).unsqueeze(-1)).squeeze(-1)

    return a, w


def procrustes_transform(points1, points2):
    weight = torch.ones((len(points1), 1)).to(points1) / len(points1)
    cov = (points2 * weight).transpose(-1, -2) @ points1
    U, S, Vh = torch.linalg.svd(cov)
    s = torch.sum(S) / torch.sum(torch.square(points2))
    S = torch.eye(3, device=device).float()
    S[2, 2] = (U.det() * Vh.det()).sign()
    R = U @ S @ Vh
    return (s, R)

def get_loss(traj: Trajectory, frame_data, imu_data, imu_batch_size=-1):

    timef, pos, quat = frame_data
    timei, accel, gyro = imu_data

    # loss for frame, L1
    p, v, a, q, q_dot = traj.forward_grad(timef)
    p_residual = p - pos
    q_residual = q - quat
    lossf = torch.norm(p_residual, dim=-1).mean() + \
        0.4 * torch.norm(q_residual, dim=1).mean()

    # loss for imu, L2
    p, v, a, q, q_dot = traj.forward_grad(timei)
    g = torch.tensor([0.0, 0.0, -9.80665]).to(accel)
    accel_g = (quat_to_rotmat(q) @ accel.unsqueeze(-1)).squeeze(-1) + g
    s, R = procrustes_transform(a, accel_g)
    # ai, wi = get_imu_measurements(a, q, q_dot)  # TODO
    a_residual = (s * a @ R.T) - accel_g
    # w_residual = wi - gyro
    assert (a_residual**2).sum() <= 1.0001*((a-accel_g)**2).sum()
    assert (a_residual**2).sum() <= 1.0001*((s*a-accel_g)**2).sum()
    lossi = torch.sum(a_residual**2, dim=-1).mean()
    # lossi = lossi + 1.0 * s * torch.sum(w_residual**2, dim=-1).mean()
    # lossi = lossi + 1.0 * s * (torch.sum(wi**2, dim=-1).mean()-torch.sum(gyro**2, dim=-1).mean())**2
    # lossi = lossi + 0.1 * torch.sum(q_dot**2, dim=-1).mean()

    return 10.0 * lossf, 0.5 * lossi


def get_rigid_transform(traj: Trajectory, imu_data):
    timei, accel, gyro = imu_data
    p, v, a, q, q_dot = traj.forward_grad(timei)
    accel_g = (quat_to_rotmat(q) @ accel.unsqueeze(-1)).squeeze(-1)
    return procrustes_transform(a, accel_g)


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
                  f'lossf {loss[0].item():.6f}',
                  f'lossi {loss[1].item():.6f}',
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
        sum(loss[1:]).backward()
        nfev += 1

        if (nfev+1) % 10 == 0:
            print(f'{nfev+1}/{num_epochs}',
                  f'lossf {loss[0].item():.6f}',
                  f'lossi {loss[1].item():.6f}',
                  sep='  ')

        return sum(loss)

    for epoch in range(1):
        optimizer.step(closure)

    return traj


def train_trajectory(traj, frame_data, imu_data):
    torch.autograd.set_detect_anomaly(True)
    # return train_trajectory_lbfgs(traj, frame_data, imu_data, 200)
    traj = train_trajectory_adam(traj, frame_data, imu_data, num_epochs=200)
    # traj = train_trajectory_adam(traj, frame_data, imu_data, num_epochs=500)
    traj = train_trajectory_lbfgs(traj, frame_data, imu_data, 200)
    return traj



@torch.no_grad()
def plot_trajectory(traj, frame_data, imu_data, point_cloud=None):

    ts, tR = get_rigid_transform(traj, imu_data)
    ts = ts.cpu().numpy()
    tR = tR.cpu().numpy()
    print('s:', ts, sep='\n')
    print('R:', tR, sep='\n')

    fig, axs = plt.subplots(2, 1, sharex='all', figsize=(12.0, 8.0))
    times, accel, gyro = [x.cpu().numpy() for x in imu_data]

    p, v, a, q, q_dot = traj.forward_grad(imu_data[0])
    a, w = [x.cpu().numpy() for x in get_imu_measurements(a, q, q_dot)]
    a = ts * a @ tR.T
    w = ts * a @ tR.T

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


    axs = plt.axes(projection='3d')
    axs.computed_zorder = False

    if point_cloud is not None:
        xyz, rgb = point_cloud
        idx = np.arange(len(xyz))
        np.random.shuffle(idx)
        idx = idx[:2000]
        xyz, rgb = xyz[idx], rgb[idx]
        xyz = ts * xyz @ tR.T
        axs.scatter(xyz.T[0], xyz.T[1], xyz.T[2], c=rgb/255.0, s=1)

    sc = (torch.linalg.det(torch.cov(frame_data[1].T))**(1/6)).item() / len(frame_data[0])**(1/3)

    _, ps, qs = [x.cpu().numpy() for x in frame_data]
    p1s, q1s = [x.cpu().numpy() for x in traj.forward(frame_data[0])]
    for p, q, p1, q1 in zip(ps, qs, p1s, q1s):
        p = ts * tR @ p
        R = ts * tR @ quat_to_rotmat(q)
        for i in range(3):
            a = p + sc * R.T[i] #* (-1 if i > 0 else 1)
            axs.plot([p[0], a[0]], [p[1], a[1]], [p[2], a[2]], 'rgb'[i])
        axs.plot(p[0], p[1], p[2], 'k.')

    p, q = [x.cpu().numpy() for x in traj.forward(imu_data[0])]
    p = ts * p @ tR.T
    axs.plot(p.T[0], p.T[1], p.T[2], 'k-', linewidth=1)

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

    traj = HermiteTrajectory(*frame_data).to(device)
    traj = train_trajectory(traj, frame_data, imu_data)
    plot_trajectory(traj, frame_data, imu_data, point_cloud)

