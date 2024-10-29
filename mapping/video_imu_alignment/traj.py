import torch
from utils import *
from typing import Optional


class Trajectory(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t):
        pos, quat = None, None
        return pos, quat

    def forward_grad(self, t):
        p, dpdt, dpdt2, q, dqdt = [None] * 5
        return p, dpdt, dpdt2, q, dqdt


class IMUTrajectory(torch.nn.Module):

    def __init__(self, traj: Trajectory,
                 extr_R: Optional[np.ndarray],
                 extr_t: Optional[np.ndarray],
                 calibrate_extr = True,
                 fps: float=-1):
        super().__init__()
        self.traj = traj
        extr_r = torch.from_numpy(log_so3(extr_R)).float().to(device)
        extr_t = torch.from_numpy(extr_t).float().to(device)
        self.extr_r = torch.nn.Parameter(extr_r)
        self.extr_t = torch.nn.Parameter(extr_t)
        self.calibrate_extr = calibrate_extr
        self.dt_ = torch.zeros((1,)).float().to(device)
        if fps > 0:
            self.dt_ = torch.nn.Parameter(self.dt_)
        self.fps = fps

    def parameters(self):
        params = []
        if self.calibrate_extr:
            params = [self.extr_r, self.extr_t]
        if self.fps > 0:
            params.append(self.dt_)
        return self.traj.parameters() + params

    @property
    def extr_R(self):
        return exp_so3(self.extr_r)

    @property
    def extr_q(self):
        return so3_to_quat(self.extr_r)

    @property
    def dt(self):
        return self.dt_
        return self.dt_ / self.fps

    def forward(self, t, imu=False):
        if imu:
            t = t - self.dt
        return self.traj.forward(t.flatten())

    def forward_grad(self, t, imu=False):
        if imu:
            t = t - self.dt
        return self.traj.forward_grad(t.flatten())

    def get_measurements(self, t):
        p, v, a, q, q_dot, q_dot2 = self.forward_grad(t, imu=True)

        with torch.no_grad():
            q_norm = torch.norm(q, dim=1, keepdim=True)
            assert (abs(q_norm-1) < 1e-5).all()

        q_inv = q * torch.tensor([[1]+[-1]*3]).to(q)
        # print(quat_mul(q_inv, q_dot)[...,0].abs().mean())
        w = 2.0 * quat_mul(q_inv, q_dot)[..., 1:]
        w_dot = 2.0 * quat_mul(q_inv, q_dot2)[..., 1:]

        g = torch.tensor([0.0, 0.0, -9.80665], device=a.device).unsqueeze(0)
        a = (quat_to_rotmat(q).transpose(-2,-1) @ (a-g).unsqueeze(-1)).squeeze(-1)

        extr_R = self.extr_R
        extr_t = self.extr_t.unsqueeze(0)
        if True:
            a_wdot = torch.cross(w_dot, extr_t, dim=1)
            a_centrifugal = torch.cross(w, torch.cross(w, extr_t, dim=1), dim=1)
            a_coriolis = torch.zeros_like(a)
            a = a + a_wdot + a_centrifugal + a_coriolis
        a = a @ extr_R.T
        w = w @ extr_R.T

        return a, w
