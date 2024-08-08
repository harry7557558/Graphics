import numpy as np
import torch

from traj import Trajectory

device = "cuda" if torch.cuda.is_available() else "cpu"


# trajectory parameterization
# https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb

class SineLayer(torch.nn.Module):
    def __init__(self, n_in, n_out, is_first=False, omega_0=30):
        super().__init__()

        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = n_in
        self.linear = torch.nn.Linear(n_in, n_out).to(device)

        w = 1 / self.in_features if self.is_first else \
            np.sqrt(6 / self.in_features) / self.omega_0
        self.linear.weight.data.uniform_(-w, w)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

    def forward_grad(self, x, dxdt):
        w = self.omega_0 * self.linear.weight.unsqueeze(0)
        b = self.omega_0 * self.linear.bias.unsqueeze(0)
        wx = (w @ x.unsqueeze(-1)).squeeze(-1)
        wxb = wx + b
        y = torch.sin(wxb)
        dydx = w * torch.cos(wxb).unsqueeze(-1)
        dydt = (dydx @ dxdt.unsqueeze(-1)).squeeze(-1)
        return y, dydt

    def forward_grad2(self, x, dxdt, dxdt2):
        w = self.omega_0 * self.linear.weight.unsqueeze(0)
        b = self.omega_0 * self.linear.bias.unsqueeze(0)
        wx = (w @ x.unsqueeze(-1)).squeeze(-1)
        wxb = wx + b
        y = torch.sin(wxb)
        dydx = w * torch.cos(wxb).unsqueeze(-1)
        dydt = (dydx @ dxdt.unsqueeze(-1)).squeeze(-1)
        dydt2 = torch.einsum('nij,nik,nj,nk->ni',
                             w, w * -y.unsqueeze(-1), dxdt, dxdt) + \
                (dydx @ dxdt2.unsqueeze(-1)).squeeze(-1)
        return y, dydt, dydt2


class Siren(torch.nn.Module):
    def __init__(self, n_in, n_out, n_hidden, depth, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.layer_in = SineLayer(
            n_in, n_hidden, is_first=True, omega_0=first_omega_0)
        # test_traj_grad(self.layer_in)
        self.layers_hidden = [
            SineLayer(n_hidden, n_hidden, is_first=False, omega_0=hidden_omega_0)
            for _ in range(depth)
        ]
        self.layer_out = torch.nn.Linear(n_hidden, n_out)
        w = np.sqrt(6 / n_hidden) / hidden_omega_0
        self.layer_out.weight.data.uniform_(-w, w)

    def forward(self, x):
        x = self.layer_in(x)
        for layer in self.layers_hidden:
            # x = layer(x)
            x = x + layer(x)
        return self.layer_out(x)

    def forward_grad(self, x, dxdt):
        x, dxdt = self.layer_in.forward_grad(x, dxdt)
        for layer in self.layers_hidden:
            # x, dxdt = layer.forward_grad(x, dxdt)
            l, dldt = layer.forward_grad(x, dxdt)
            x, dxdt = x+l, dxdt+dldt
        w = self.layer_out.weight.unsqueeze(0)
        b = self.layer_out.bias.unsqueeze(0)
        y = (w @ x.unsqueeze(2)).squeeze(2) + b
        dydt = (w @ dxdt.unsqueeze(2)).squeeze(2)
        return y, dydt

    def forward_grad2(self, x, dxdt, dxdt2):
        x, dxdt, dxdt2 = self.layer_in.forward_grad2(x, dxdt, dxdt2)
        for layer in self.layers_hidden:
            # x, dxdt, dxdt2 = layer.forward_grad2(x, dxdt, dxdt2)
            l, dldt, dldt2 = layer.forward_grad2(x, dxdt, dxdt2)
            x, dxdt, dxdt2 = x+l, dxdt+dldt, dxdt2+dldt2
        w = self.layer_out.weight.unsqueeze(0)
        b = self.layer_out.bias.unsqueeze(0)
        y = (w @ x.unsqueeze(-1)).squeeze(-1) + b
        dydt = (w @ dxdt.unsqueeze(-1)).squeeze(-1)
        dydt2 = (w @ dxdt2.unsqueeze(-1)).squeeze(-1)
        return y, dydt, dydt2


class SirenTrajectory(Trajectory):
    def __init__(self, n_hidden, depth, first_omega_0=1.0):
        super().__init__()
        self.model = Siren(1, 7, n_hidden, depth, first_omega_0=first_omega_0)
        # test_traj_grad(self.model)

    def forward(self, t):
        t = t.reshape((-1, 1))
        y = self.model(t)
        pos = y[:, :3]
        quat = y[:, 3:] / torch.norm(y[:, 3:], dim=1, keepdim=True)
        # return quat
        return pos, quat

    def forward_grad(self, t):
        t = t.reshape((-1, 1))
        y, dydt, dydt2 = self.model.forward_grad2(t, torch.ones_like(t), torch.zeros_like(t))
        p, dpdt, dpdt2 = y[:, :3], dydt[:, :3], dydt2[:, :3]
        q, dqdt = y[:, 3:], dydt[:, 3:]
        q_invnorm = 1.0 / torch.norm(q, dim=1, keepdim=True)
        q = q * q_invnorm
        dqdt = (((torch.eye(4, device=device).unsqueeze(0) - torch.einsum('ni,nj->nij', q, q)) \
                 * q_invnorm.unsqueeze(-1)) @ dqdt.unsqueeze(2)).squeeze(2)
        # return q, dqdt
        return p, dpdt, dpdt2, q, dqdt
