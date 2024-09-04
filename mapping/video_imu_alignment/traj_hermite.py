import torch

from typing import Union, Tuple
from jaxtyping import Float
from torch import Tensor

from traj import Trajectory

device = "cuda" if torch.cuda.is_available() else "cpu"


class HermiteSpline(torch.nn.Module):
    def __init__(self,
                 times: Float[Tensor, "n"],
                 positions: Float[Tensor, "n dim"],
                 degree: int=5, continuity_order: int=2,
                 mie_init_order: int=3
                ):
        super().__init__()

        self.n, self.dim = positions.shape
        self.t = times.float().to(device)
        assert (self.t[1:] > self.t[:-1]).all(), "t must be monotonically increasing"
        self.c0 = positions.float().to(device)  # 0th derivative

        assert degree == 5 and continuity_order == 2, "unsupported degree and continuity order"
        self.degree = degree
        self.continuity_order = continuity_order
        # TODO: generate a smooth curve by minimizing integrated jerk/snap
        self.c1 = torch.nn.Parameter(torch.zeros_like(self.c0))  # 1st derivative
        self.c2 = torch.nn.Parameter(torch.zeros_like(self.c0))  # 2nd derivative

        # Hermite weight matrix
        dt = (self.t[1:] - self.t[:-1]).reshape(-1, 1, 1)  # (n-1, 1, 1)
        cmat = torch.tensor([[
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [0, 1, 2, 3, 4, 5],
            [0, 0, 2, 6, 12, 20],
        ]]).double().to(device)  # (1, 6, 6)
        emat = torch.tensor([[
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 2, 3, 4, 5],
            [0, 0, 1, 2, 3, 4],
            [0, 0, 0, 1, 2, 3],
        ]]).double().to(device)  # (1, 6, 6)
        wmat = cmat * dt.double() ** emat  # (n-1, 6, 6)
        hmat = torch.linalg.inv(wmat)  # (n-1, 6, 6)
        self.hmat = hmat.float()  # (n-1, 6, 6)

        # Minimum jerk/snap initialization
        if mie_init_order == -1:
            return
        # compute matrix T = \int p''(t)^T p''(t) dt
        cmat = torch.ones(self.degree+1).double().to(device)
        emat = torch.arange(self.degree+1).double().to(device)
        for _ in range(mie_init_order):
            cmat *= emat
            emat = torch.concat((torch.zeros(1).to(emat), emat))[:-1]
        cmat = cmat.unsqueeze(0) * cmat.unsqueeze(-1)  # (6, 6)
        emat = emat.unsqueeze(0) + emat.unsqueeze(-1)  # (6, 6)
        emat = emat.unsqueeze(0) + 1.0  # (1, 6, 6)
        cmat = cmat.unsqueeze(0) / emat  # (1, 6, 6)
        tmat = cmat * dt.double() ** emat  # (n-1, 6, 6)
        # compute matrix W = H^T T H
        wmat = hmat.transpose(-1,-2) @ tmat @ hmat  # (n-1, 6, 6)
        # minimize[c] c^T W c
        max_iter = 200
        optimizer = torch.optim.LBFGS(
            [self.c1, self.c2], max_iter=max_iter,
            line_search_fn=None, history_size=40,
            tolerance_grad=1e-1, tolerance_change=1e-2
        )
        loss = 0.0
        nfev = 0
        def closure():
            nonlocal nfev, loss
            optimizer.zero_grad()
            c = torch.stack([
                self.c0[:-1], self.c1[:-1], self.c2[:-1],
                self.c0[1:], self.c1[1:], self.c2[1:]
            ]).transpose(0, 1).double()  # (n-1, 6, dim)
            # l = (c.transpose(-1,-2) @ wmat @ c).sum()
            l = ((c.transpose(-1,-2) @ wmat @ c)**2).sum()**0.5
            l.backward()
            nfev += 1
            if (nfev+1) % 20 == 0:
                print(f'{nfev+1}/{max_iter}',
                    f'loss {l.item():.6f}',
                    sep='  ')
            return l
        optimizer.step(closure)
        optimizer.zero_grad()
        print(f"Trajectory initialized in {nfev} iterations")

    def forward(self, x: Float[Tensor, "batch"], derivative_order: int=0) \
          -> Union[Float[Tensor, "batch dim"], Tuple[Float[Tensor, "batch dim"]]]:
        """
            If derivative order is 0, return value;
            otherwise, return a tuple of all derivatives up to the order.
        """
        assert (x[1:] > x[:-1]).all(), "x must be monotonically increasing"
        assert x[0] >= self.t[0] and x[-1] <= self.t[-1], "parameter out of range"
        i = torch.searchsorted(self.t, x, right=True) - 1
        i = torch.clip(i, 0, self.n-2)
        c = torch.stack([
            self.c0[:-1], self.c1[:-1], self.c2[:-1],
            self.c0[1:], self.c1[1:], self.c2[1:]
        ]).transpose(0, 1)  # (n-1, 6, dim)
        p_mat = torch.matmul(self.hmat, c)  # (n-1, 6, dim)
        p_mat = p_mat[i]  # (batch, 6, dim)
        f = (x-self.t[i]).unsqueeze(-1)  # (batch, 1)
        cs = torch.ones(self.degree+1)
        es = torch.arange(self.degree+1)
        t_pow = cs.unsqueeze(0).to(device) * \
            f ** es.unsqueeze(0).to(device)  # (batch, 6)
        p = torch.einsum('ni,nij->nj', t_pow, p_mat)  # (batch, dim)
        assert derivative_order >= 0, "derivative order must be non-negative"
        if derivative_order == 0:
            return p
        results = [p]
        for _ in range(derivative_order):
            cs *= es
            es = torch.concat((torch.zeros(1), es))[:-1]
            t_pow = cs.unsqueeze(0).to(device) * \
                f ** es.unsqueeze(0).to(device)  # (batch, 6)
            p = torch.einsum('ni,nij->nj', t_pow, p_mat)  # (batch, dim)
            results.append(p)
        return (*results,)


class HermiteTrajectory(Trajectory):
    def __init__(self,
                 times: Float[Tensor, "n"],
                 positions: Float[Tensor, "n 3"],
                 quats: Float[Tensor, "n 4"]
                ):
        super().__init__()
        self.model = HermiteSpline(
            times,
            torch.concat((positions, quats), dim=1)
        )

    def forward(self, t):
        t = t.flatten()
        y = self.model.forward(t, 0)
        pos = y[:, :3]
        quat = y[:, 3:] / torch.norm(y[:, 3:], dim=1, keepdim=True)
        # return pos
        # return quat
        return pos, quat

    def forward_grad(self, t):
        t = t.flatten()
        y, dydt, dydt2 = self.model.forward(t, 2)
        p, dpdt, dpdt2 = y[:, :3], dydt[:, :3], dydt2[:, :3]
        q, dqdt = y[:, 3:], dydt[:, 3:]
        q_invnorm = 1.0 / torch.norm(q, dim=1, keepdim=True)
        q = q * q_invnorm
        dqdt = (((torch.eye(4, device=device).unsqueeze(0) - torch.einsum('ni,nj->nij', q, q)) \
                 * q_invnorm.unsqueeze(-1)) @ dqdt.unsqueeze(2)).squeeze(2)
        # return p, dpdt, dpdt2
        # return q, dqdt
        return p, dpdt, dpdt2, q, dqdt


def check_close(msg, a, b):
    err = abs(b-a).mean().item()
    mag = abs(a).mean().item()
    print(msg, "{:.4f}".format(err/(mag+1e-12)), '\t', err, mag)

def test_pos_grad(traj, t):
    h = 0.01
    x0 = traj(t)
    xt0 = (traj(t+h)-traj(t-h)) / (2.0*h)
    xtt0 = (traj(t+h)+traj(t-h)-2.0*x0) / (h*h)
    x2, xt2, xtt2 = traj.forward_grad(t)
    check_close("x1|x2:", x0, x2)
    check_close("xt0|xt1:", xt0, xt2)
    check_close("xtt0|xtt2:", xtt0, xtt2)
    __import__('sys').exit(0)

def test_quat_grad(traj, t):
    h = 0.01
    x0 = traj(t)
    xt0 = (traj(t+h)-traj(t-h)) / (2.0*h)
    xtt0 = (traj(t+h)+traj(t-h)-2.0*x0) / (h*h)
    x1, xt1 = traj.forward_grad(t)
    check_close("x0|x1:", x0, x1)
    check_close("xt0|xt1:", xt0, xt1)
    __import__('sys').exit(0)

if __name__ == "__main__":
    torch.manual_seed(42)
    times = torch.tensor([-0.1, 1, 3, 8])
    pnts2 = torch.tensor([[-1, -1], [2, -2], [3, 3], [0, 2]])
    pnts3 = torch.tensor([[-1, -1, -1], [2, -2, 2], [3, 3, -3], [0, 2, 1]])
    quats = torch.randn(4, 4)
    quats /= torch.norm(quats, dim=1)
    t = torch.linspace(0, 8, 100).to(device)

    # traj = HermiteTrajectory(times, pnts3, quats)
    # test_pos_grad(traj, t)
    # test_quat_grad(traj, t)

    traj = HermiteSpline(times, pnts2)
    p = traj(t, 2)[0]

    import matplotlib.pyplot as plt
    p = p.detach().cpu().numpy()
    plt.plot(p[:, 0], p[:, 1], '.-')
    p = traj.c0.cpu().numpy()
    plt.plot(p[:, 0], p[:, 1], 'o')
    plt.show()
