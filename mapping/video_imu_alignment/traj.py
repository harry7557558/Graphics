import torch

class Trajectory(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t):
        pos, quat = None, None
        return pos, quat

    def forward_grad(self, t):
        p, dpdt, dpdt2, q, dqdt = [None] * 5
        return p, dpdt, dpdt2, q, dqdt

