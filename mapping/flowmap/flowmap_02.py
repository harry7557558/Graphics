import numpy as np
import torch
import cv2 as cv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from typing import List, Dict, Tuple, Union, Optional, Literal
from jaxtyping import Int, Float
from torch import Tensor
from torchvision.utils import flow_to_image


device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device:", device)



# Global Map

IMG_SHAPE = None
all_images = []  # type: List[np.ndarray]
all_flows = {}  # type: Dict[Tuple[Int, Int], Tuple[Tensor, Tensor]]


# Optical Flow

import torchvision.transforms
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


@torch.no_grad()
def preprocess_optical_flow(img: Union[np.ndarray, Tensor]) -> Tensor:
    if isinstance(img, np.ndarray):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return transform(img).unsqueeze(0).to(device)
    elif isinstance(img, Tensor):
        transform = torchvision.transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return transform(img.squeeze()).unsqueeze(0).to(device)


@torch.no_grad()
def compute_optical_flow(
        img1: Tensor, img2: Tensor, squeeze=True, model=[None]
    ) -> Tuple[Tensor, Tensor]:

    if model[0] is None:
        model[0] = raft_large(
            weights=Raft_Large_Weights.C_T_SKHT_V2,
            progress=True).to(device)
        model[0].eval()

    f12 = model[0](img1, img2)[-1]

    batch, _, height, width = f12.shape
    y_coords = torch.arange(height).float().to(device) + 0.5
    x_coords = torch.arange(width).float().to(device) + 0.5
    y_coords = y_coords.view(1, 1, height, 1).expand(batch, 1, height, width)
    x_coords = x_coords.view(1, 1, 1, width).expand(batch, 1, height, width)
    grid = torch.cat([x_coords, y_coords], dim=1)

    g12 = grid + f12
    x, y = g12[:, 0], g12[:, 1]
    pad = 1
    within = (x >= pad) & (x < width-pad) & (y >= pad) & (y < height-pad)

    g12[:, 0] = 2 * g12[:, 0] / width - 1
    g12[:, 1] = 2 * g12[:, 1] / height - 1
    g12 = g12.permute(0, 2, 3, 1)  # (batch, h, w, 2)

    warped_img2 = torch.nn.functional.grid_sample(
        img2, g12, mode='bilinear', padding_mode='zeros', align_corners=False)

    diff = torch.abs(img1 - warped_img2)
    diff = diff.mean(dim=1, keepdim=True)

    diff_flattened = torch.masked_select(diff, within)
    threshold = torch.quantile(diff_flattened, 0.85)
    mask = (diff < threshold) & within.unsqueeze(1)

    if squeeze:
        return f12.squeeze(), mask.squeeze()
    return f12, mask


@torch.no_grad()
def compute_optical_flow_adj(
        images: List[np.ndarray]
        ) -> Dict[Tuple[Int, Int], Tuple[Tensor, Tensor]]:
    images = [preprocess_optical_flow(img) for img in images]
    res = {}
    for i0 in range(len(images)-1):
        i1 = i0 + 1
        res[(i0, i1)] = compute_optical_flow(images[i0], images[i1])
        res[(i1, i0)] = compute_optical_flow(images[i1], images[i0])
    return res


@torch.no_grad()
def compute_optical_flow_all(
        images: List[np.ndarray]
        ) -> Dict[Tuple[Int, Int], Tuple[Tensor, Tensor]]:
    images = [preprocess_optical_flow(img) for img in images]
    res = {}
    for i in range(len(images)):
        for j in range(len(images)):
            if i == j:
                continue
            res[(i, j)] = compute_optical_flow(images[i], images[j])
    return res


@torch.no_grad()
def plot_optical_flows(
        flows: Dict[Tuple[int, int], Tuple[Tensor, Tensor]],
        mask=False
    ):

    n = int(np.ceil(np.sqrt(len(flows))))
    m = int(np.ceil(len(flows) / n))
    fig, axes = plt.subplots(m, n, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, key in enumerate(sorted(flows.keys())):
        if mask:
            tensor = flows[key][1].squeeze()
            axes[i].imshow(tensor.cpu().numpy(),
                        cmap='binary', interpolation='none')
        else:
            tensor = flow_to_image(flows[key][0].squeeze())
            axes[i].imshow(tensor.permute(1, 2, 0).cpu().numpy(),
                           interpolation='none')
        axes[i].set_title(str(key))
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    for j in range(i+1, n*m):
        fig.delaxes(axes[j])
    
    plt.savefig("plot_optical_flows.pdf")
    plt.show()


# Model


class Model(torch.nn.Module):

    def __init__(self,
                 images: List[np.ndarray]
            ) -> None:
        super().__init__()

        # depth model
        if False:
            self.depth_model_raw = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True)
            self.depth_model = lambda x: 1e3 / (self.depth_model_raw(x) + 0.1)
            self.depth_preprocess = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

            def depth_estimator(img: np.ndarray):
                transformed = self.depth_preprocess(img).to(device)
                output = self.depth_model(transformed)  # type: Tensor
                depth = torch.nn.functional.interpolate(
                    output.unsqueeze(0),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                return depth

            self.depth_estimator = depth_estimator
        else:
            def depth_preprocess(x):
                x = torch.from_numpy(x / 255).float().to(device)
                return x.permute((2, 0, 1)).unsqueeze(0)
            self.depth_preprocess = depth_preprocess
        # self.depth_preprocess = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

        # images
        self.n_images = len(images)
        assert len(images) > 0, "Empty image list"
        self.raw_width = images[0].shape[1]
        self.raw_height = images[0].shape[0]
        assert self.raw_width > 1 and self.raw_height > 1, "Empty image"
        for img in images:
            assert img.shape == images[0].shape, "Different image shapes"
        self.depths = None
        self.points = None

        self.images_transformed = torch.concat([
            self.depth_preprocess(img).to(device)
            for img in images
        ], dim=0)
        self.height, self.width = self.images_transformed.shape[-2:]
        assert self.width > 0 and self.height > 0, "Empty image"

        # camera
        f = 0.6 * np.sqrt(self.width * self.height)
        self.camera_f = torch.nn.Parameter(Tensor([f, f]).float())  # fx, fy
        self.camera_c = torch.nn.Parameter(Tensor([0.5*self.width, 0.5*self.height]).float())  # cx, cy
        self.camera_dist = torch.nn.Parameter(torch.zeros(4).float())  # k1, k2, p1, p2

    def compute_transform_and_triangulate(self, points1, points2, mask):
        fx, fy = self.camera_f
        cx, cy = self.camera_c
        assert (self.camera_dist == 0).all(), "Camera distortion is not supported"

        # Reshape and convert to UV coordinates
        h, w = points1.shape[1], points1.shape[2]
        u1, v1 = points1[0].flatten(), points1[1].flatten()
        u2, v2 = points2[0].flatten(), points2[1].flatten()
        print(u1.shape, v1.shape, u2.shape, v2.shape)

        # Normalize coordinates
        x1 = (u1 - cx) / fx
        y1 = (v1 - cy) / fy
        x2 = (u2 - cx) / fx
        y2 = (v2 - cy) / fy
        print(self.width, self.height)
        print(fx.item(), fy.item())
        print('x1', x1.mean().item(), x1.std().item())
        print('y1', y1.mean().item(), y1.std().item())

        # Construct coefficient matrix for the 8-point algorithm
        # A = torch.stack([
        #     x1 * x2, x1 * y2, x1, 
        #     y1 * x2, y1 * y2, y1, 
        #     x2, y2, torch.ones_like(x1)
        # ], dim=-1).view(-1, 9)
        A = torch.stack([
            x2 * x1, x2 * y1, x2,
            y2 * x1, y2 * y1, y2,
            x1, y1, torch.ones_like(x1)
        ], dim=-1)
        A = A[torch.where(mask.flatten())]
        print(A.shape)

        # Solve for the essential matrix using SVD
        # findings:
        #  - eigh is much faster than svd, svd gives MLE for large A
        #  - eigh(A^T A) is numerically inaccurate
        #  - numerical inaccuracy of A^T A affects eigenvalues more than eigenvectors
        #  - eigenvectors for larger eigenvalues have higher accuracy
        if 0:
            U, S, Vh = torch.linalg.svd(A)
            E = Vh[-1].view(3, 3)
        elif 0:
            # ATA = A.T.double() @ A.double()
            ATA = (A.T @ A).double()
            # ATA = (A.T @ A)
            S, V = torch.linalg.eigh(ATA)
            S, V = S.float(), V.float()
            S, Vh = torch.sqrt(S), V.T
            E = Vh[0].view(3, 3)
            a = torch.zeros(9, device=device).float()
            a[0] = 1.0
        else:
            ATA = (A.T @ A)
            invATA = torch.linalg.inv(ATA)
            S, V = torch.linalg.eigh(invATA)
            # S = 1.0 / torch.sqrt(S)
            S = torch.linalg.norm(A @ V, axis=0)
            # print(torch.dist(V.T @ V, torch.eye(9).to(V)))
            Vh = V.T
            E = Vh[-1].view(3, 3)
            a = torch.zeros(9, device=device).float()
            a[-1] = 1.0
        print("S:", S)
        print("E:", E, sep='\n')
        print("proj err:", torch.linalg.norm(A @ E.flatten()).item())
        for s, e in zip(S, Vh):
            print(s.item(), torch.linalg.norm(A @ e).item())
        # U, S, Vh = torch.linalg.svd(E)
        # print("S:", S)

        nc = 10  # number of constraints

        # https://en.wikipedia.org/wiki/Essential_matrix#Properties
        I3 = torch.eye(3, device=device)
        delta4 = torch.zeros((3, 3, 3, 3), device=device)
        for i in range(3):
            for j in range(3):
                delta4[i, j, i, j] = 1.0
        def F(E):
            c1 = torch.linalg.det(E)
            EET = E @ E.T
            c2 = 2.0 * EET @ E - torch.trace(EET) * E
            f = torch.concat((c1.flatten(), 0.0*c2.flatten()))
            return f
        def F_with_jac(E):
            c1 = torch.linalg.det(E)
            g_c1 = c1 * torch.linalg.inv(E).T
            EET = E @ E.T
            g_EET = torch.einsum('ikab,jk->ijab', delta4, E) + \
                torch.einsum('jkab,ik->ijab', delta4, E)
            trEET = torch.trace(EET)
            g_trEET = torch.einsum('ij,ijab->ab', I3, g_EET)
            EETE = EET @ E
            g_EETE = torch.einsum('ikab,kj->ijab', g_EET, E) + \
                torch.einsum('kjab,ik->ijab', delta4, EET)
            trEETE = trEET * E
            g_trEETE = torch.einsum('ab,ij->ijab', g_trEET, E) + \
                delta4 * trEET
            c2 = 2.0 * EETE - trEETE
            g_c2 = 2.0 * g_EETE - g_trEETE
            f = torch.concat((c1.flatten(), 0.0*c2.flatten()))
            g = torch.concat((g_c1.reshape((-1, 3, 3)), 0.0*g_c2.reshape((-1, 3, 3))))
            return f, g
        print("cons err:", F(E).norm().item(), F(E))

        # TODO: iteratively find better E subject to essential matrix constraints
        # Introduce a, E_i = V[i], E = \sum_i a_i E_i
        # Minimize L = || \sum_i a_i S_i ||^2
        # s.t. g1 = || a ||^2 - 1 = 0, g2 = F(\sum_i a_i E_i) = 0
        # F(E): https://en.wikipedia.org/wiki/Essential_matrix#Properties
        # d L / d a_i = 2 S_i (\sum_j a_j S_j)
        # d^2 L / d a_i a_j = 2 S_i S_j
        # d g1 / d a = 2 a
        # d g2 / d a_i = E_i \cdot \nabla F(\sum_j a_j E_j)
        # H = [ d^2 L / d a^2, (d g / d a)^T; d g / d a, 0 ]
        # G = [d L / d a + (d g / d a)^T \lambda; g(a)]
        # \Delta [a, \lambda] = - (H + \epsilon I) \ G

        l_scale = 1.0 / (S*S).sum()
        g1_scale = 0.01
        lambdam = torch.zeros((nc+1,), device=device)
        for i in range(20):
            E = (V @ a).reshape((3, 3))
            F_E1, dFdE = F_with_jac(E)
            g1 = g1_scale * ((a*a).sum() - 1.0)
            g2 = F_E1
            g = torch.concat((g1.flatten(), g2.flatten()))
            dLda = l_scale * 2.0 * S * (a*S).sum()
            d2Lda2 = l_scale * 2.0 * torch.tensordot(S, S, dims=0)
            dg1da = g1_scale * 2.0 * a
            # print(dFdE.reshape((nc, 9)))
            dg2da = dFdE.reshape((nc, 9)) @ Vh
            dgda = torch.concat((dg1da.view(-1, 9), dg2da.view(-1, 9)))
            H = torch.zeros((9+nc+1, 9+nc+1))
            H[:9, :9] = d2Lda2
            H[:9, 9:] = dgda.T
            H[9:, :9] = dgda
            G = torch.zeros((9+nc+1,))
            G[:9] = dLda + dgda.T @ lambdam
            G[9:] = g
            # print(torch.linalg.eigvalsh(H.double()))
            # X = -torch.linalg.solve(H, G)

            lr = 1.0 / torch.linalg.norm(H)
            # da = dLda
            # da = 2.0 * g @ dgda
            da = dLda + dgda.T @ lambdam
            # print(da)
            dl = g

            delta_l = dl / torch.sqrt((dgda**2).sum()/(nc+1))
            delta_a = (da-dgda.T@delta_l) / torch.sqrt((d2Lda2**2).sum()/9)
            delta_l = dl
            delta_a = da

            a = a - lr * delta_a
            # a = a / torch.linalg.norm(a)
            lambdam = lambdam - lr * delta_l
            E = (V @ a).reshape((3, 3))
            print("cons err:", F(E).norm().item(),
                  "|a|:", torch.linalg.norm(a).item(),
                  "proj err:", torch.linalg.norm(A @ E.flatten()).item())
        print(lambdam)
        print("a:", torch.linalg.norm(a).item(), a)
        print("proj err:", torch.linalg.norm(A @ E.flatten()).item())
        print("cons err:", F(E).norm().item(), F(E))


        E1, _ = cv.findEssentialMat(
            points1.reshape(2, -1).T.detach().cpu().numpy(),
            points2.reshape(2, -1).T.detach().cpu().numpy(),
            torch.sqrt(fx*fy).item(),
            self.camera_c.detach().cpu().numpy(),
            method=cv.FM_8POINT)
        E = torch.tensor(E1).float().to(device)
        print("E:", E, sep='\n')
        print("proj err:", torch.linalg.norm(A @ E.flatten()).item())
        print("cons err:", F(E).norm().item(), F(E))
        __import__("sys").exit(0)

        # `EMEstimatorCallback` in
        # https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/five-point.cpp

        # Enforce the essential matrix constraint: E = U * diag(1, 1, 0) * V^T
        U, S, Vh = torch.linalg.svd(E)
        print("S:", S)
        # __import__('sys').exit(0)
        S = torch.tensor([1, 1, 0], dtype=E.dtype, device=E.device)
        E = U @ torch.diag(S) @ Vh
        print("E:", E, sep='\n')
        if torch.det(U @ Vh) < 0:
            U[:, -1] *= -1
        print("U:", U, sep='\n')
        print("Vh:", Vh, sep='\n')
        print("U Vh:", U @ Vh, sep='\n')

        # Decompose E to get R and t
        Rz = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                        dtype=E.dtype, device=E.device)
        R1 = U @ Rz @ Vh
        R2 = U @ Rz.T @ Vh
        t = U[:, 2]
        print("R1:", R1, sep='\n')
        print("R2:", R2, sep='\n')
        print("t:", t)

        # __import__('sys').exit(0)

        # Check positive depth to choose the correct R and t
        def positive_depth_count(R, t):
            # R, t = R.T, -R.T @ t
            points3d = self.triangulate_points(R, t, x1, y1, x2, y2)
            depths1 = points3d[:, 2]
            # return (depths1 > 0).sum().item()
            depths2 = (R[2:, :] @ points3d.T).squeeze() + t[2]
            return (depths1 > 0).sum().item() + (depths2 > 0).sum().item()

        counts = [
            positive_depth_count(R1, t),
            positive_depth_count(R1, -t),
            positive_depth_count(R2, t),
            positive_depth_count(R2, -t)
        ]
        print(counts, 2*w*h)
        max_count_idx = torch.argmax(torch.tensor(counts))
        R, t = [(R1, t), (R1, -t), (R2, t), (R2, -t)][max_count_idx]

        # R = torch.tensor([
        #     [0.9970402,  -0.05024029,  0.05819201],
        #     [0.05053127,  0.998716,   -0.00353295],
        #     [-0.05794021,  0.00646281,  0.9982989]
        # ]).float().to(device)
        # t = 4 * torch.tensor([
        #     -0.04350451, 0.01415826, -0.02344608
        # ]).float().to(device)

        # Triangulate points
        points3d = self.triangulate_points(R, t, x1, y1, x2, y2).view(h, w, 3)

        # Construct the transformation matrix
        transform = torch.eye(4, dtype=R.dtype, device=R.device)
        transform[:3, :3] = R
        transform[:3, 3] = t

        print(transform)
        print(points3d.shape)
        points_plot = points3d.reshape((-1, 3))[torch.where(mask.flatten())]
        plot_3d_points([points3d, points_plot])

        return transform, points3d

    @staticmethod
    def triangulate_points(R, t, x1, y1, x2, y2):
        num_points = x1.numel()

        # z (x1, y1) ~ (x, y)
        # z' (x2, y2) ~ (x', y')

        A = torch.zeros((num_points, 4, 3), dtype=R.dtype, device=R.device)
        A[:, 0, 0] = 1.0
        A[:, 0, 2] = -x1
        A[:, 1, 1] = 1.0
        A[:, 1, 2] = -y1
        A[:, 2] = x2.unsqueeze(-1) * R[2] - R[0]
        A[:, 3] = y2.unsqueeze(-1) * R[2] - R[1]

        b = torch.zeros((num_points, 4), dtype=R.dtype, device=R.device)
        b[:, 2] = t[0] - x2 * t[2]
        b[:, 3] = t[1] - y2 * t[2]

        At = A.transpose(-1, -2)
        AtA = torch.matmul(At, A)
        Atb = torch.matmul(At, b.unsqueeze(-1))
        x = torch.linalg.solve(AtA, Atb)
        return x.squeeze()

    def project_to_3d(self, depth: Float[Tensor, "batch h w"]) -> Float[Tensor, "batch h w 3"]:
        batch_size, height, width = depth.shape
        fx, fy = self.camera_f
        cx, cy = self.camera_c
        k1, k2, p1, p2 = self.camera_dist

        y, x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij")
        x = x.expand(batch_size, -1, -1).float() + 0.5
        y = y.expand(batch_size, -1, -1).float() + 0.5

        x = (x - cx) / fx
        y = (y - cy) / fy

        r2 = x**2 + y**2
        dist = 1 + k1 * r2 + k2 * r2**2
        x, y = (
            x * dist + 2 * p1 * x * y + p2 * (r2 + 2 * x**2),
            y * dist + 2 * p2 * x * y + p1 * (r2 + 2 * y**2)
        )

        return torch.stack((x*depth, y*depth, depth), dim=-1)

    def generate_grid(self):
        y, x = torch.meshgrid(
            torch.arange(self.height, device=device) + 0.5,
            torch.arange(self.width, device=device) + 0.5,
            indexing="ij")
        return torch.stack([x, y], dim=0)

    def procrustes_transform(
            self,
            points1: Float[Tensor, "h w 3"],
            points2: Float[Tensor, "h w 3"],
            flow12: Float[Tensor, "2 h w"],
            mask: Float[Tensor, "h w"]
            ) -> Float[Tensor, "4 4"]:
        _, h, w = flow12.shape

        grid1 = self.generate_grid()
        grid2 = grid1 + flow12
        grid1 = torch.masked_select(grid1, mask).view(2, -1)
        grid2 = torch.masked_select(grid2, mask).view(2, -1)

        def sample_grid(points, grid):
            grid[0, :] = 2.0 * (grid[0, :] / (w - 1)) - 1.0
            grid[1, :] = 2.0 * (grid[1, :] / (h - 1)) - 1.0
            points = torch.nn.functional.grid_sample(
                points.permute(2, 0, 1).unsqueeze(0),  # (1, 3, h, w)
                grid.T.unsqueeze(0).unsqueeze(0),  # (1, 1, n, 2)
                mode="bilinear", padding_mode="border",
                align_corners=False
            )
            return points.squeeze(0).permute(1, 2, 0).squeeze(0)  # (n, 3)

        points1 = sample_grid(points1, grid1)
        points2 = sample_grid(points2, grid2)
        # points = [points1, points2, torch.concatenate((points1, points2), dim=0)]
        # plot_3d_points(points, 12345)

        centroid1 = torch.mean(points1, dim=0)
        centroid2 = torch.mean(points2, dim=0)
        centered1 = points1 - centroid1
        centered2 = points2 - centroid2
        cov = centered2.transpose(-1, -2) @ centered1
        U, _, Vh = torch.linalg.svd(cov)
        S = torch.eye(3, device=device).float()
        S[2, 2] = (U.det() * Vh.det()).sign()
        R = U @ S @ Vh
        t = centroid2 - R @ centroid1

        loss0 = ((points1-points2)**2).mean()
        residual = (points1 @ R.T + t) - points2
        loss = (residual**2).mean()
        assert loss <= loss0

        T = torch.eye(4, device=device).float()
        T[:3, :3] = R
        T[:3, 3] = t
        return T, loss
    
    def forward(
            self,
            flows: Dict[Tuple[Int, Int], Tuple[Tensor, Tensor]],
            returnv: Literal["depths", "points", "loss", "poses"] = "loss"
        ) -> Float:

        grid1 = self.generate_grid()
        grid2 = grid1 + flows[(0, 1)][0]
        mask = flows[(0, 1)][1]
        self.compute_transform_and_triangulate(grid1, grid2, mask)

        __import__('sys').exit(0)

        self.depths = torch.concat([
            self.depth_model(self.images_transformed[i:i+1])
            for i in range(self.n_images)
        ], dim=0)
        self.depths = self.depths / self.depths.mean()
        if returnv == "depths":
            return self.depths
        
        self.points = self.project_to_3d(self.depths)
        if returnv == "points":
            return self.points

        if returnv == "loss":
            total_loss = 0.0
            for (i, j), flow in flows.items():
                T, loss = self.procrustes_transform(
                    self.points[i], self.points[j], *flow)
                total_loss = total_loss + loss
            return total_loss / len(flows)

        if returnv == "poses":
            poses = torch.zeros(len(flows), 4, 4, device=device).float()
            poses[0] = torch.eye(4, device=device).float()
            for i in range(1, self.n_images):
                T, loss = self.procrustes_transform(
                    self.points[i-1], self.points[i], *flows[(i-1, i)])
                poses[i] = T @ poses[i-1]
            return poses


@torch.no_grad()
def plot_depths(depths: List[Tensor]):

    n = int(np.ceil(np.sqrt(len(depths))))
    m = int(np.ceil(len(depths) / n))
    fig, axes = plt.subplots(m, n, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(len(depths)):
        tensor = depths[i].squeeze()
        axes[i].imshow(tensor.cpu().numpy(),
                       cmap='gnuplot', interpolation='none')
        mean, std = tensor.mean().item(), tensor.std().item()
        axes[i].set_title("{:.1f} ± {:.1f}".format(mean, std))
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    for j in range(i+1, n*m):
        fig.delaxes(axes[j])
    
    plt.savefig("plot_depths.pdf")
    plt.show()


@torch.no_grad()
def plot_3d_points(points: List[Float[Tensor, "batch h w 3"]], fn=4567):

    n = int(np.ceil(np.sqrt(len(points))))
    m = int(np.ceil(len(points) / n))
    fig, axes = plt.subplots(m, n, figsize=(12, 12),
                             subplot_kw=dict(projection='3d'))
    axes = axes.flatten()

    fn = int(fn * 8/len(points) + 0.5)
    for i in range(len(points)):
        tensor = points[i].reshape(-1, 3).cpu().numpy()
        tensor = tensor[(np.arange(fn) * len(tensor) / fn).astype(np.int32)]
        axes[i].scatter(tensor[:, 0], tensor[:, 2], tensor[:, 1],
                        s=0.5, c=tensor[:, 2], cmap='gnuplot')
        setup_3d_axis(axes[i])

    for j in range(i+1, n*m):
        fig.delaxes(axes[j])
    
    plt.savefig("plot_3d_points.pdf")
    plt.show()


# Training

def optimize_model(model, flows, num_epochs=200):

    depth_params = list(model.depth_model_raw.parameters())

    # optimizer_depth = torch.optim.Adam(depth_params, lr=1e-4)
    optimizer_camera_f = torch.optim.Adam([model.camera_f], lr=1e-2)
    optimizer_camera_c = torch.optim.Adam([model.camera_c], lr=1e-2)
    optimizer_camera_dist = torch.optim.Adam([model.camera_dist], lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        # optimizer_depth.zero_grad()
        optimizer_camera_f.zero_grad()
        optimizer_camera_c.zero_grad()
        optimizer_camera_dist.zero_grad()

        loss = model(flows)**0.5

        loss.backward()
        # optimizer_depth.step()
        optimizer_camera_f.step()
        optimizer_camera_c.step()
        # optimizer_camera_dist.step()

        with torch.no_grad():
            camera = torch.concatenate([model.camera_f, model.camera_c])
            dist = model.camera_dist
            camera = camera.detach().cpu().numpy()
            dist = dist.detach().cpu().numpy()
            camera = np.round(camera, 1)
            dist = np.round(dist, 4)

        if epoch == 0 or (epoch+1) % 10 == 0:
            print(f'{epoch+1}/{num_epochs}',
                  f'loss {loss.item():.4f}',
                  f'camera {camera} {dist}',
                  sep='  ')
    
    return model


# Frame Selection

def add_frame(frame):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    all_images.append(frame)




# Plotting and Exporting

@torch.no_grad()
def plot_cameras(model: Model, flows, sc=0.05):
    points = np.array([
        (0, 0, 0),
        (0, 0, 1),
        (model.width, 0, 1),
        (model.width, model.height, 1),
        (0, model.height, 1)
    ]).T * sc
    fx, fy = model.camera_f.cpu().numpy()
    cx, cy = model.camera_c.cpu().numpy()
    K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    points = np.linalg.inv(K) @ points
    poses = model(flows, returnv="poses").cpu().numpy()

    plt.figure(figsize=(8, 8))
    ax = plt.axes(projection='3d')
    ax.computed_zorder = False
    for T in poses:
        R, t = T[:3, :3], T[:3, 3:]
        points_3d = R @ points - t
        idx = [0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1]
        vertices = points_3d[:,idx]
        ax.plot(vertices[0], vertices[2], vertices[1], '-')
    setup_3d_axis(ax)
    plt.savefig("plot_cameras.pdf")
    plt.show()

def setup_3d_axis(ax):
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
    plot_radius = 0.3*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    # https://stackoverflow.com/a/44002650
    ax.xaxis.set_pane_color((1, 1, 1, 0))
    ax.yaxis.set_pane_color((1, 1, 1, 0))
    ax.zaxis.set_pane_color((1, 1, 1, 0))
    ax.xaxis._axinfo["grid"]['color'] =  (.5, .5, .5, .0)
    ax.yaxis._axinfo["grid"]['color'] =  (.5, .5, .5, .0)
    ax.zaxis._axinfo["grid"]['color'] =  (.5, .5, .5, .0)
    # opencv convention
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.invert_zaxis()



if __name__ == "__main__":

    # Select video

    import os, sys
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)
    import sfm_calibrated.img.videos
    video_filename = sfm_calibrated.img.videos.videos[4]
    # video_filename = "/home/harry7557558/adr.mp4"
    cap = cv.VideoCapture(video_filename)
    assert cap.isOpened(), "Fail to open video"

    # Load frames

    target_img_size = 720

    frame_i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if IMG_SHAPE is None:
            img_size = np.prod(frame.shape[:2])**0.5
            sc = target_img_size / img_size
            IMG_SHAPE = sc * np.array([frame.shape[1], frame.shape[0]])
            IMG_SHAPE = 8 * np.round(IMG_SHAPE/8).astype(np.int32)
            print("Image shape:", IMG_SHAPE)

        frame = cv.resize(frame, IMG_SHAPE)

        cv.imshow('Frame', frame)
        if cv.waitKey(25) & 0xFF == ord('q'):
            break

        if frame_i % 8 == 0:
            add_frame(frame)
        frame_i += 1

        if len(all_images) >= 2:
            break

    cap.release()
    cv.destroyAllWindows()

    print(f"Selected {len(all_images)} frames")
    print()

    # Create model

    print("Create model...")
    model = Model(all_images).to(device)
    print("Model created.")
    print()

    print("Compute optical flow...")
    shape = np.array([model.width, model.height])
    print("Image shape:", shape)
    all_images_reshaped = [cv.resize(img, shape) for img in all_images]
    # all_flows = compute_optical_flow_all(all_images_reshaped)
    all_flows = compute_optical_flow_adj(all_images_reshaped)
    print("Done.")
    print()
    # plot_optical_flows(all_flows)

    with torch.no_grad():
        output = model(all_flows, returnv="points")
    # plot_depths(output)
    # plot_3d_points(output)

    print("Training model...")
    model = optimize_model(model, all_flows)
    print("Training done.")

    plot_cameras(model, all_flows)

    with torch.no_grad():
        output = model(all_flows, returnv="points")
    plot_3d_points(output)

    __import__('sys').exit(0)
