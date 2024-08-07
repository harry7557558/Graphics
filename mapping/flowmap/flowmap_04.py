import numpy as np
import torch
import torch.nn as nn
import cv2 as cv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from typing import List, Dict, Tuple, Union, Optional, Literal
from jaxtyping import Int, Float
from torch import Tensor
from torchvision.utils import flow_to_image


device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device:", device)



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
        img1: Tensor, img2: Tensor, model=[None]
    ) -> Tensor:

    if model[0] is None:
        model[0] = raft_large(
            weights=Raft_Large_Weights.C_T_SKHT_V2,
            progress=True).to(device)
        model[0].eval()

    f12 = model[0](img1, img2)[-1]

    return f12.squeeze()

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
        flows: Dict[Tuple[int, int], Float[Tensor, "2 h w"]],
    ):

    n = int(np.ceil(np.sqrt(len(flows))))
    m = int(np.ceil(len(flows) / n))
    n, m = m, n
    fig, axes = plt.subplots(m, n, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, key in enumerate(sorted(flows.keys())):
        tensor = flow_to_image(flows[key].squeeze())
        axes[i].imshow(tensor.permute(1, 2, 0).cpu().numpy(),
                        interpolation='none')
        axes[i].set_title(str(key))
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    for j in range(i+1, n*m):
        fig.delaxes(axes[j])
    
    plt.savefig("plot_optical_flows.pdf")
    plt.show()

@torch.no_grad()
def plot_weights(
        weights: Union[
            Dict[Tuple[int, int], Float[Tensor, "h w"]],
            List[Tensor]
        ]
    ):

    n = int(np.ceil(np.sqrt(len(weights))))
    m = int(np.ceil(len(weights) / n))
    n, m = m, n
    fig, axes = plt.subplots(m, n, figsize=(15, 15))
    axes = axes.flatten()
    
    if isinstance(weights, dict):
        for i, key in enumerate(sorted(weights.keys())):
            tensor = weights[key].squeeze()
            axes[i].imshow(tensor.cpu().numpy(),
                        cmap='gnuplot', interpolation='none')
            axes[i].set_title(str(key))
            axes[i].set_xticks([])
            axes[i].set_yticks([])
    else:
        for i in range(len(weights)):
            tensor = weights[i].squeeze()
            axes[i].imshow(tensor.cpu().numpy(),
                            cmap='gnuplot', interpolation='none')
            mean, std = tensor.mean().item(), tensor.std().item()
            axes[i].set_title("{:.1f} ± {:.1f}".format(mean, std))
            axes[i].set_xticks([])
            axes[i].set_yticks([])

    for j in range(i+1, n*m):
        fig.delaxes(axes[j])
    
    plt.savefig("plot_weights.pdf")
    plt.show()


# Model

class TransformerBlock(nn.Module):
    def __init__(self, n_embed, n_heads, mlp_ratio=2.):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_embed)
        self.attn = nn.MultiheadAttention(n_embed, n_heads)
        self.norm2 = nn.LayerNorm(n_embed)
        self.fc1 = nn.Linear(n_embed, int(n_embed*mlp_ratio))
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(int(n_embed*mlp_ratio), n_embed)

    def forward(self, x):
        x1 = self.norm1(x)
        x = x + self.attn(x1, x1, x1)[0]
        x = x + self.fc2(self.act(self.fc1(self.norm2(x))))
        return x

class Transformer(nn.Module):
    def __init__(
            self,
            n_in: int,
            n_out: int,
            n_embed: int,
            n_layers: int,
            n_heads: int,
            image_size: Tuple[int, int],
            patch_size: int
        ):
        super().__init__()
        
        self.n_in, self.n_out = n_in, n_out
        self.image_size = image_size
        self.patch_size = patch_size
        self.reduced_size = (image_size[0]//patch_size, image_size[1]//patch_size)
        self.proj = nn.Conv2d(n_in, n_embed, kernel_size=patch_size, stride=patch_size)
        n_patches = self.reduced_size[0] * self.reduced_size[1]
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, n_embed))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches+1, n_embed))
        
        self.transformer = nn.ModuleList([
            TransformerBlock(n_embed, n_heads)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, n_out*patch_size**2)

    def forward(self, x: Float[Tensor, "batch nin h w"]):
        x = self.proj(x)  # batch n_embed *reduced_size
        x = x.flatten(2).transpose(1, 2)  # batch n_patches n_embed

        cls_tokens = self.cls_token.expand(len(x), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed  # batch n_patches+1 n_embed

        for layer in self.transformer:
            x = layer(x)

        x = self.norm(x[:,1:])
        x = self.head(x)  # batch n_patches n_embed
        x = x.reshape((len(x), *self.reduced_size, self.n_out, self.patch_size, self.patch_size))
        x = x.permute(0, 3, 1, 4, 2, 5).reshape((len(x), self.n_out, *self.image_size))
        return x

class UNet(nn.Module):

    def __init__(self, nin: int, nout: int, nh: int, image_size: Tuple[int, int]):
        super().__init__()

        def conv3(nin, nout):
            return nn.Conv2d(nin, nout, 3, padding=1, padding_mode="reflect", bias=True)
        def dconv4(nin, nout):
            return nn.ConvTranspose2d(nin, nout, 4, 2, 1, bias=True)
        def conv1(nin, nout):
            return nn.Conv2d(nin, nout, 1, padding=0, bias=True)

        n0, n1, n2, n3, n4, ne = 2*nh, 3*nh, 4*nh, 6*nh, 8*nh, 12*nh*4
        self.convi = conv3(nin, n0)
        self.econv0a = conv3(n0, n0) # 1
        self.econv0b = conv3(n0, n0) # 1
        self.econv1a = conv3(n0, n1) # 1/2
        self.econv1b = conv3(n1, n1) # 1/2
        self.econv2a = conv3(n1, n2) # 1/4
        self.econv2b = conv3(n2, n2) # 1/4
        self.econv3a = conv3(n2, n3) # 1/8
        self.econv3b = conv3(n3, n3) # 1/8
        self.econv4a = conv3(n3, n4) # 1/16
        self.econv4b = conv3(n4, n4) # 1/16
        # self.bottleneck = torch.nn.ReLU()
        self.bottleneck = Transformer(n4, n4, ne, 6, 8, (image_size[0]//16, image_size[1]//16), 2)
        self.dconv3a = dconv4(n4, n3) # 1/16->1/8
        self.dconv3b = conv1(n3+n3, n3) # 1/8
        self.dconv2a = dconv4(n3, n2) # 1/8->1/4
        self.dconv2b = conv1(n2+n2, n2) # 1/4
        self.dconv1a = dconv4(n2, n1) # 1/4->1/2
        self.dconv1b = conv1(n1+n1, n1) # 1/2
        self.dconv0a = dconv4(n1, n0) # 1/2->1
        self.dconv0b = conv1(n0+n0, n0) # 1
        self.convo = conv3(n0+n0, nout)

    def forward(self, input):
        relu = torch.relu
        pool = lambda x: torch.max_pool2d(x, 2)
        concat = lambda *args: torch.concat((*args,), axis=1)
        ci = self.convi(input)
        e0 = self.econv0b(relu(self.econv0a(relu(ci)))) # 1
        e1 = self.econv1b(relu(self.econv1a(pool(e0)))) # 1/2
        e2 = self.econv2b(relu(self.econv2a(pool(e1)))) # 1/4
        e3 = self.econv3b(relu(self.econv3a(pool(e2)))) # 1/8
        e4 = self.econv4b(relu(self.econv4a(pool(e3)))) # 1/16
        t = self.bottleneck(e4)  # 1/16
        d3 = relu(self.dconv3a(t)) # 1/8
        d3 = relu(self.dconv3b(concat(d3, e3))) # 1/8
        d2 = relu(self.dconv2a(e3)) # 1/4
        d2 = relu(self.dconv2b(concat(d2, e2))) # 1/4
        d1 = relu(self.dconv1a(d2)) # 1/2
        d1 = relu(self.dconv1b(concat(d1, e1))) # 1/2
        d0 = relu(self.dconv0a(d1)) # 1
        do = relu(self.dconv0b(concat(d0, e0))) # 1
        return self.convo(concat(do,ci))


class DepthModel(nn.Module):

    def __init__(self, image_size: Tuple[int, int]):
        super().__init__()

        self.num_flows = 4
        self.n_in = 3+2*self.num_flows
        # self.n_in = 2*self.num_flows

        self.in_bn = nn.BatchNorm2d(self.n_in)
        self.main = UNet(self.n_in, 2, 8, image_size)
        # self.main = Transformer(self.n_in, 2, 512, 8, 8, image_size, 16)

        self.main_in = None

    def forward(
        self,
        images: List[Float[Tensor, "h w 3"]],
        flows: Dict[Tuple[Int, Int], Float[Tensor, "2 h w"]],
        batch_indices: Optional[List[Int]]=None
    ) -> List[Tuple[Float[Tensor, "h w"], Float[Tensor, "h w"]]]:

        if self.main_in is None:
            if not isinstance(images, Tensor):
                images = torch.stack((*images,))  # batch 3 h w

            flow_list = [[] for _ in range(len(images))]
            for (i, j), flow in flows.items():
                flow_list[i].append(flow)
                flow_list[j].append(flow)
            for i in range(len(flow_list)):
                while len(flow_list[i]) < self.num_flows:
                    flow_list[i].append(torch.zeros_like(flow))
                flow_list[i] = torch.concat(flow_list[i])
            flow_list = torch.stack(flow_list)
            self.main_in = torch.hstack((images, flow_list))  # batch nin h w
            # self.unet_in = flow_list

        if batch_indices is None:
            unet_out = self.main(self.in_bn(self.main_in))  # batch 1 h w
        else:
            unet_out = self.main(self.in_bn(self.main_in[batch_indices]))
        depth = torch.exp(unet_out[:, 0])  # batch h w
        weight = torch.softmax(unet_out[:, 1].reshape(len(depth),-1), dim=1).view(depth.shape)  # batch h w
        return depth, weight

class WeightModel(nn.Module):

    def __init__(self, image_size: Tuple[int, int]):
        super().__init__()

        self.in_bn = nn.BatchNorm2d(4)
        self.main = UNet(4, 2, 4, image_size)

        self.flow_idx_map = {}
        self.main_in = None

    def forward(
        self,
        flows: Dict[Tuple[Int, Int], Float[Tensor, "2 h w"]]
    ) -> Dict[Tuple[Int, Int], Float[Tensor, "h w"]]:

        if self.main_in is None:
            flow_list = []
            for (i, j) in flows.keys():
                if i > j:
                    continue
                self.flow_idx_map[(i, j)] = len(flow_list)
                flow_list.append(torch.vstack((flows[(i, j)], flows[(j, i)])))
            self.main_in = torch.stack(flow_list)  # batch 4 h w

        weight = self.main(self.in_bn(self.main_in)).squeeze()  # batch 2 h w
        weight = torch.softmax(weight.reshape(len(weight),2,-1), dim=2).view(weight.shape)  # batch h w

        res = {}
        for (i, j), idx in self.flow_idx_map.items():
            res[(i, j)] = weight[idx, 0]
            res[(j, i)] = weight[idx, 1]
        return res


class Model(nn.Module):

    def __init__(self,
                 images: List[np.ndarray]
            ) -> None:
        super().__init__()

        # images
        self.n_images = len(images)
        assert len(images) > 0, "Empty image list"
        self.raw_width = images[0].shape[1]
        self.raw_height = images[0].shape[0]
        assert self.raw_width > 1 and self.raw_height > 1, "Empty image"
        for img in images:
            assert img.shape == images[0].shape, "Different image shapes"
        self.raw_images = torch.tensor(np.array(images))
        self.raw_images = self.raw_images.float().permute(0,3,1,2) / 255.0

        block_size = 32
        self.width = int(round(self.raw_width/block_size)*block_size)
        self.height = int(round(self.raw_height/block_size)*block_size)
        assert self.width > 0 and self.height > 0, "Image too small"
        if self.width != self.raw_width or self.height != self.raw_height:
            self.images = nn.functional.interpolate(
                self.raw_images,
                size=(self.height, self.width),
                mode="bicubic",
                align_corners=False,
            )
        else:
            self.images = self.raw_images

        # depth
        self.depth_model = DepthModel((self.height, self.width))
        self.depths = None
        self.weights = None

        # flow - Dict[Tuple[Int, Int], Float[Tensor, "2 h w"]]
        self.weight_model = WeightModel((self.height, self.width))
        print("Compute optical flow...")
        # self.flows = compute_optical_flow_all(all_images)
        self.flows = compute_optical_flow_adj(self.images)

        # camera
        f = 0.75 * np.sqrt(self.width * self.height)
        self.camera_res = torch.tensor([self.width, self.height]).float()
        self.camera_f_rel = nn.Parameter(Tensor([f/self.width, f/self.height]).float())  # rel fx, fy
        self.camera_c_rel = nn.Parameter(Tensor([0.5, 0.5]).float())  # rel cx, cy
        self.camera_dist = nn.Parameter(torch.zeros(4).float())  # k1, k2, p1, p2

    @property
    def camera_f(self):
        self.camera_res = self.camera_res.to(self.camera_f_rel)
        return self.camera_f_rel * self.camera_res

    @property
    def camera_c(self):
        self.camera_res = self.camera_res.to(self.camera_c_rel)
        return self.camera_c_rel * self.camera_res

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

    def procrustes_transform(
            self,
            points1: Float[Tensor, "h w 3"],
            points2: Float[Tensor, "h w 3"],
            flow12: Float[Tensor, "2 h w"],
            weight: Float[Tensor, "h w"]
            ) -> Float[Tensor, "4 4"]:
        _, h, w = flow12.shape

        y, x = torch.meshgrid(
            torch.arange(h, device=device) + 0.5,
            torch.arange(w, device=device) + 0.5,
            indexing="ij")
        grid1 = torch.stack([x, y], dim=0)
        grid2 = grid1 + flow12

        def sample_grid(points, grid):
            grid[0, :] = 2.0 * (grid[0, :] / (w - 1)) - 1.0
            grid[1, :] = 2.0 * (grid[1, :] / (h - 1)) - 1.0
            grid = grid.reshape(2, -1).T
            points = nn.functional.grid_sample(
                points.permute(2, 0, 1).unsqueeze(0),  # (1, 3, h, w)
                grid.unsqueeze(0).unsqueeze(0),  # (1, 1, n, 2)
                mode="bilinear", padding_mode="border",
                align_corners=False
            )
            return points.squeeze(0).permute(1, 2, 0).squeeze(0)  # (n, 3)

        points1 = sample_grid(points1, grid1)
        points2 = sample_grid(points2, grid2)
        # points = [points1, points2, torch.concatenate((points1, points2), dim=0)]
        # plot_3d_points(points, 12345)

        weight = weight.reshape(-1, 1)
        assert abs(weight.sum()-1) < 1e-5

        centroid1 = (weight * points1).sum(dim=0)
        centroid2 = (weight * points2).sum(dim=0)
        centered1 = points1 - centroid1
        centered2 = points2 - centroid2
        cov = (centered2 * weight).transpose(-1, -2) @ centered1
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
            returnv: Literal["depths", "weights", "points", "loss", "poses"] = "loss",
            batch_indices: Optional[List[Int]]=None
        ) -> Float:

        self.images = self.images.to(device)
        if batch_indices is None:
            self.depths, self.weights = self.depth_model(self.images, self.flows)
        if returnv == "depths":
            return self.depths
        
        self.points = self.project_to_3d(self.depths / self.depths.mean())
        if returnv == "points":
            return self.points
        
        # self.weights = self.weight_model(self.flows)
        if returnv == "weights":
            return self.weights

        if returnv == "loss":
            total_loss = 0.0
            # rigid transform loss
            for (i, j), flow in self.flows.items():
                T, loss = self.procrustes_transform(
                    self.points[i], self.points[j],
                    flow,
                    self.weights[(i, j)] if isinstance(self.weights, dict) else self.weights[i]
                )
                total_loss = total_loss + loss / len(self.flows)
            # weight regularization
            for weight in (self.weights.values() if isinstance(self.weights, dict) else self.weights):
                total_loss = total_loss + 1e-5 * torch.var(weight*weight.numel()) / len(self.weights)
            return total_loss

        if returnv == "poses":
            poses = torch.zeros(len(self.flows), 4, 4, device=device).float()
            poses[0] = torch.eye(4, device=device).float()
            for i in range(1, self.n_images):
                T, loss = self.procrustes_transform(
                    self.points[i-1], self.points[i],
                    self.flows[(i-1, i)],
                    self.weights[(i-1, i)] if isinstance(self.weights, dict) else self.weights[i-1]
                )
                poses[i] = T @ poses[i-1]
                # poses[i] = torch.linalg.inv(T) @ poses[i-1]
            return poses


@torch.no_grad()
def plot_depths(depths: List[Tensor]):

    n = int(np.ceil(np.sqrt(len(depths))))
    m = int(np.ceil(len(depths) / n))
    n, m = m, n
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

def optimize_model_adam(model: Model, num_epochs=200):

    optimizer_depth = torch.optim.Adam(model.depth_model.parameters(), lr=1e-3)
    optimizer_weight = torch.optim.Adam(model.weight_model.parameters(), lr=1e-3)
    optimizer_camera_f = torch.optim.Adam([model.camera_f_rel], lr=1e-4)
    optimizer_camera_c = torch.optim.Adam([model.camera_c_rel], lr=1e-4)
    optimizer_camera_dist = torch.optim.Adam([model.camera_dist], lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        optimizer_depth.zero_grad()
        optimizer_weight.zero_grad()
        optimizer_camera_f.zero_grad()
        optimizer_camera_c.zero_grad()
        optimizer_camera_dist.zero_grad()

        loss = model()**0.5
        loss.backward()

        optimizer_depth.step()
        optimizer_weight.step()
        # optimizer_camera_f.step()
        # optimizer_camera_c.step()
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
                  f'loss {loss.item():.6f}',
                  f'camera {camera} {dist}',
                  sep='  ')
    
    return model

def optimize_model_lbfgs(model: Model, num_epochs=200):

    params = [
        *model.depth_model.parameters(),
        *model.weight_model.parameters(),
        # model.camera_f_rel, model.camera_c_rel,# model.camera_dist,
    ]
    optimizer = torch.optim.LBFGS(
        params, max_iter=num_epochs,
        line_search_fn="strong_wolfe",
        history_size=40
    )

    loss = 0.0
    nfev = 0
    def closure():
        nonlocal nfev, loss
        optimizer.zero_grad()
        loss = model()**0.5
        loss.backward()
        nfev += 1

        if nfev == 0 or (nfev+1) % 10 == 0:
            with torch.no_grad():
                camera = torch.concatenate([model.camera_f, model.camera_c])
                dist = model.camera_dist
                camera = camera.detach().cpu().numpy()
                dist = dist.detach().cpu().numpy()
                camera = np.round(camera, 1)
                dist = np.round(dist, 4)
                print(f'{nfev+1}/{num_epochs}',
                    f'loss {loss.item():.6f}',
                    f'camera {camera} {dist}',
                    sep='  ')

        return loss

    for epoch in range(1):
        optimizer.step(closure)

    return model

def optimize_model(model):
    return optimize_model_adam(model, 1000)
    model = optimize_model_adam(model, 200)
    model = optimize_model_lbfgs(model, 800)
    return model


# Plotting and Exporting

@torch.no_grad()
def plot_camera(model: Model, sc=0.05):
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
    poses = model(returnv="poses").cpu().numpy()

    plt.figure(figsize=(8, 8))
    ax = plt.axes(projection='3d')
    ax.computed_zorder = False
    for T in poses:
        R, t = T[:3, :3], T[:3, 3:]
        points_3d = R.T @ (points - t)
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
    video_filename = sfm_calibrated.img.videos.videos[0]
    # video_filename = "/home/harry7557558/adr.mp4"
    cap = cv.VideoCapture(video_filename)
    assert cap.isOpened(), "Fail to open video"

    # Load frames

    IMG_SHAPE = None
    target_img_size = 240

    images = []

    frame_i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        block_size = 32
        if IMG_SHAPE is None:
            img_size = np.prod(frame.shape[:2])**0.5
            sc = target_img_size / img_size
            IMG_SHAPE = sc * np.array([frame.shape[1], frame.shape[0]])
            IMG_SHAPE = block_size * np.round(IMG_SHAPE/block_size).astype(np.int32)
            print("Image shape:", IMG_SHAPE)

        frame = cv.resize(frame, IMG_SHAPE)

        cv.imshow('Frame', frame)
        if cv.waitKey(25) & 0xFF == ord('q'):
            break

        if frame_i % 8 == 0:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            images.append(frame)
        frame_i += 1

        if len(images) >= 12:
            break

    cap.release()
    cv.destroyAllWindows()

    print(f"Selected {len(images)} frames")
    print()

    # Create model

    print("Create model...")
    model = Model(images).to(device)
    print("Model created.")
    print()

    # plot_optical_flows(model.flows)
    with torch.no_grad():
        # plot_weights(model("weights"))
        # plot_depths(model("depths"))
        # plot_3d_points(model("points"))
        pass
    # __import__('sys').exit(0)

    print("Training model...")
    model = optimize_model(model)
    print("Training done.")

    plot_camera(model)

    with torch.no_grad():
        plot_weights(model("weights"))
        plot_depths(model("depths"))
        plot_3d_points(model("points"))

    __import__('sys').exit(0)
