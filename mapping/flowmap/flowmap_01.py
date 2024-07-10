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
    
    plt.show()


# Model


class Model(torch.nn.Module):

    def __init__(self,
                 images: List[np.ndarray]
            ) -> None:
        super().__init__()

        # depth model
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
        f = 0.75 * np.sqrt(self.width * self.height)
        self.camera_f = torch.nn.Parameter(Tensor([f, f]).float())  # fx, fy
        self.camera_c = torch.nn.Parameter(Tensor([0.5*self.width, 0.5*self.height]).float())  # cx, cy
        self.camera_dist = torch.nn.Parameter(torch.zeros(4).float())  # k1, k2, p1, p2

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
            mask: Float[Tensor, "h w"]
            ) -> Float[Tensor, "4 4"]:
        _, h, w = flow12.shape

        y, x = torch.meshgrid(
            torch.arange(h, device=device) + 0.5,
            torch.arange(w, device=device) + 0.5,
            indexing="ij")
        grid1 = torch.stack([x, y], dim=0)
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

        self.depths = torch.concat([
            self.depth_model(self.images_transformed[i:i+1])
            for i in range(self.n_images)
        ], dim=0)
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
    
    plt.show()


@torch.no_grad()
def plot_3d_points(points: List[Float[Tensor, "batch h w 3"]], fn=4567):

    n = int(np.ceil(np.sqrt(len(points))))
    m = int(np.ceil(len(points) / n))
    fig, axes = plt.subplots(m, n, figsize=(12, 12),
                             subplot_kw=dict(projection='3d'))
    axes = axes.flatten()

    for i in range(len(points)):
        tensor = points[i].reshape(-1, 3).cpu().numpy()
        tensor = tensor[(np.arange(fn) * len(tensor) / fn).astype(np.int32)]
        axes[i].scatter(tensor[:, 0], tensor[:, 2], tensor[:, 1],
                        s=0.5, c=tensor[:, 2], cmap='gnuplot')
        setup_3d_axis(axes[i])

    for j in range(i+1, n*m):
        fig.delaxes(axes[j])
    
    plt.show()


# Training

def optimize_model(model, flows, num_epochs=100):

    depth_params = list(model.depth_model_raw.parameters())

    optimizer_depth = torch.optim.Adam(depth_params, lr=1e-4)
    optimizer_camera_f = torch.optim.Adam([model.camera_f], lr=1e-1)
    optimizer_camera_c = torch.optim.Adam([model.camera_c], lr=1e-1)
    optimizer_camera_dist = torch.optim.Adam([model.camera_dist], lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        optimizer_depth.zero_grad()
        optimizer_camera_f.zero_grad()
        optimizer_camera_c.zero_grad()
        optimizer_camera_dist.zero_grad()

        loss = model(flows)

        loss.backward()
        optimizer_depth.step()
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
def plot_camera(model: Model, flows, sc=0.05):
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

        if len(all_images) >= 12:
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
        output = model(all_flows)
    # plot_depths(output)
    # plot_3d_points(output)

    print("Training model...")
    model = optimize_model(model, all_flows)
    print("Training done.")

    plot_camera(model, all_flows)

    with torch.no_grad():
        output = model(all_flows, returnv="points")
    plot_3d_points(output)

    __import__('sys').exit(0)
