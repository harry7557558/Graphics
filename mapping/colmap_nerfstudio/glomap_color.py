# Script for coloring point clouds for nerfstudio, especially those generated with GLOMAP

# To use: rename sparse_pc.ply to sparse_pc_raw.ply,
# then run this script in the same directory as nerfstudio dataset

import json
import numpy as np
import torch
import cv2
from plyfile import PlyData, PlyElement
from tqdm import tqdm

def load_transforms(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def load_ply(ply_path):
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    return torch.tensor(np.vstack([vertices['x'], vertices['y'], vertices['z']]).T, dtype=torch.float32).cuda()

def get_intrinsics(transforms):
    w, h = transforms['w'], transforms['h']
    K = torch.tensor([
        [transforms['fl_x'], 0, transforms['cx']],
        [0, transforms['fl_y'], transforms['cy']],
        [0, 0, 1]
    ], dtype=torch.float32).cuda()
    return w, h, K

def main(transforms_path, input_ply_path, output_ply_path):
    transforms = load_transforms(transforms_path)
    points = load_ply(input_ply_path)
    
    colors_val = torch.zeros((points.shape[0], 3), dtype=torch.float32).cuda()
    colors_depth = 1e10 * torch.ones((points.shape[0], 1), dtype=torch.float32).cuda()

    wg, hg, Kg = [None]*3
    if 'fl_x' in transforms:
        wg, hg, Kg = get_intrinsics(transforms)

    applied_transform = transforms['applied_transform']
    if len(applied_transform) < 4:
        applied_transform.append([0.0, 0.0, 0.0, 1.0])
    applied_transform = torch.tensor(applied_transform, dtype=torch.float32).cuda()

    transforms['frames'].sort(key=lambda _: _['file_path'])

    for frame in tqdm(transforms['frames']):
        image = cv2.imread(frame['file_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image).cuda().float()

        w, h, K = wg, hg, Kg
        if 'fl_x' in frame:
            w, h, K = get_intrinsics(frame)
        T = torch.tensor(frame['transform_matrix'], dtype=torch.float32).cuda()
        T = torch.linalg.inv(T)  # world to cam
        R = T[:3, :3]
        t = T[:3, 3:4]
        
        pts_cam = points @ R.T + t.T
        pts_cam[:, 2:] *= -1
        pts_2d = pts_cam / pts_cam[:, 2:]
        pts_2d = pts_2d @ K.T

        valid_mask = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] <= w-1) & \
                     (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] <= h-1) & \
                     (pts_cam[:, 2] > 0)
        
        valid_pts_2d = (pts_2d[valid_mask]+0.5).long()
        colors = image_tensor[h-1-valid_pts_2d[:, 1], valid_pts_2d[:, 0]]

        colors_val[valid_mask] = torch.where(
            (pts_cam[:, 2:] < colors_depth)[valid_mask],
            colors, colors_val[valid_mask])
        colors_depth[valid_mask] = torch.fmin(colors_depth, pts_cam[:, 2:])[valid_mask]

    mean_colors = colors_val.clamp(0, 255).byte().cpu().numpy()
    
    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertices = np.empty(points.shape[0], dtype=vertex_dtype)
    vertices['x'] = points.cpu().numpy()[:, 0]
    vertices['y'] = points.cpu().numpy()[:, 1]
    vertices['z'] = points.cpu().numpy()[:, 2]
    vertices['red'] = mean_colors[:, 0]
    vertices['green'] = mean_colors[:, 1]
    vertices['blue'] = mean_colors[:, 2]
    
    el = PlyElement.describe(vertices, 'vertex')
    PlyData([el]).write(output_ply_path)

if __name__ == "__main__":
    transforms_path = "transforms.json"
    input_ply_path = "sparse_pc_raw.ply"
    output_ply_path = "sparse_pc.ply"
    main(transforms_path, input_ply_path, output_ply_path)

