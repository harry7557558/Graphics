

- `flowmap_01`: simplified original FlowMap https://arxiv.org/abs/2404.15259, depth model takes long to converge
- `flowmap_02`: try to recover depth from optical flow using epipolar constraints, hardly differentiable
- `flowmap_03`: optimize poses for consistency of triangulated points based on optical flow, converges to poor results unless initial guess is close
- `flowmap_04`: simplified original FlowMap with a smaller depth predictor (based on UNet and transformer) trained from scratch, take optical flows as additional input for faster/better convergence
