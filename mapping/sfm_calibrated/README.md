SfM with images taken from the same calibrated camera, on ordered video frames.

`triangulation`: give 2 frames, estimate relative pose and feature point positions
 - `triangulation_01`: recover pose from essential matrix + triangulation, looks good

`ba_3`: give 3 frames, estimate relative poses and feature point positions
 - `ba_3_01`: estimate pairwise relative pose, minimize relative translation constraint; fail, translation becomes negative
 - `ba_3_02`: estimate relative pose for two frames, find pose for third frame using PnP; working

`ba_n`: give $n\ge3$ frames, estimate relative poses and feature point positions
 - `ba_n_01`: fail, optimization becomes unstable
 - `ba_n_02`: $O(N^2)$ pairwise feature matching, choose two frames for initial reconstruction, incrementally add more frames; largely working but fails on some scenes
 - `ba_n_03`: try autodiff BA Jacobian using Jax, doesn't make faster
 - `ba_n_04`: incrementally add frames in order after initial matches, run BA after adding each PnP point cloud; incorporate optical flow in feature matching; works on much longer sequences, fails on scenes with large rotation and coplanar geometry
 - `ba_n_05`: multi-frame optical flow feature tracking, skipped bundle adjustment; able to track long sequences of frames under good conditions, fails at large drift
 - `ba_n_06`: bundle adjustment with Ceres solver; outlier rejection

`feature`: explore better management of features
 - `feature_01`: using OpenCV RANSAC outlier rejection for matched features; performance isn't promising
 - `feature_02`: 
